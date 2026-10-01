#!/usr/bin/env python3
"""
build_nwp_calibration.py - turn the downloaded NWP verification data into the
per-city calibration files used by weather/nwp_error_model.py.

Input : the unzipped nwp_extracts folder (from fetch_nwp_data.py)
Output: weather/nwp/calibration/<city>.npz  (+ <city>_summary.json, readable)

    python weather/nwp/build_nwp_calibration.py --data <nwp_extracts> \
        --out weather/nwp/calibration

Forecasts: ECMWF IFS HRES 9 km (Open-Meteo Single Runs API), 00/12 UTC runs,
hourly values as served, lead 0..192 h.

What is stored per city
-----------------------
1. tqw_errors [run, lead 0..192, 3]: the real forecast errors (forecast minus
   reference) of temperature (C), relative humidity (%) and wind speed (m/s)
   of every run. The degradation replays one whole run (same season, same
   initialisation hour), so biases, growth with lead time, hour-to-hour
   persistence and the links between the three variables are those of the
   real forecasts. Gaps of up to 3 h in the reference are interpolated along
   the lead time; longer gaps stay NaN (such runs are not drawn).
   Reference ("truth") = the source the city's training covariates come from:
     seoul      : NOAA ISD station 47108 (= KMA ASOS 108, the station of the
                  Seoul bike data), Mar 2024 - Aug 2025
     london, washington : ERA5 / ERA5-Land (Open-Meteo archive era5_seamless),
                  the source of their covariates (Open-Meteo archive)
2. Solar radiation (reference ERA5): per lead, the heteroscedastic error model
   error = obs * (b + s * z): b and s by weighted least squares over daylight
   hours, z standard normal with lag-1 correlation phi (AR(1)).
3. Precipitation (reference ERA5; wet = >= 0.1 mm/h): per lead (pooled over
   +/-12 lead hours) miss rate and false-alarm ratio; for hits the mean and SD
   of log(forecast/observed); amounts of false alarms (lognormal, all leads);
   persistence of misses and of false alarms from one hour to the next
   (tetrachoric correlation of consecutive hours) and of the hit log ratio.
4. Visibility (reference: the station; capped at the city's data cap C):
   two parts. Observed at or above C: probability that the forecast is below
   C and, when it is, the depth -log(forecast/C) (lognormal). Observed below C:
   mean and SD of log(forecast/observed) (the degradation cuts the result at
   C once). Lag-1 persistence phi.

Lead-hour statistics use leads 1..89 for persistence (hourly native model
output; beyond 90 h Open-Meteo interpolates and inflates hour-to-hour
correlation).
"""

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_nwp_errors as A  # noqa: E402

MAX_LEAD = 192
PERSIST_LEADS = (1, 89)
WET_MM = 0.1 - 1e-9          # wet = >= 0.1 mm/h (data resolution 0.1 mm)
POOL = 12                    # +/- lead hours pooled for precipitation rates
TQW = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
REF_TQW = {"seoul": "isd", "london": "era5s", "washington": "era5s"}
# Visibility cap of each city's bike-data covariate (km); None = no cap.
# Checked on the bike data (1 Oct 2026): Seoul 2000 x 10 m = 20 km (26 % of
# hours), Washington 16.0 km (80 % of hours), London no cap (max 65 km).
VIS_CAP_DEFAULT = {"seoul": 20.0, "london": None, "washington": 16.0}


def lead_matrix(df, value_col, runs):
    """long (init, lead, value) -> [run, lead 0..MAX_LEAD] array."""
    w = df.pivot_table(index="init", columns="lead", values=value_col, dropna=False)
    w = w.reindex(index=runs, columns=range(0, MAX_LEAD + 1))
    return w


def tetrachoric(pairs_a, pairs_b):
    """Latent normal correlation of two binary series (consecutive hours)."""
    a = np.asarray(pairs_a, bool)
    b = np.asarray(pairs_b, bool)
    if len(a) < 200 or a.mean() in (0, 1) or b.mean() in (0, 1):
        return 0.0, len(a)
    p11 = np.mean(a & b)
    qa, qb = stats.norm.ppf(a.mean()), stats.norm.ppf(b.mean())

    def f(r):
        return stats.multivariate_normal.cdf([qa, qb], mean=[0, 0],
                                             cov=[[1, r], [r, 1]]) - p11
    lo, hi = -0.99, 0.99
    if f(lo) * f(hi) > 0:
        return (0.99 if f(hi) < 0 else -0.99), len(a)
    return float(optimize.brentq(f, lo, hi, xtol=1e-4)), len(a)


def ar1_from_runs(mat, valid_mask=None):
    """Lag-1 correlation of a [run, lead] matrix over PERSIST_LEADS."""
    l0, l1 = PERSIST_LEADS
    x = mat[:, l0:l1]
    y = mat[:, l0 + 1:l1 + 1]
    m = np.isfinite(x) & np.isfinite(y)
    if valid_mask is not None:
        m &= valid_mask[:, l0:l1] & valid_mask[:, l0 + 1:l1 + 1]
    if m.sum() < 200:
        return 0.0, int(m.sum())
    return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())


def interp_leads(v):
    """Fill NaN entries of a per-lead array (lead 0..MAX_LEAD) by linear interpolation."""
    s = pd.Series(v, dtype=float)
    s.iloc[0] = np.nan
    s = s.interpolate(limit_direction="both")
    return s.to_numpy()


def build_city(city, fc, refs, vis_cap):
    lat, lon = A.CITIES[city]["lat"], A.CITIES[city]["lon"]
    f = fc[(fc.city == city) & (fc.lead.between(0, MAX_LEAD))].copy()
    runs = pd.DatetimeIndex(sorted(f.init.unique()))
    out, summary = {}, {"city": city, "n_runs_total": int(len(runs))}

    # ---------------- 1. temperature / humidity / wind: error trajectories
    ref = refs[REF_TQW[city]]
    errs = []
    for v in TQW:
        e = f[["init", "lead", "valid", v]].copy()
        e["err"] = e[v].values - ref[v].reindex(e["valid"]).values
        w = lead_matrix(e, "err", runs)
        w = w.T.interpolate(limit=3, limit_area="inside").T     # gaps <= 3 h
        errs.append(w.to_numpy(dtype=np.float32))
    tqw = np.stack(errs, axis=-1)                                  # [run, lead, 3]
    out["run_init_utc"] = runs.values.astype("datetime64[h]")
    out["tqw_errors"] = tqw
    out["tqw_variables"] = np.array(["temperature", "humidity", "wind_speed"])
    complete = np.isfinite(tqw[:, 1:MAX_LEAD + 1, :]).all(axis=(1, 2))
    summary["tqw_reference"] = A.OBS_NAMES[REF_TQW[city]].format(station=A.CITIES[city]["isd"])
    summary["tqw_runs_complete_1_192h"] = int(complete.sum())
    summary["tqw_check_24h"] = {name: dict(bias=round(float(np.nanmean(tqw[:, 24, k])), 3),
                                          sd=round(float(np.nanstd(tqw[:, 24, k], ddof=1)), 3))
                                for k, name in enumerate(out["tqw_variables"])}

    # ---------------- 2. solar radiation (ERA5 reference)
    era = refs["era5s"]
    s = f[["init", "lead", "valid", "shortwave_radiation"]].rename(columns={"shortwave_radiation": "fc"})
    s["ob"] = era["shortwave_radiation"].reindex(s["valid"]).values
    s["day"] = A.solar_elevation(s["valid"] - pd.Timedelta(minutes=30), lat, lon) > 0
    s = s[s.day & (s.ob > 0) & s.fc.notna()]
    b = np.full(MAX_LEAD + 1, np.nan)
    sd = np.full(MAX_LEAD + 1, np.nan)
    relmae = np.full(MAX_LEAD + 1, np.nan)
    for L, d in s.groupby("lead"):
        if L < 1 or len(d) < 50:
            continue
        o, e = d.ob.to_numpy(), (d.fc - d.ob).to_numpy()
        b[L] = np.sum(e * o) / np.sum(o * o)
        sd[L] = np.sqrt(np.sum((e - b[L] * o) ** 2) / np.sum(o * o))
        relmae[L] = np.abs(e).sum() / o.sum()
    out["solar_bias_rel"], out["solar_sd_rel"] = interp_leads(b), interp_leads(sd)
    out["solar_relmae_measured"] = relmae
    s["r"] = ((s.fc - s.ob) - out["solar_bias_rel"][s.lead.to_numpy()] * s.ob) / s.ob
    s.loc[s.ob < 50, "r"] = np.nan
    rmat = lead_matrix(s, "r", runs).to_numpy()
    out["solar_phi"], n = ar1_from_runs(rmat)
    summary["solar"] = dict(reference=A.OBS_NAMES["era5s"], phi=round(out["solar_phi"], 3), n_pairs=n,
                            bias_rel_24h=round(float(out["solar_bias_rel"][24]), 3),
                            sd_rel_24h=round(float(out["solar_sd_rel"][24]), 3),
                            relmae_measured_24h=round(float(relmae[24]), 3))

    # ---------------- 3. precipitation (ERA5 reference)
    p = f[["init", "lead", "valid", "precipitation"]].rename(columns={"precipitation": "fc"})
    p["ob"] = era["precipitation"].reindex(p["valid"]).values
    p = p.dropna(subset=["fc", "ob"])
    p["ow"], p["fw"] = p.ob > WET_MM, p.fc > WET_MM
    g = p.groupby("lead")
    H = g.apply(lambda d: int((d.ow & d.fw).sum()))
    M = g.apply(lambda d: int((d.ow & ~d.fw).sum()))
    F = g.apply(lambda d: int((~d.ow & d.fw).sum()))
    D = g.apply(lambda d: int((~d.ow).sum()))
    hits = p[p.ow & p.fw].copy()
    hits["lr"] = np.log(hits.fc / hits.ob)
    miss = np.full(MAX_LEAD + 1, np.nan)
    far = np.full(MAX_LEAD + 1, np.nan)
    pofd = np.full(MAX_LEAD + 1, np.nan)
    hm = np.full(MAX_LEAD + 1, np.nan)
    hs = np.full(MAX_LEAD + 1, np.nan)
    for L in range(1, MAX_LEAD + 1):
        win = [l for l in range(L - POOL, L + POOL + 1) if 1 <= l <= MAX_LEAD and l in H.index]
        h_, m_, f_, d_ = H[win].sum(), M[win].sum(), F[win].sum(), D[win].sum()
        miss[L] = m_ / (h_ + m_)
        far[L] = f_ / (h_ + f_)
        pofd[L] = f_ / d_
        lr = hits.lr[hits.lead.isin(win)]
        hm[L], hs[L] = lr.mean(), lr.std(ddof=1)
    out["precip_miss_rate"], out["precip_far"] = miss, far
    out["precip_pofd_measured"] = pofd
    out["precip_hit_mean_log"], out["precip_hit_sd_log"] = hm, hs
    fa = p[~p.ow & p.fw]
    out["precip_fa_mean_log"] = float(np.log(fa.fc).mean())
    out["precip_fa_sd_log"] = float(np.log(fa.fc).std(ddof=1))
    out["precip_wet_share_ref"] = float(p.ow.mean())
    # persistence: consecutive hours of the same run
    ow = lead_matrix(p.assign(x=p.ow.astype(float)), "x", runs).to_numpy()
    fw = lead_matrix(p.assign(x=p.fw.astype(float)), "x", runs).to_numpy()
    l0, l1 = PERSIST_LEADS
    a_ow, b_ow = ow[:, l0:l1], ow[:, l0 + 1:l1 + 1]
    a_fw, b_fw = fw[:, l0:l1], fw[:, l0 + 1:l1 + 1]
    ok = np.isfinite(a_ow) & np.isfinite(b_ow) & np.isfinite(a_fw) & np.isfinite(b_fw)
    wetwet = ok & (a_ow == 1) & (b_ow == 1)
    drydry = ok & (a_ow == 0) & (b_ow == 0)
    out["precip_rho_miss"], n_ww = tetrachoric(a_fw[wetwet] == 0, b_fw[wetwet] == 0)
    out["precip_rho_fa"], n_dd = tetrachoric(a_fw[drydry] == 1, b_fw[drydry] == 1)
    lrm = lead_matrix(hits, "lr", runs).to_numpy()
    out["precip_hit_phi"], n_hh = ar1_from_runs(lrm)
    summary["precipitation"] = dict(
        reference=A.OBS_NAMES["era5s"], wet_threshold_mm=0.1,
        wet_share_ref=round(out["precip_wet_share_ref"], 3),
        miss_24h=round(float(miss[24]), 3), far_24h=round(float(far[24]), 3),
        miss_168h=round(float(miss[168]), 3), far_168h=round(float(far[168]), 3),
        hit_mean_log_24h=round(float(hm[24]), 3), hit_sd_log_24h=round(float(hs[24]), 3),
        fa_amount_median_mm=round(float(np.exp(out["precip_fa_mean_log"])), 3),
        fa_amount_sd_log=round(out["precip_fa_sd_log"], 3),
        rho_miss=round(out["precip_rho_miss"], 3), n_wetwet=n_ww,
        rho_false_alarm=round(out["precip_rho_fa"], 3), n_drydry=n_dd,
        hit_phi=round(out["precip_hit_phi"], 3), n_hithit=n_hh)

    # ---------------- 4. visibility (station reference)
    isd = refs["isd"]
    v = f[["init", "lead", "valid", "visibility"]].rename(columns={"visibility": "fc"})
    v["ob"] = isd["visibility"].reindex(v["valid"]).values
    v["ob_cap"] = isd["vis_cap"].reindex(v["valid"]).values
    v = v.dropna(subset=["fc", "ob"])
    v = v[(v.fc > 0) & (v.ob > 0)]
    C = vis_cap
    if C is None:
        # no cap in the covariate: use only uncensored station reports
        v = v[~(v.ob >= v.ob_cap.fillna(np.inf) - 1e-6)]
        at = pd.Series(False, index=v.index)
    else:
        # censored station reports above the city cap count as "at or above C"
        at = v.ob >= C - 1e-6
        below_own_cap = ~(v.ob >= v.ob_cap.fillna(np.inf) - 1e-6)
        v = v[at | below_own_cap]
        at = at[v.index]
    vb = v[~at]
    # forecast not cut at the cap here: the degradation applies the cap once
    lr_b = np.log(vb.fc / vb.ob)
    out["vis_cap_km"] = np.nan if C is None else float(C)
    out["vis_below_mean_log"], out["vis_below_sd_log"] = float(lr_b.mean()), float(lr_b.std(ddof=1))
    if C is not None:
        va = v[at]
        out["vis_at_cap_p_below"] = float((va.fc < C).mean())
        # depth below the cap, -log(forecast / C) > 0, is lognormal
        depth = -np.log(va.fc[va.fc < C] / C)
        ld = np.log(depth[depth > 0])
        out["vis_at_cap_depth_mean_log"] = float(ld.mean())
        out["vis_at_cap_depth_sd_log"] = float(ld.std(ddof=1))
        out["vis_at_cap_median_ratio"] = float(np.exp(-np.median(depth)))
    else:
        out["vis_at_cap_p_below"] = out["vis_at_cap_depth_mean_log"] = np.nan
        out["vis_at_cap_depth_sd_log"] = out["vis_at_cap_median_ratio"] = np.nan
    # persistence of the log error (both variants together, capped at C)
    cap_ = C if C else np.inf
    v["lr"] = np.log(np.minimum(v.fc, cap_) / np.minimum(v.ob, cap_))
    out["vis_phi"], n_v = ar1_from_runs(lead_matrix(v, "lr", runs).to_numpy())
    summary["visibility"] = dict(
        reference=A.OBS_NAMES["isd"].format(station=A.CITIES[city]["isd"]),
        cap_km=C, share_obs_at_cap=round(float(at.mean()), 3) if C else 0.0,
        p_forecast_below_cap_when_obs_at_cap=None if C is None else round(out["vis_at_cap_p_below"], 3),
        median_forecast_over_cap_when_below=None if C is None else round(out["vis_at_cap_median_ratio"], 3),
        below_cap_mean_log=round(out["vis_below_mean_log"], 3),
        below_cap_sd_log=round(out["vis_below_sd_log"], 3),
        phi=round(out["vis_phi"], 3), n_pairs=n_v)
    return out, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="unzipped nwp_extracts folder")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "calibration"))
    ap.add_argument("--vis-cap", nargs=2, action="append", metavar=("CITY", "KM"),
                    help="visibility cap of a city's covariate in km ('none' = no cap)")
    a = ap.parse_args()
    caps = dict(VIS_CAP_DEFAULT)
    for city, km in (a.vis_cap or []):
        caps[city] = None if km.lower() == "none" else float(km)
    os.makedirs(a.out, exist_ok=True)
    fc = A.load_single_runs(a.data)
    for city in A.CITIES:
        isd_h, _ = A.load_isd(a.data, city)
        refs = {"isd": isd_h, "era5s": A.load_archive(a.data, city, "recent_era5seamless")}
        out, summary = build_city(city, fc, refs, caps[city])
        summary.update(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                       forecast_source=A.FC_ECMWF_SERVED,
                       run_period=f"{pd.Timestamp(out['run_init_utc'][0])} .. "
                                  f"{pd.Timestamp(out['run_init_utc'][-1])}")
        out["summary_json"] = np.array(json.dumps(summary))
        np.savez_compressed(os.path.join(a.out, f"{city}.npz"), **out)
        with open(os.path.join(a.out, f"{city}_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=1)
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
