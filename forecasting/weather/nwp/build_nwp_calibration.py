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
3. Precipitation. Reference: Seoul = the station (Apr-Oct hourly; Nov-Mar KMA
   3-hour totals split evenly over their three covered hours by load_isd(),
   as in the bike data); London, Washington = ERA5, the source of their
   covariates. Seoul Nov-Mar (Korean time): the forecast is split the same
   way (sum of the run's three hourly values per KMA window / 3), and an
   hour is wet if its 3-hour total is >= 0.1 mm, for station and forecast
   alike. Otherwise wet = >= 0.1 mm/h. Per lead
   (pooled over +/-12 lead hours) miss rate and false-alarm ratio; for hits
   the mean and SD of log(forecast/observed). Miss rate is additionally
   stored for two observed-intensity classes: light < 1 mm/h and stronger
   >= 1 mm/h. These values are stored year-round and per time of year (12
   bins centred on the 15th of each month). Seasonal windows start at +/-
   PRECIP_SEASON_DAYS and widen in 15-day steps; the two intensity classes
   widen independently until every pooled lead has >= PRECIP_MIN_EVENTS wet
   observations. FAR and hit amount error keep the existing overall windows.
   Year-round only: amounts of false alarms (lognormal, all leads);
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
WET_MM_SPLIT = 0.1 / 3 - 1e-9  # Seoul Nov-Mar split hours: 3-hour total >= 0.1 mm
INTENSITY_SPLIT_MM = 1.0      # light < 1 mm/h; stronger >= 1 mm/h
POOL = 12                    # +/- lead hours pooled for precipitation rates
TQW = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
REF_TQW = {"seoul": "isd", "london": "era5s", "washington": "era5s"}
# Precipitation reference: Seoul = the station (its bike data are this
# station's rain; checked 2 Oct 2026), London/Washington = ERA5
REF_PRECIP = {"seoul": "isd", "london": "era5s", "washington": "era5s"}
PRECIP_SEASON_DAYS = 30         # +/- days of year around each monthly centre
PRECIP_MIN_EVENTS = 200         # wet (and forecast-wet) hours per pooled lead window
PRECIP_MIN_HITS = 50            # hits per pooled lead window for the amount error
SEASON_CENTRES = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])
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


def precip_counts(p):
    """Per-lead hits, misses, false alarms, dry hours of a precipitation table."""
    g = p.groupby("lead")
    idx = range(0, MAX_LEAD + 1)
    H = g.apply(lambda d: int((d.ow & d.fw).sum())).reindex(idx, fill_value=0)
    M = g.apply(lambda d: int((d.ow & ~d.fw).sum())).reindex(idx, fill_value=0)
    F = g.apply(lambda d: int((~d.ow & d.fw).sum())).reindex(idx, fill_value=0)
    D = g.apply(lambda d: int((~d.ow).sum())).reindex(idx, fill_value=0)
    return H, M, F, D


def precip_rates(p):
    """Per lead 1..MAX_LEAD (pooled over +/-POOL lead hours): miss rate,
    false-alarm ratio, POFD, mean / SD of log(forecast/observed) of hits, and
    the smallest pooled numbers of wet hours, forecast-wet hours and hits."""
    H, M, F, D = precip_counts(p)
    hits = p[p.ow & p.fw]
    lr_by_lead = {L: v.to_numpy() for L, v in np.log(hits.fc / hits.ob).groupby(hits.lead)}
    miss, far, pofd, hm, hs = (np.full(MAX_LEAD + 1, np.nan) for _ in range(5))
    n_wet, n_fwet, n_hit = [], [], []
    for L in range(1, MAX_LEAD + 1):
        win = list(range(max(1, L - POOL), min(MAX_LEAD, L + POOL) + 1))
        h_, m_, f_, d_ = H[win].sum(), M[win].sum(), F[win].sum(), D[win].sum()
        miss[L] = m_ / (h_ + m_) if h_ + m_ else np.nan
        far[L] = f_ / (h_ + f_) if h_ + f_ else np.nan
        pofd[L] = f_ / d_ if d_ else np.nan
        lr = np.concatenate([lr_by_lead.get(l, np.empty(0)) for l in win])
        if len(lr) > 1:
            hm[L], hs[L] = lr.mean(), lr.std(ddof=1)
        if L <= 168:
            n_wet.append(h_ + m_); n_fwet.append(h_ + f_); n_hit.append(h_)
    return miss, far, pofd, hm, hs, dict(min_wet=int(min(n_wet)), min_fc_wet=int(min(n_fwet)),
                                          min_hits=int(min(n_hit)))


def precip_miss_only(p):
    """Miss rate for a table containing observed wet hours of one class.

    Returns the pooled per-lead miss rate and the smallest pooled number of
    observed wet hours over leads 1..168.
    """
    lead = p["lead"].to_numpy(dtype=int)
    fw = p["fw"].to_numpy(dtype=bool)
    H = np.bincount(lead, weights=fw.astype(int), minlength=MAX_LEAD + 1)
    M = np.bincount(lead, weights=(~fw).astype(int), minlength=MAX_LEAD + 1)
    miss = np.full(MAX_LEAD + 1, np.nan)
    n_wet = []
    for L in range(1, MAX_LEAD + 1):
        win = list(range(max(1, L - POOL), min(MAX_LEAD, L + POOL) + 1))
        h_, m_ = H[win].sum(), M[win].sum()
        miss[L] = m_ / (h_ + m_) if h_ + m_ else np.nan
        if L <= 168:
            n_wet.append(h_ + m_)
    return miss, int(min(n_wet))


def intensity_mask(p, class_idx):
    """Observed-wet intensity class: 0 = light, 1 = stronger."""
    if class_idx == 0:
        return p.ow & (p.ob < INTENSITY_SPLIT_MM)
    if class_idx == 1:
        return p.ow & (p.ob >= INTENSITY_SPLIT_MM)
    raise ValueError(class_idx)


def intensity_miss_rates(p):
    """Year-round miss rates [2, lead] and pooled count diagnostics."""
    arr = np.full((2, MAX_LEAD + 1), np.nan)
    counts = []
    shares = []
    n_all = int(p.ow.sum())
    for k in range(2):
        q = p[intensity_mask(p, k)]
        miss, n = precip_miss_only(q)
        arr[k] = interp_leads(miss)
        counts.append(int(n))
        shares.append(float(len(q) / n_all) if n_all else np.nan)
    return arr, counts, shares


def seasonal_intensity_miss(p):
    """Intensity-specific seasonal miss rates.

    Each of the two classes gets its own seasonal window, widened in 15-day
    steps until every pooled lead window through 168 h has at least
    PRECIP_MIN_EVENTS observed wet cases (or the circular half-year limit is
    reached, matching seasonal_precip()).
    """
    doy = pd.DatetimeIndex(p["init"]).dayofyear.to_numpy()
    arr = np.full((len(SEASON_CENTRES), 2, MAX_LEAD + 1), np.nan)
    days_used = np.zeros((len(SEASON_CENTRES), 2), dtype=int)
    min_counts = np.zeros((len(SEASON_CENTRES), 2), dtype=int)
    class_shares = np.full((len(SEASON_CENTRES), 2), np.nan)
    for m, c in enumerate(SEASON_CENTRES):
        d = np.abs(doy - c)
        d = np.minimum(d, 366 - d)
        for k in range(2):
            days = PRECIP_SEASON_DAYS
            while True:
                win_p = p[d <= days]
                q = win_p[intensity_mask(win_p, k)]
                miss, n = precip_miss_only(q)
                if n >= PRECIP_MIN_EVENTS or days >= 183:
                    break
                days += 15
            arr[m, k] = interp_leads(miss)
            days_used[m, k] = int(days)
            min_counts[m, k] = int(n)
            n_wet = int(win_p.ow.sum())
            class_shares[m, k] = float(len(q) / n_wet) if n_wet else np.nan
    return arr, days_used, min_counts, class_shares


def precipitation_wet_masks(p, city):
    """Observed/forecast wet flags, the same rule for both.

    Seoul hours whose KMA 3-hour window ends in Nov-Mar (Korean time): station
    and forecast are split 3-hour totals (load_isd(),
    analyze_nwp_errors.split_seoul_winter_forecast()); wet = 3-hour total
    >= 0.1 mm, i.e. split value >= 0.1/3 mm/h (the station's split values are
    multiples of 0.1/3, so this is every positive value). Elsewhere wet =
    >= 0.1 mm/h (data resolution 0.1 mm).
    """
    winter = np.zeros(len(p), dtype=bool)
    if city == "seoul":
        winter = A.seoul_winter(p["valid"])
    thr = np.where(winter, WET_MM_SPLIT, WET_MM)
    ow = p.ob.to_numpy() > thr
    fw = p.fc.to_numpy() > thr
    return ow.astype(bool), fw.astype(bool)


def seasonal_precip(p):
    """Miss rate, false-alarm ratio and hit log-ratio mean / SD per time of
    year: 12 bins centred on SEASON_CENTRES (day of year of the run start),
    window +/- PRECIP_SEASON_DAYS, widened by 15 days until the counts in
    every pooled lead window (leads 1-168) reach PRECIP_MIN_EVENTS (wet and
    forecast-wet hours) and PRECIP_MIN_HITS (hits)."""
    doy = pd.DatetimeIndex(p["init"]).dayofyear.to_numpy()
    arrs = {k: np.full((len(SEASON_CENTRES), MAX_LEAD + 1), np.nan)
            for k in ("miss", "far", "hm", "hs")}
    days_used, counts = [], []
    for m, c in enumerate(SEASON_CENTRES):
        d = np.abs(doy - c)
        d = np.minimum(d, 366 - d)
        days = PRECIP_SEASON_DAYS
        while True:
            miss, far, _, hm, hs, n = precip_rates(p[d <= days])
            ok = (n["min_wet"] >= PRECIP_MIN_EVENTS and n["min_fc_wet"] >= PRECIP_MIN_EVENTS
                  and n["min_hits"] >= PRECIP_MIN_HITS)
            if ok or days >= 183:
                break
            days += 15
        for k, v in zip(("miss", "far", "hm", "hs"), (miss, far, hm, hs)):
            arrs[k][m] = interp_leads(v)
        days_used.append(int(days)); counts.append(n)
    return arrs, days_used, counts


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

    # ---------------- 3. precipitation (reference REF_PRECIP)
    pref = refs[REF_PRECIP[city]]
    p = f[["init", "lead", "valid", "precipitation"]].rename(columns={"precipitation": "fc"})
    p["ob"] = pref["precipitation"].reindex(p["valid"]).values
    if city == "seoul":
        # station Nov-Mar values are split 3-hour totals: split the forecast
        # over the same windows (like for like, as in the bike data)
        p = A.split_seoul_winter_forecast(p)
    p = p.dropna(subset=["fc", "ob"])
    p["ow"], p["fw"] = precipitation_wet_masks(p, city)
    miss, far, pofd, hm, hs, n_year = precip_rates(p)
    hits = p[p.ow & p.fw].copy()
    hits["lr"] = np.log(hits.fc / hits.ob)
    out["precip_miss_rate"], out["precip_far"] = interp_leads(miss), interp_leads(far)
    out["precip_pofd_measured"] = pofd
    out["precip_hit_mean_log"], out["precip_hit_sd_log"] = interp_leads(hm), interp_leads(hs)
    imiss, imiss_n, intensity_share = intensity_miss_rates(p)
    out["precip_intensity_split_mm"] = np.array(INTENSITY_SPLIT_MM)
    out["precip_miss_rate_intensity"] = imiss
    out["precip_intensity_wet_share"] = np.asarray(intensity_share)
    seas, seas_days, seas_n = seasonal_precip(p)
    out["precip_season_doy"] = SEASON_CENTRES
    out["precip_season_window_days"] = np.array(seas_days)
    out["precip_miss_rate_seasonal"] = seas["miss"]
    out["precip_far_seasonal"] = seas["far"]
    out["precip_hit_mean_log_seasonal"] = seas["hm"]
    out["precip_hit_sd_log_seasonal"] = seas["hs"]
    imiss_seas, imiss_days, imiss_counts, intensity_share_seas = seasonal_intensity_miss(p)
    out["precip_miss_rate_intensity_seasonal"] = imiss_seas
    out["precip_intensity_season_window_days"] = imiss_days
    out["precip_intensity_min_pooled_wet"] = imiss_counts
    out["precip_intensity_wet_share_seasonal"] = intensity_share_seas
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
        reference=A.OBS_NAMES[REF_PRECIP[city]].format(station=A.CITIES[city]["isd"]),
        wet_threshold_mm=("Nov-Mar (KST): station and forecast split over KMA 3-hour "
                          "windows, wet = 3-hour total >=0.1; otherwise >=0.1 per hour"
                          if city == "seoul" else 0.1),
        intensity_classes_mm=["wet and <1", ">=1"],
        intensity_min_pooled_counts_year_round=imiss_n,
        intensity_wet_share_year_round=[round(float(v), 3) for v in intensity_share],
        intensity_miss_24h=[round(float(v), 3) for v in imiss[:, 24]],
        intensity_miss_168h=[round(float(v), 3) for v in imiss[:, 168]],
        intensity_seasonal=dict(
            window_days=imiss_days.tolist(),
            min_pooled_wet=imiss_counts.tolist(),
            miss_24h=[[round(float(v), 3) for v in row]
                      for row in imiss_seas[:, :, 24]],
        ),
        n_hours=int(len(p)),
        runs_used=int(p["init"].nunique()),
        min_pooled_counts_year_round=n_year,
        seasonal=dict(window_days=seas_days,
                      miss_24h=[round(float(v), 3) for v in seas["miss"][:, 24]],
                      far_24h=[round(float(v), 3) for v in seas["far"][:, 24]],
                      min_pooled_counts=seas_n),
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
