#!/usr/bin/env python3
"""
analyze_nwp_errors.py - forecast-error statistics for the bike-demand paper.

Input : the folder produced by unzipping nwp_extracts.zip (from fetch_nwp_data.py)
Output: <out>/nwp_error_statistics.csv   (all statistics + straight-line fits)
        <out>/diagnostics.json            (coverage, grid points, caps, choices)

    python analyze_nwp_errors.py --data nwp_extracts --out results

Requires: numpy, pandas.

Forecast sources
  ECMWF  : ECMWF IFS HRES 9 km, Open-Meteo Single Runs API, 00/12 UTC runs.
           Two variants:
             "as served"  - Open-Meteo's hourly series at every lead 1-168 h
                            (beyond 90 h ECMWF only outputs 3-/6-hourly; the
                            hours in between are Open-Meteo interpolation).
             "native"     - leads 1-90 h hourly, 93-144 h at 3-h steps/windows,
                            150-168 h at 6-h steps/windows (pure model output).
  GEFS   : NOAA GEFSv12 reforecast control run, 00 UTC, 3-hourly steps.
References ("observations")
  ISD    : NOAA ISD station reports (T, RH from dew point, wind, visibility,
           precipitation where the station reports it).
  ERA5s  : Open-Meteo archive, model era5_seamless (ERA5-Land T/RH + ERA5).
  IFSbm  : Open-Meteo archive, best_match (ECMWF IFS analysis based, 2017+).

Error = forecast - reference. Nothing is imputed; every statistic carries its
sample size. Statistics that cannot be computed are simply absent (and the
reason is listed in diagnostics.json).
"""

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

CITIES = {
    "seoul": dict(lat=37.5714, lon=126.9658, isd="47108099999"),
    "london": dict(lat=51.4790, lon=-0.4490, isd="03772099999"),
    "washington": dict(lat=38.8483, lon=-77.0342, isd="72405013743"),
}
VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
        "shortwave_radiation", "precipitation", "visibility"]
INSTANT = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "visibility"]
LAGS = [1, 3, 6, 12, 24]
MAX_LEAD = 168
WET_THRESHOLDS = {"gt0.1": 0.1, "gt0": 0.0}
ISD_OK_QC = set("014569ACIMPRU")          # accepted ISD quality codes
# Stations whose hourly SYNOP reports (FM-12) carry a 1-hour precipitation
# group in every wet hour and omit it in dry hours (checked 2 Oct 2026 for
# Seoul 47108, 2024-25: the hourly amounts add up to the station's running
# 24-h totals, median ratio 1.00; Seoul bike-data rain = this station's rain,
# 2017-18 6-/12-h totals match in 99 % of cases). A routine report without the
# group and without precipitation in the present-weather group is a dry hour.
SYNOP_HOURLY_PRECIP = {"seoul"}

FC_ECMWF_SERVED = ("ECMWF IFS HRES 9km (Open-Meteo Single Runs, 00/12 UTC runs; hourly as "
                   "served, leads >90 h interpolated by Open-Meteo from 3-/6-hourly output)")
FC_ECMWF_NATIVE = ("ECMWF IFS HRES 9km (Open-Meteo Single Runs, 00/12 UTC runs; native output "
                   "only: hourly to 90 h, 3-h steps/windows to 144 h, 6-h to 168 h)")
FC_GEFS = "NOAA GEFSv12 reforecast control member c00 (0.25 deg, 00 UTC runs, 3-hourly steps)"
OBS_NAMES = {
    "isd": "NOAA ISD station {station} (nearest report within +/-30 min)",
    "era5s": "Open-Meteo archive era5_seamless (ERA5-Land T/RH + ERA5), nearest grid cell",
    "ifsbm": "Open-Meteo archive best_match (ECMWF IFS analysis based), nearest grid cell",
    "userdata": "weather columns of the user's bike dataset",
}

DIAG = defaultdict(dict)


# ===========================================================================
# Helpers
# ===========================================================================
def solar_elevation(times, lat, lon):
    """Solar elevation (deg) for UTC timestamps (NOAA general solar position)."""
    t = pd.DatetimeIndex(times)
    doy = t.dayofyear.values.astype(float)
    hour = t.hour.values + t.minute.values / 60.0
    g = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
            - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = hour * 60 + eqt + 4 * lon
    ha = np.radians(tst / 4.0 - 180.0)
    la = np.radians(lat)
    cz = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    return 90.0 - np.degrees(np.arccos(np.clip(cz, -1, 1)))


def rh_from_t_td(t, td):
    """Relative humidity (%) over water, Magnus form (Alduchov & Eskridge 1996)."""
    a, b = 17.625, 243.04
    return 100.0 * np.exp(a * td / (b + td) - a * t / (b + t))


def rh_from_q(t_c, q, p_pa):
    """RH (%) from specific humidity (kg/kg), pressure (Pa), temperature (C)."""
    e = q * (p_pa / 100.0) / (0.622 + 0.378 * q)          # hPa
    es = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))      # hPa, Bolton 1980
    return 100.0 * e / es


def cv_from_log_sd(sd):
    """Coefficient of variation equivalent to a log-normal with log-SD sd."""
    return float(np.sqrt(np.expm1(sd ** 2))) if np.isfinite(sd) else np.nan


def linfit(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or np.ptp(x) == 0:
        return None
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return dict(a=float(a), b=float(b), r2=r2, rmse=math.sqrt(ss_res / len(x)),
                n_leads=len(x), lead_min=float(x.min()), lead_max=float(x.max()))


# ===========================================================================
# Loading
# ===========================================================================
def load_single_runs(root):
    """-> DataFrame city, init, lead, valid, <vars> (hourly, as served)."""
    frames = []
    grid = {}
    for city in CITIES:
        files = sorted(glob.glob(os.path.join(root, "raw", "openmeteo_single_runs", city,
                                              "[0-9]*.json")))
        files = [f for f in files if not f.endswith(".error.json")]
        errs = glob.glob(os.path.join(root, "raw", "openmeteo_single_runs", city, "*.error.json"))
        DIAG["ecmwf_runs"][city] = dict(n_runs=len(files), n_error_files=len(errs))
        for f in files:
            js = json.load(open(f))
            h = js.get("hourly", {})
            if "time" not in h:
                continue
            init = pd.to_datetime(os.path.basename(f)[:10], format="%Y%m%d%H")
            d = pd.DataFrame({v: pd.to_numeric(pd.Series(h.get(v, [None] * len(h["time"]))),
                                               errors="coerce") for v in VARS})
            d["valid"] = pd.to_datetime(h["time"])
            d["lead"] = ((d["valid"] - init) / pd.Timedelta(hours=1)).round().astype(int)
            d["init"] = init
            d["city"] = city
            frames.append(d)
            grid.setdefault(city, dict(lat=js.get("latitude"), lon=js.get("longitude"),
                                       elevation=js.get("elevation")))
    DIAG["ecmwf_grid_cell"] = grid
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["visibility"] = df["visibility"] / 1000.0              # m -> km
    for city in CITIES:
        sub = df[df.city == city]
        if len(sub):
            DIAG["ecmwf_runs"][city].update(
                first_run=str(sub.init.min()), last_run=str(sub.init.max()),
                nonnull_fraction={v: round(float(sub[v].notna().mean()), 4) for v in VARS})
    return df


def load_archive(root, city, tag_prefix):
    files = sorted(glob.glob(os.path.join(root, "raw", "openmeteo_archive", city,
                                          f"{tag_prefix}_*.json")))
    frames = []
    for f in files:
        js = json.load(open(f))
        h = js.get("hourly", {})
        if "time" not in h:
            continue
        d = pd.DataFrame({k: pd.to_numeric(pd.Series(v), errors="coerce")
                          for k, v in h.items() if k != "time"})
        d.index = pd.to_datetime(h["time"])
        frames.append(d)
        DIAG["archive_grid_cell"][f"{city}/{tag_prefix}"] = dict(
            lat=js.get("latitude"), lon=js.get("longitude"), elevation=js.get("elevation"))
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames).sort_index()
    return d[~d.index.duplicated(keep="first")]


# ---------------------------------------------------------------- ISD -----
def _num_qc(series, scale, missing):
    """ISD 'value,qc' strings -> float (value/scale); bad qc / missing -> NaN."""
    s = series.fillna("").astype(str).str.split(",", expand=True)
    if s.shape[1] < 2:
        return pd.Series(np.nan, index=series.index)
    v = pd.to_numeric(s[0], errors="coerce")
    ok = s[1].isin(list(ISD_OK_QC)) & (v != missing)
    return (v / scale).where(ok)


def _precip_groups(df):
    """All AA1..AA4 groups -> long table (report index, period_h, mm)."""
    out = []
    for col in ["AA1", "AA2", "AA3", "AA4"]:
        if col not in df.columns:
            continue
        s = df[col].fillna("").astype(str)
        s = s[s != ""]
        if s.empty:
            continue
        p = s.str.split(",", expand=True)
        per = pd.to_numeric(p[0], errors="coerce")
        amt = pd.to_numeric(p[1], errors="coerce")
        qc = p[3] if p.shape[1] > 3 else pd.Series("1", index=p.index)
        ok = (amt != 9999) & qc.isin(list(ISD_OK_QC)) & per.notna()
        out.append(pd.DataFrame({"row": s.index[ok], "period": per[ok].astype(int).values,
                                 "mm": (amt[ok] / 10.0).values}))
    return pd.concat(out) if out else pd.DataFrame(columns=["row", "period", "mm"])


def _present_weather_precip(df):
    """True where a present-weather group reports falling precipitation."""
    flag = pd.Series(False, index=df.index)
    for col in ["MW1", "MW2"]:                       # manual ww code 50-99
        if col in df.columns:
            ww = pd.to_numeric(df[col].fillna("").astype(str).str.split(",").str[0],
                               errors="coerce")
            flag |= ww.between(50, 99)
    for col in ["AW1", "AW2"]:                       # automated code 40-99 (precip)
        if col in df.columns:
            aw = pd.to_numeric(df[col].fillna("").astype(str).str.split(",").str[0],
                               errors="coerce")
            flag |= aw.between(40, 99)
    for col in ["AU1", "AU2"]:                       # ASOS: precipitation descriptor
        if col in df.columns:
            pr = df[col].fillna("").astype(str).str.split(",").str[2]
            flag |= pr.fillna("").isin([f"{i:02d}" for i in range(1, 10)])
    return flag


REPORT_RANK = {"FM-15": 0, "FM-12": 1, "SY-MT": 1, "SAO": 2, "AUTO": 2, "SY-SA": 2, "FM-16": 5}


def load_isd(root, city):
    st = CITIES[city]["isd"]
    files = sorted(glob.glob(os.path.join(root, "isd_trimmed", f"{st}_*.csv")))
    if not files:
        return pd.DataFrame(), pd.DataFrame()
    raw = pd.concat([pd.read_csv(f, dtype=str) for f in files], ignore_index=True)
    raw["DATE"] = pd.to_datetime(raw["DATE"])
    raw = raw.drop_duplicates(subset=["DATE", "REPORT_TYPE"]).reset_index(drop=True)
    rep = raw["REPORT_TYPE"].fillna("").str.strip()
    raw["rank"] = rep.map(REPORT_RANK).fillna(3)
    raw["H"] = raw["DATE"].dt.round("h")
    raw["dt_min"] = (raw["DATE"] - raw["H"]).abs() / pd.Timedelta(minutes=1)

    t = _num_qc(raw["TMP"], 10.0, 9999) if "TMP" in raw else np.nan
    td = _num_qc(raw["DEW"], 10.0, 9999) if "DEW" in raw else np.nan
    raw["temperature_2m"] = t
    raw["relative_humidity_2m"] = rh_from_t_td(t, td).clip(upper=100.0)
    w = raw["WND"].fillna("").astype(str).str.split(",", expand=True).reindex(
        columns=range(5))
    spd = pd.to_numeric(w[3], errors="coerce")
    wok = w[4].isin(list(ISD_OK_QC)) & (spd != 9999)
    raw["wind_speed_10m"] = (spd / 10.0).where(wok)
    raw.loc[(w[2] == "C") & w[4].isin(list(ISD_OK_QC)), "wind_speed_10m"] = 0.0   # calm
    v = raw["VIS"].fillna("").astype(str).str.split(",", expand=True).reindex(columns=range(4))
    vis = pd.to_numeric(v[0], errors="coerce")
    raw["visibility"] = (vis / 1000.0).where(v[1].isin(list(ISD_OK_QC)) & (vis != 999999))
    # Reporting cap per (year, report type): the maximum value if >= 5 % of the
    # reports sit there (METAR "9999" = 10 km or more, US ASOS 10 SM = 16.093 km,
    # Korean SYNOP 20 km). Values at the cap are censored ("at least").
    raw["vis_cap"] = np.nan
    yr = raw["DATE"].dt.year
    caps = {}
    for (y, rt), g in raw["visibility"].groupby([yr, rep]):
        g = g.dropna()
        if len(g) < 100:
            continue
        mx = g.max()
        if (g >= mx - 1e-6).mean() >= 0.05:
            raw.loc[(yr == y) & (rep == rt), "vis_cap"] = mx
            caps[f"{y}/{rt}"] = float(mx)
    DIAG["isd_visibility_caps_km"][city] = caps

    # --- precipitation
    pg = _precip_groups(raw)
    diag_p = pg.groupby("period").size().to_dict() if len(pg) else {}
    DIAG["isd_precip_groups_by_period_h"][city] = {str(k): int(v) for k, v in diag_p.items()}
    raw["precip_1h"] = np.nan
    p1 = pg[pg.period == 1].drop_duplicates("row")
    raw.loc[p1["row"].values, "precip_1h"] = p1["mm"].values
    # hourly amounts only from routine reports (SPECI totals cover part-hours)
    raw.loc[~rep.isin(["FM-15", "FM-12"]), "precip_1h"] = np.nan
    # A station that reports hourly amounts omits the group in dry hours
    # (METAR "P" group). Zero-fill only routine reports of station-years with
    # >= 200 hourly groups and no precipitation in the present-weather groups.
    raw["year"] = raw["DATE"].dt.year
    metar = rep == "FM-15"
    pw = _present_weather_precip(raw)
    n_p1_year = raw.loc[raw.precip_1h.notna() & metar].groupby("year").size()
    n_metar_year = raw.loc[metar].groupby("year").size()
    share = (n_p1_year / n_metar_year).fillna(0)
    # station-year reports hourly amounts: >= 50 groups and >= 2 % of its METARs
    hourly_years = set(share[(share >= 0.02) & (n_p1_year.reindex(share.index).fillna(0) >= 50)]
                       .index)
    # US-style METAR (ASOS): the hourly "P" group is only sent when precipitation
    # fell, so a routine METAR without it and without precipitation in the
    # present-weather groups is a dry hour.
    fill = metar & raw["year"].isin(hourly_years) & raw["precip_1h"].isna() & ~pw
    raw.loc[fill, "precip_1h"] = 0.0
    # Hourly SYNOP stations (SYNOP_HOURLY_PRECIP, e.g. Seoul from 2024): same
    # rule for full-hour FM-12 reports of station-years with hourly groups.
    if city in SYNOP_HOURLY_PRECIP:
        synop = (rep == "FM-12") & (raw["DATE"].dt.minute == 0)
        n_s = raw.loc[raw.precip_1h.notna() & synop].groupby("year").size()
        syn_years = set(n_s[n_s >= 200].index)
        fill_s = synop & raw["year"].isin(syn_years) & raw["precip_1h"].isna() & ~pw
        raw.loc[fill_s, "precip_1h"] = 0.0
        DIAG["isd_precip_synop_zero_fill"][city] = dict(
            years=sorted(int(y) for y in syn_years), n_filled=int(fill_s.sum()),
            n_unknown_present_weather=int((synop & raw["year"].isin(syn_years)
                                           & raw["precip_1h"].isna() & pw).sum()))
    # An hourly series that contains (almost) no zeros only reports wet hours
    # and cannot give wet/dry statistics -> not used.
    usable = {}
    for y, d in raw[raw.precip_1h.notna()].groupby("year"):
        share0 = float((d.precip_1h == 0).mean())
        usable[int(y)] = dict(n=int(len(d)), share_zero=round(share0, 3), used=share0 >= 0.5)
        if share0 < 0.5:
            raw.loc[raw.year == y, "precip_1h"] = np.nan
    DIAG["isd_precip_hourly"][city] = usable

    hourly = {}
    for var in INSTANT + ["precip_1h"]:
        extra = ["vis_cap"] if var == "visibility" else []
        s = raw.loc[raw[var].notna() & (raw.dt_min <= 30),
                    ["H", "rank", "dt_min", var] + extra]
        s = s.sort_values(["H", "dt_min", "rank"]).drop_duplicates("H")
        hourly[var] = s.set_index("H")[var]
        if extra:
            hourly["vis_cap"] = s.set_index("H")["vis_cap"]
    hourly = pd.DataFrame(hourly).sort_index()
    hourly = hourly.rename(columns={"precip_1h": "precipitation"})

    # multi-hour precipitation windows (3 h and 6 h) as reported
    win = pg[pg.period.isin([3, 6])].copy()
    if len(win):
        win["end"] = raw.loc[win["row"].values, "H"].values
        win = (win.sort_values(["end", "period"]).drop_duplicates(["end", "period"])
               [["end", "period", "mm"]])
        # keep (period, year) series that also report dry windows
        win["year"] = pd.DatetimeIndex(win["end"]).year
        keep, info = [], {}
        for (per, y), g in win.groupby(["period", "year"]):
            share0 = float((g.mm == 0).mean())
            ok = share0 >= 0.5 and len(g) >= 100
            info[f"{per}h/{y}"] = dict(n=int(len(g)), share_zero=round(share0, 3), used=ok)
            if ok:
                keep.append(g)
        DIAG["isd_precip_windows"][city] = info
        win = (pd.concat(keep)[["end", "period", "mm"]] if keep
               else pd.DataFrame(columns=["end", "period", "mm"]))
    DIAG["isd_coverage"][city] = {v: int(hourly[v].notna().sum()) for v in hourly.columns}
    return hourly, win


# --------------------------------------------------------------- GEFS -----
def load_gefs(root):
    files = sorted(glob.glob(os.path.join(root, "gefs_points", "[0-9]*.csv")))
    if not files:
        return pd.DataFrame(), pd.DataFrame()
    g = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    g["init"] = pd.to_datetime(g["init_utc"])
    DIAG["gefs"] = dict(n_runs=int(g["init"].nunique()),
                        runs_per_city=g.groupby("city")["init"].nunique().to_dict(),
                        grid=g.groupby("city")[["grid_lat", "grid_lon"]].first()
                        .to_dict(orient="index"))
    inst = g[g.kind == "instant"].pivot_table(index=["city", "init", "step_end_h"],
                                              columns="file_var", values="value")
    inst = inst.reset_index().rename(columns={"step_end_h": "lead"})
    inst["temperature_2m"] = inst["tmp_2m"] - 273.15
    inst["relative_humidity_2m"] = rh_from_q(inst["temperature_2m"], inst["spfh_2m"],
                                             inst["pres_sfc"]).clip(0, 100)
    inst["wind_speed_10m"] = np.hypot(inst["ugrd_hgt"], inst["vgrd_hgt"])
    inst["valid"] = inst["init"] + pd.to_timedelta(inst["lead"], unit="h")

    # windows: APCP acc and DSWRF ave over (s0, s1] -> 3-hourly pieces
    rows = []
    for (city, init, var), d in g[g.kind.isin(["acc", "ave"])].groupby(
            ["city", "init", "file_var"]):
        val = {(int(a), int(b)): float(v) for a, b, v in
               zip(d.step_start_h, d.step_end_h, d.value)}
        for (s0, s1), v in val.items():
            if s1 - s0 == 3:
                x = v
            elif s1 - s0 == 6 and (s0, s0 + 3) in val:
                first = val[(s0, s0 + 3)]
                # acc: total(6h) - total(first 3h); ave: 2*mean(6h) - mean(first 3h)
                x = v - first if var == "apcp_sfc" else 2 * v - first
                s0 = s0 + 3
            else:
                continue
            rows.append((city, init, var, s1, x))
    w = pd.DataFrame(rows, columns=["city", "init", "var", "lead", "value"])
    w = w.pivot_table(index=["city", "init", "lead"], columns="var", values="value").reset_index()
    w["precipitation"] = w["apcp_sfc"].clip(lower=0)            # mm per 3 h
    w["shortwave_radiation"] = w["dswrf_sfc"].clip(lower=0)     # W/m2 mean over 3 h
    w["valid"] = w["init"] + pd.to_timedelta(w["lead"], unit="h")
    return inst, w


# ===========================================================================
# Statistics
# ===========================================================================
class Collector:
    def __init__(self):
        self.rows = []

    def add(self, city, var, lead, stat, value, n, fc, ob, years):
        if value is None or not np.isfinite(value):
            return
        self.rows.append(dict(city=city, variable=var, lead_time_h=lead, statistic=stat,
                              value=float(value), n_samples=int(n), forecast_source=fc,
                              observation_source=ob, years=years))


def per_lead_continuous(m, var, col, city, fc, ob, years, C, leads=None):
    """bias & SD of error for temperature/humidity/wind."""
    for lead, d in m.groupby("lead"):
        if leads is not None and lead not in leads:
            continue
        e = (d["fc"] - d["ob"]).dropna()
        if len(e) < 10:
            continue
        C.add(city, var, lead, "bias", e.mean(), len(e), fc, ob, years)
        C.add(city, var, lead, "sd_error", e.std(ddof=1), len(e), fc, ob, years)


def per_lead_solar(m, city, fc, ob, years, C, leads=None, suffix=""):
    for lead, d in m.groupby("lead"):
        if leads is not None and lead not in leads:
            continue
        d = d[d["day"]].dropna(subset=["fc", "ob"])
        if len(d) < 10 or d["ob"].mean() <= 0:
            continue
        mae = (d["fc"] - d["ob"]).abs().mean()
        C.add(city, "shortwave_radiation", lead, "rel_mae" + suffix,
              mae / d["ob"].mean(), len(d), fc, ob, years)
        C.add(city, "shortwave_radiation", lead, "mae_wm2" + suffix, mae, len(d), fc, ob, years)
        C.add(city, "shortwave_radiation", lead, "mean_obs_wm2" + suffix, d["ob"].mean(),
              len(d), fc, ob, years)


def vis_variants(d):
    """Station visibility is censored at a reporting cap (column 'cap', NaN = none).
    all_hours_capped : forecast and observation both cut at the report's cap
    obs_below_cap    : only hours whose observation is below its cap"""
    cap = d["cap"] if "cap" in d else pd.Series(np.nan, index=d.index)
    if cap.notna().any():
        c = cap.fillna(np.inf)
        return {"all_hours_capped": d.assign(fc=np.minimum(d.fc, c), ob=np.minimum(d.ob, c)),
                "obs_below_cap": d[~(d.ob >= c - 1e-6)]}
    return {"all_hours": d}


def per_lead_visibility(m, city, fc, ob, years, C, leads=None):
    for lead, d in m.groupby("lead"):
        if leads is not None and lead not in leads:
            continue
        d = d.dropna(subset=["fc", "ob"])
        d = d[(d.fc > 0) & (d.ob > 0)]
        variants = vis_variants(d)
        for name, x in variants.items():
            if len(x) < 10:
                continue
            lr = np.log(x.fc / x.ob)
            sd = lr.std(ddof=1)
            C.add(city, "visibility", lead, f"sd_log_ratio_{name}", sd, len(x), fc, ob, years)
            C.add(city, "visibility", lead, f"cv_{name}", cv_from_log_sd(sd), len(x), fc, ob, years)
            C.add(city, "visibility", lead, f"mean_log_ratio_{name}", lr.mean(), len(x), fc, ob,
                  years)


def per_lead_precip(m, city, fc, ob, years, C, leads=None, suffix=""):
    for lead, d in m.groupby("lead"):
        if leads is not None and lead not in leads:
            continue
        d = d.dropna(subset=["fc", "ob"])
        n = len(d)
        if n < 10:
            continue
        for tname, thr in WET_THRESHOLDS.items():
            ow, fw = d.ob > thr, d.fc > thr
            hits = int((ow & fw).sum())
            miss = int((ow & ~fw).sum())
            fa = int((~ow & fw).sum())
            dry = int((~ow).sum())
            s = f"_{tname}{suffix}"
            C.add(city, "precipitation", lead, "p_wet_obs" + s, ow.mean(), n, fc, ob, years)
            C.add(city, "precipitation", lead, "p_wet_fc" + s, fw.mean(), n, fc, ob, years)
            if hits + miss > 0:
                C.add(city, "precipitation", lead, "pod" + s, hits / (hits + miss), hits + miss,
                      fc, ob, years)
                C.add(city, "precipitation", lead, "miss_rate" + s, miss / (hits + miss),
                      hits + miss, fc, ob, years)
            if hits + fa > 0:
                C.add(city, "precipitation", lead, "far" + s, fa / (hits + fa), hits + fa, fc, ob,
                      years)
            if dry > 0:
                C.add(city, "precipitation", lead, "false_alarm_prob_per_dry" + s, fa / dry, dry,
                      fc, ob, years)
            h = d[ow & fw]
            if len(h) >= 5:
                lr = np.log(h.fc / h.ob)
                sd = lr.std(ddof=1)
                C.add(city, "precipitation", lead, "hits_sd_log_ratio" + s, sd, len(h), fc, ob,
                      years)
                C.add(city, "precipitation", lead, "hits_cv" + s, cv_from_log_sd(sd), len(h), fc,
                      ob, years)


def autocorr(m, var, city, fc, ob, years, C, lags, err_col="err", max_lead=MAX_LEAD):
    """Correlation of the error at lead L with the error at lead L+k of the same run."""
    wide = m.pivot_table(index="init", columns="lead", values=err_col)
    for k in lags:
        for L in wide.columns:
            if L > max_lead or (L + k) not in wide.columns:
                continue
            x = wide[[L, L + k]].dropna()
            if len(x) < 20 or x[L].std() == 0 or x[L + k].std() == 0:
                continue
            r = np.corrcoef(x[L], x[L + k])[0, 1]
            C.add(city, var, int(L), f"acf_lag{k}h", r, len(x), fc, ob, years)


def add_fits(C):
    """Straight line value = a + b*lead for every (city, var, stat, fc, ob) group."""
    df = pd.DataFrame(C.rows)
    if df.empty:
        return
    df = df[df.lead_time_h.apply(lambda x: isinstance(x, (int, np.integer)))]
    keys = ["city", "variable", "statistic", "forecast_source", "observation_source", "years"]
    for k, d in df.groupby(keys):
        d = d[(d.lead_time_h >= 1) & (d.lead_time_h <= MAX_LEAD)]
        f = linfit(d.lead_time_h, d.value)
        if f is None:
            continue
        lab = f"fit_{int(f['lead_min'])}-{int(f['lead_max'])}"
        base = dict(zip(keys, k))
        for s, v in [("a", f["a"]), ("b_per_h", f["b"]), ("r2", f["r2"]), ("rmse", f["rmse"]),
                     ("n_leads", f["n_leads"])]:
            C.rows.append(dict(base, lead_time_h=lab, statistic=f"{base['statistic']}__fit_{s}",
                               value=float(v), n_samples=int(d.n_samples.sum())))


# ===========================================================================
# Pairing forecasts with references
# ===========================================================================
def years_label(inits):
    return f"{inits.min():%Y-%m-%d}..{inits.max():%Y-%m-%d} (run dates)"


def visibility_cap(ob):
    """Reporting cap of a station = its maximum value if >= 5 % of reports sit there."""
    ob = ob.dropna()
    if ob.empty:
        return None
    mx = ob.max()
    share = float((ob >= mx - 1e-6).mean())
    return float(mx) if share >= 0.05 else None


def native_windows(fc_city, var, how):
    """ECMWF native windows beyond 90 h from hourly values: (L-3, L] to 144 h,
    (L-6, L] beyond. Returns DataFrame init, lead, valid, fc (window value)."""
    out = []
    f = fc_city.set_index(["init", "lead"])[var]
    for L in list(range(93, 145, 3)) + list(range(150, MAX_LEAD + 1, 6)):
        w = 3 if L <= 144 else 6
        parts = [f.xs(l, level="lead") for l in range(L - w + 1, L + 1)]
        mat = pd.concat(parts, axis=1)
        val = mat.sum(axis=1, min_count=w) if how == "sum" else mat.mean(axis=1)
        val = val.where(mat.notna().all(axis=1))
        out.append(pd.DataFrame({"init": val.index, "lead": L, "fc": val.values,
                                 "window_h": w}))
    d = pd.concat(out, ignore_index=True)
    d["valid"] = d["init"] + pd.to_timedelta(d["lead"], unit="h")
    return d


def ref_window(ref_hourly, valid, w, how):
    """Aggregate an hourly reference over (valid-w, valid]."""
    s = ref_hourly
    parts = [s.reindex(valid - pd.Timedelta(hours=i)).values for i in range(w)]
    mat = np.vstack(parts).T
    ok = np.isfinite(mat).all(axis=1)
    val = mat.sum(axis=1) if how == "sum" else mat.mean(axis=1)
    return np.where(ok, val, np.nan)


def analyse_ecmwf(fc, refs, C):
    for city in CITIES:
        f = fc[(fc.city == city) & (fc.lead >= 1)]
        if f.empty:
            continue
        yl = years_label(f.init)
        lat, lon = CITIES[city]["lat"], CITIES[city]["lon"]
        for rname, ref in refs[city].items():
            if ref is None or ref.empty:
                continue
            obname = OBS_NAMES[rname].format(station=CITIES[city]["isd"])
            for var in VARS:
                if var not in ref.columns or ref[var].notna().sum() == 0:
                    continue
                m = f[["init", "lead", "valid", var]].rename(columns={var: "fc"})
                m["ob"] = ref[var].reindex(m["valid"]).values
                m = m[m.lead <= MAX_LEAD + 24]            # +24 h for the lag-24 ACF
                m_main = m[m.lead <= MAX_LEAD]
                if m_main[["fc", "ob"]].dropna().empty:
                    continue
                native_inst = [L for L in range(1, MAX_LEAD + 1)
                               if L <= 90 or (L <= 144 and L % 3 == 0) or L % 6 == 0]
                if var in ("temperature_2m", "relative_humidity_2m", "wind_speed_10m"):
                    per_lead_continuous(m_main, var, "fc", city, FC_ECMWF_SERVED, obname, yl, C)
                    per_lead_continuous(m_main, var, "fc", city, FC_ECMWF_NATIVE, obname, yl, C,
                                        leads=set(native_inst))
                    m["err"] = m.fc - m.ob
                    autocorr(m, var, city, FC_ECMWF_SERVED, obname, yl, C, LAGS)
                elif var == "visibility":
                    capcol = ref["vis_cap"] if "vis_cap" in ref else None
                    m["cap"] = capcol.reindex(m["valid"]).values if capcol is not None else np.nan
                    m_main = m[m.lead <= MAX_LEAD]
                    per_lead_visibility(m_main, city, FC_ECMWF_SERVED, obname, yl, C)
                    per_lead_visibility(m_main, city, FC_ECMWF_NATIVE, obname, yl, C,
                                        leads=set(native_inst))
                    x = m.dropna(subset=["fc", "ob"])
                    x = x[(x.fc > 0) & (x.ob > 0)]
                    x = list(vis_variants(x).values())[0].copy()   # capped (or all) hours
                    x["err"] = np.log(x.fc / x.ob)
                    autocorr(x, var, city, FC_ECMWF_SERVED, obname, yl, C, LAGS)
                elif var == "shortwave_radiation":
                    m["day"] = solar_elevation(m["valid"] - pd.Timedelta(minutes=30),
                                               lat, lon) > 0
                    m_main = m[m.lead <= MAX_LEAD]
                    per_lead_solar(m_main, city, FC_ECMWF_SERVED, obname, yl, C)
                    per_lead_solar(m_main, city, FC_ECMWF_NATIVE, obname, yl, C,
                                   leads=set(range(1, 91)))
                    nw = native_windows(f[f.lead <= MAX_LEAD], var, "mean")
                    nw["ob"] = np.nan
                    nw["day"] = False
                    for w in (3, 6):
                        sel = nw.window_h == w
                        nw.loc[sel, "ob"] = ref_window(ref[var], pd.DatetimeIndex(
                            nw.loc[sel, "valid"]), w, "mean")
                        mid = nw.loc[sel, "valid"] - pd.Timedelta(minutes=30 * w)
                        nw.loc[sel, "day"] = solar_elevation(mid, lat, lon) > 0
                    nw["day"] = nw["day"].astype(bool)
                    per_lead_solar(nw, city, FC_ECMWF_NATIVE, obname, yl, C, suffix="_window")
                    x = m[m.day].copy()
                    x["err"] = x.fc - x.ob
                    autocorr(x, var, city, FC_ECMWF_SERVED, obname, yl, C, LAGS)
                elif var == "precipitation":
                    per_lead_precip(m_main, city, FC_ECMWF_SERVED, obname, yl, C)
                    per_lead_precip(m_main, city, FC_ECMWF_NATIVE, obname, yl, C,
                                    leads=set(range(1, 91)))
                    nw = native_windows(f[f.lead <= MAX_LEAD], var, "sum")
                    nw["ob"] = np.nan
                    for w in (3, 6):
                        sel = nw.window_h == w
                        nw.loc[sel, "ob"] = ref_window(ref[var], pd.DatetimeIndex(
                            nw.loc[sel, "valid"]), w, "sum")
                    per_lead_precip(nw, city, FC_ECMWF_NATIVE, obname, yl, C, suffix="_window")
                    m["err"] = m.fc - m.ob
                    autocorr(m, var, city, FC_ECMWF_SERVED, obname, yl, C, LAGS)


def analyse_ecmwf_isd_windows(fc, isd_windows, C):
    """ECMWF vs station multi-hour precipitation totals (where no hourly amounts exist)."""
    for city in CITIES:
        win = isd_windows.get(city)
        f = fc[(fc.city == city) & (fc.lead >= 1)]
        if win is None or win.empty or f.empty:
            continue
        obname = OBS_NAMES["isd"].format(station=CITIES[city]["isd"])
        yl = years_label(f.init)
        piv = f.set_index(["init", "lead"])["precipitation"]
        for P in sorted(win.period.unique()):
            wp = win[win.period == P].set_index("end")["mm"]
            rows = []
            for L in range(P, MAX_LEAD + 1):
                parts = [piv.xs(l, level="lead") for l in range(L - P + 1, L + 1)]
                mat = pd.concat(parts, axis=1)
                val = mat.sum(axis=1).where(mat.notna().all(axis=1))
                valid = val.index + pd.Timedelta(hours=L)
                ob = wp.reindex(valid).values
                rows.append(pd.DataFrame({"init": val.index, "lead": L, "fc": val.values,
                                          "ob": ob}))
            m = pd.concat(rows, ignore_index=True).dropna()
            per_lead_precip(m, city, FC_ECMWF_SERVED, obname, yl, C, suffix=f"_{P}h_total")


def analyse_gefs(inst, win, refs_datayears, isd_windows, C):
    for city in CITIES:
        fi = inst[inst.city == city] if len(inst) else inst
        fw = win[win.city == city] if len(win) else win
        if fi is None or len(fi) == 0:
            continue
        yl = f"{fi.init.min():%Y}-{fi.init.max():%Y} (run dates {fi.init.min():%Y-%m-%d}.." \
             f"{fi.init.max():%Y-%m-%d}, same years as the bike data)"
        lat, lon = CITIES[city]["lat"], CITIES[city]["lon"]
        for rname, ref in refs_datayears[city].items():
            if ref is None or ref.empty:
                continue
            obname = OBS_NAMES[rname].format(station=CITIES[city]["isd"])
            for var in ("temperature_2m", "relative_humidity_2m", "wind_speed_10m"):
                if var not in ref.columns:
                    continue
                m = fi[["init", "lead", "valid", var]].rename(columns={var: "fc"})
                m["ob"] = ref[var].reindex(m["valid"]).values
                per_lead_continuous(m[m.lead <= MAX_LEAD], var, "fc", city, FC_GEFS, obname, yl, C)
                m["err"] = m.fc - m.ob
                autocorr(m, var, city, FC_GEFS, obname, yl, C, [3, 6, 12, 24],
                         max_lead=MAX_LEAD)
            if len(fw) == 0:
                continue
            for var, how in (("shortwave_radiation", "mean"), ("precipitation", "sum")):
                if var not in ref.columns or ref[var].notna().sum() == 0:
                    continue
                m = fw[["init", "lead", "valid", var]].rename(columns={var: "fc"})
                m["ob"] = ref_window(ref[var], pd.DatetimeIndex(m["valid"]), 3, how)
                if var == "shortwave_radiation":
                    m["day"] = solar_elevation(m["valid"] - pd.Timedelta(minutes=90), lat, lon) > 0
                    per_lead_solar(m[m.lead <= MAX_LEAD], city, FC_GEFS, obname, yl, C,
                                   suffix="_3h_mean")
                    x = m[m.day].copy()
                else:
                    per_lead_precip(m[m.lead <= MAX_LEAD], city, FC_GEFS, obname, yl, C,
                                    suffix="_3h_total")
                    x = m.copy()
                x["err"] = x.fc - x.ob
                autocorr(x, var, city, FC_GEFS, obname, yl, C, [3, 6, 12, 24])
        # station multi-hour precipitation totals
        wi = isd_windows.get(city)
        if wi is not None and len(wi) and len(fw):
            obname = OBS_NAMES["isd"].format(station=CITIES[city]["isd"])
            piv = fw.set_index(["init", "lead"])["precipitation"]
            for P in sorted(wi.period.unique()):
                wp = wi[wi.period == P].set_index("end")["mm"]
                rows = []
                for L in range(P, MAX_LEAD + 1, 3):
                    parts = [piv.xs(l, level="lead") for l in range(L - P + 3, L + 1, 3)
                             if l in piv.index.get_level_values("lead")]
                    if len(parts) != P // 3:
                        continue
                    mat = pd.concat(parts, axis=1)
                    val = mat.sum(axis=1).where(mat.notna().all(axis=1))
                    rows.append(pd.DataFrame({"init": val.index, "lead": L, "fc": val.values,
                                              "ob": wp.reindex(val.index + pd.Timedelta(
                                                  hours=L)).values}))
                if rows:
                    m = pd.concat(rows, ignore_index=True).dropna()
                    per_lead_precip(m, city, FC_GEFS, obname, yl, C, suffix=f"_{P}h_total")


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="unzipped nwp_extracts folder")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    C = Collector()

    fc = load_single_runs(a.data)
    refs_recent, refs_datayears, isd_windows = {}, {}, {}
    for city in CITIES:
        isd_h, isd_w = load_isd(a.data, city)
        isd_windows[city] = isd_w
        refs_recent[city] = {
            "isd": isd_h,
            "era5s": load_archive(a.data, city, "recent_era5seamless"),
            "ifsbm": load_archive(a.data, city, "recent_bestmatch"),
        }
        refs_datayears[city] = {
            "isd": isd_h,
            "era5s": load_archive(a.data, city, "datayears_era5seamless"),
        }
    if len(fc):
        analyse_ecmwf(fc, refs_recent, C)
        analyse_ecmwf_isd_windows(fc, isd_windows, C)
    inst, win = load_gefs(a.data)
    if len(inst):
        analyse_gefs(inst, win, refs_datayears, isd_windows, C)
    add_fits(C)

    out = pd.DataFrame(C.rows, columns=["city", "variable", "lead_time_h", "statistic", "value",
                                        "n_samples", "forecast_source", "observation_source",
                                        "years"])
    out.to_csv(os.path.join(a.out, "nwp_error_statistics.csv"), index=False)
    with open(os.path.join(a.out, "diagnostics.json"), "w") as f:
        json.dump(DIAG, f, indent=1, default=str)
    print(f"{len(out)} rows -> {os.path.join(a.out, 'nwp_error_statistics.csv')}")


if __name__ == "__main__":
    main()
