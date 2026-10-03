#!/usr/bin/env python3
"""
check_seoul_winter_split.py - evidence for the Seoul winter precipitation
treatment in build_nwp_calibration.py (3 Oct 2026).

1. Format of the station's winter amounts (NOAA ISD 47108, routine full-hour
   SYNOPs FM-12, 2024+): are the November-March amounts 3-hour totals at 00,
   03, ..., 21 KST (the KMA portal format of the bike data)? Counts positive
   amounts at those hours and at the other hours, reports with precipitation
   in the present-weather group but no amount, and compares the amounts with
   the station's running 24-hour totals.
2. Comparison method, tested on April-October, where the station is truly
   hourly: the hourly station data are turned into 3-hour totals and split
   like the winter data. Miss rates of the ECMWF forecast (24 h, pooled over
   +/-12 lead hours, as in the calibration):
     A  true hourly station  vs hourly forecast   (what hourly data would give)
     B  split station        vs hourly forecast
     C  split station        vs split forecast    (the calibration's method)

Usage (from forecasting/):
    python weather/nwp/check_seoul_winter_split.py --data <nwp_extracts>
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_nwp_errors as A  # noqa: E402
import build_nwp_calibration as B  # noqa: E402


def winter_format(data):
    st = A.CITIES["seoul"]["isd"]
    files = sorted(glob.glob(os.path.join(data, "isd_trimmed", f"{st}_*.csv")))
    raw = pd.concat([pd.read_csv(f, dtype=str) for f in files], ignore_index=True)
    raw["DATE"] = pd.to_datetime(raw["DATE"])
    raw = raw.drop_duplicates(subset=["DATE", "REPORT_TYPE"]).reset_index(drop=True)
    rep = raw["REPORT_TYPE"].fillna("").str.strip()
    keep = (rep == "FM-12") & (raw["DATE"].dt.minute == 0) & (raw["DATE"].dt.year >= 2024)
    raw["season"] = np.where((raw["DATE"] + A.KST).dt.month.isin(A.SEOUL_WINTER_MONTHS),
                             "Nov-Mar", "Apr-Oct")
    raw["hours"] = np.where(raw["DATE"].dt.hour % 3 == 0, "00,03,..,21 KST", "other")
    pg = A._precip_groups(raw)
    pg = pg[pg["row"].isin(raw.index[keep])]
    p1 = pg[pg.period == 1].drop_duplicates("row").set_index("row")["mm"]
    sub = raw[keep].copy()
    sub["amount"] = p1.reindex(sub.index)
    sub["pw"] = A._present_weather_precip(raw)[keep]
    t = sub.groupby(["season", "hours"]).apply(lambda d: pd.Series(dict(
        reports=len(d), amount_positive=int((d.amount > 0).sum()),
        no_amount_but_precipitation=int((d.amount.isna() & d.pw).sum()))))
    print("1. Station reports (FM-12, full hour, 2024+), season in Korean time\n")
    print(t.to_string())
    # amounts vs the station's running 24-hour totals (winter, totals > 0)
    pg = pg.assign(H=raw.loc[pg["row"].values, "DATE"].values)
    a1 = pg[pg.period == 1].drop_duplicates("row").set_index("H")["mm"].sort_index()
    a24 = pg[pg.period == 24].drop_duplicates("row").set_index("H")["mm"].sort_index()
    ratios = []
    for t_end, v in a24.items():
        if v > 0 and (t_end + A.KST).month in A.SEOUL_WINTER_MONTHS:
            w = a1[(a1.index > t_end - pd.Timedelta(hours=24)) & (a1.index <= t_end)]
            ratios.append(w.sum() / v)
    print(f"\nNov-Mar 24-hour totals > 0: {len(ratios)}; median (sum of amounts / 24-h total): "
          f"{np.median(ratios):.2f}")


def method_test(data):
    fc = A.load_single_runs(data)
    isd, _ = A.load_isd(data, "seoul")
    ob = isd["precipitation"]
    f = fc[(fc.city == "seoul") & fc.lead.between(1, B.MAX_LEAD)]
    p = f[["init", "lead", "valid", "precipitation"]].rename(columns={"precipitation": "fc"})
    p["ob"] = ob.reindex(p["valid"]).values
    end = A.seoul_window_end(p["valid"])
    hourly = ~np.isin((end + A.KST).month, list(A.SEOUL_WINTER_MONTHS))   # Apr-Oct windows
    p, end = p[hourly].copy(), end[hourly]
    w = ob.groupby(A.seoul_window_end(ob.index)).agg(["sum", "count"])
    p["ob_split"] = (w["sum"] / 3).where(w["count"] == 3).reindex(end).values
    g = p.groupby([p["init"].to_numpy(), end])["fc"]
    p["fc_split"] = (g.transform("sum") / 3).where(g.transform("count") == 3)
    p = p.dropna(subset=["fc", "ob", "ob_split", "fc_split"])
    print("\n2. April-October, station turned into split 3-hour totals; miss rate at 24 h")
    print("   (pooled +/-12 lead hours), light < 1 mm/h / stronger >= 1 mm/h\n")
    for name, o, fcv, to, tf in [
            ("A true hourly station vs hourly forecast", p.ob, p.fc, B.WET_MM, B.WET_MM),
            ("B split station       vs hourly forecast", p.ob_split, p.fc, B.WET_MM_SPLIT, B.WET_MM),
            ("C split station       vs split forecast ", p.ob_split, p.fc_split,
             B.WET_MM_SPLIT, B.WET_MM_SPLIT)]:
        q = pd.DataFrame({"init": p["init"].to_numpy(), "lead": p.lead.to_numpy(),
                          "ob": o.to_numpy(), "fc": fcv.to_numpy()})
        q["ow"], q["fw"] = q.ob > to, q.fc > tf
        miss = B.precip_rates(q)[0]
        im = B.intensity_miss_rates(q)[0]
        print(f"   {name}: all {miss[24]:.3f}   light {im[0, 24]:.3f}   stronger {im[1, 24]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="unzipped nwp_extracts folder")
    a = ap.parse_args()
    winter_format(a.data)
    method_test(a.data)


if __name__ == "__main__":
    main()
