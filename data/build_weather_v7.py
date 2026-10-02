"""Provenance record (2 Oct 2026): how the v7 weather columns were built.
Not part of the pipeline.

London and Washington (LondonBikeData.csv, WashingtonBikeData.csv):

1. Daylight saving: the Open-Meteo weather columns were stored on standard
   time all year (London UTC+0, Washington UTC-5), while the bike counts
   (and visibility, from Visual Crossing) follow the local clock with
   daylight saving. Each Open-Meteo column is re-aligned to the clock: the
   row with local time T gets the value stored for its true UTC hour.
   Outside daylight saving nothing changes; during daylight saving each
   value moves down by one row. Values are copied as stored (no rounding).
2. precipitation_mm = rainfall_mm + snowfall_cm / 0.7 (Open-Meteo's own
   definition of total precipitation; 0.7 cm of snow per mm of water).
3. snow_depth_cm = Open-Meteo archive snow_depth (best_match = ERA5-Land,
   metres) x 100 at the row's UTC hour, grid cell of the airport station
   (London Heathrow 51.479, -0.449; Washington Reagan National 38.8483,
   -77.0342; files open-meteo-51.49N0.49W24m.csv / open-meteo-38.84N77.09W3m.csv).
Old columns are kept (rainfall_mm and snowfall_cm re-aligned too).
Not re-aligned: visibility_km (Visual Crossing, already clock time), and
London t2 / weather_code (original Kaggle columns, not used).

Seoul (SeoulBikeData.csv; original UCI columns unchanged):
4. precipitation_mm = Rainfall (KMA ASOS 강수량, total precipitation incl.
   melted snow), with the November-March values spread evenly over their
   three hours: KMA reports precipitation as 3-hour totals in Nov-Mar, at
   00, 03, ..., 21 h; the value at hour t covers the hours labelled t-2,
   t-1 and t (labels = end of the hour), so each of them gets value / 3.
   April-October (hourly) unchanged. Totals are kept.
5. snow_depth_cm = Snowfall (KMA ASOS 적설 = snow depth on the ground, cm;
   copied under the name used in all cities).
"""
import sys
import numpy as np
import pandas as pd

CITIES = {
    "london": dict(file="LondonBikeData.csv", om="open-meteo-51.49N0.49W24m.csv", tz="Europe/London", std_offset_h=0,
                   cols=["t1", "hum", "wind_speed", "temperature_c", "humidity_percent", "dew_point_c",
                         "rainfall_mm", "snowfall_cm", "wind_speed_ms", "solar_radiation_wm2",
                         "solar_radiation_mjm2"]),
    "washington": dict(file="WashingtonBikeData.csv", om="open-meteo-38.84N77.09W3m.csv", tz="America/New_York",
                       std_offset_h=-5,
                       cols=["temperature_c", "humidity_percent", "dew_point_c", "rainfall_mm",
                             "snowfall_cm", "wind_speed_ms", "solar_radiation_wm2",
                             "solar_radiation_mjm2"]),
}


def build(city, data_dir, om_dir, out_dir):
    c = CITIES[city]
    raw = pd.read_csv(f"{data_dir}/{c['file']}", dtype=str, keep_default_na=False)
    if "precipitation_mm" in raw.columns:
        raise RuntimeError(f"{c['file']} is already a v7 file: use the original data as input")
    ts = pd.DatetimeIndex(pd.to_datetime(raw["timestamp"]))
    # true UTC hour of each row (autumn duplicate: first = summer time; spring
    # gap hour, which does not exist on the clock: the next real hour)
    utc = ts.tz_localize(c["tz"], ambiguous="infer", nonexistent="shift_forward") \
            .tz_convert("UTC").tz_localize(None)
    # label under which the old (standard-time) data stored that UTC hour
    label = utc + pd.Timedelta(hours=c["std_offset_h"])
    first = ~ts.duplicated()
    out = raw.copy()
    for col in c["cols"]:
        old = pd.Series(raw[col].to_numpy()[first], index=ts[first])
        new = old.reindex(label)
        if new.isna().any():
            raise RuntimeError(f"{city} {col}: {new.isna().sum()} rows without a value")
        out[col] = new.to_numpy()

    rain = out["rainfall_mm"].astype(float)
    snow = out["snowfall_cm"].astype(float)
    precip = np.round(rain + snow / 0.7, 1)

    om = pd.read_csv(f"{om_dir}/{c['om']}", skiprows=3)
    om.columns = [x.split(" (")[0].replace("_archive", "") for x in om.columns]
    om.index = pd.to_datetime(om["time"])
    sd = om["snow_depth_best_match"].reindex(utc)
    if sd.isna().any():
        raise RuntimeError(f"{city}: snow depth missing for {sd.isna().sum()} rows")
    sd_cm = np.round(sd.to_numpy() * 100, 1)

    pos = list(out.columns).index("snowfall_cm") + 1
    out.insert(pos, "precipitation_mm", [f"{v:.1f}" for v in precip])
    out.insert(pos + 1, "snow_depth_cm", [f"{v:.1f}" for v in sd_cm])
    out.to_csv(f"{out_dir}/{c['file']}", index=False)
    return raw, out, utc


WINTER_MONTHS = (11, 12, 1, 2, 3)


def _fmt(v):
    t = f"{v:.6f}".rstrip("0")
    return t + "0" if t.endswith(".") else t


def build_seoul(data_dir, out_dir):
    raw = pd.read_csv(f"{data_dir}/SeoulBikeData.csv", dtype=str, keep_default_na=False)
    if "precipitation_mm" in raw.columns:
        raise RuntimeError("SeoulBikeData.csv is already a v7 file: use the original data as input")
    ts = pd.to_datetime(raw["Date"])
    if not (ts.diff().dropna() == pd.Timedelta(hours=1)).all():
        raise RuntimeError("Seoul: rows are not consecutive hours")
    r = raw["Rainfall"].astype(float).to_numpy()
    winter = ts.dt.month.isin(WINTER_MONTHS).to_numpy()
    slot = (ts.dt.hour % 3 == 0).to_numpy()
    if ((r > 0) & winter & ~slot).any():
        raise RuntimeError("Seoul: winter precipitation outside the 3-hour slots")
    p = r.copy()
    split = np.zeros(len(r), dtype=bool)
    for i in np.flatnonzero((r > 0) & winter & slot):
        if i < 2 or r[i - 1] > 0 or r[i - 2] > 0:
            raise RuntimeError(f"Seoul: cannot split the 3-hour total in row {i}")
        p[i - 2:i + 1] = r[i] / 3.0
        split[i - 2:i + 1] = True
    out = raw.copy()
    pos = list(out.columns).index("Snowfall") + 1
    # unsplit rows keep the original text of Rainfall
    out.insert(pos, "precipitation_mm", [_fmt(v) if sp else txt
                                         for v, sp, txt in zip(p, split, raw["Rainfall"])])
    out.insert(pos + 1, "snow_depth_cm", raw["Snowfall"].to_numpy())
    out.to_csv(f"{out_dir}/SeoulBikeData.csv", index=False)
    return raw, out


if __name__ == "__main__":
    # python build_weather_v7.py <folder with the ORIGINAL (pre-v7) CSVs>
    #                              <folder with the two Open-Meteo downloads> <output folder>
    data_dir, om_dir, out_dir = sys.argv[1:4]
    for city in CITIES:
        build(city, data_dir, om_dir, out_dir)
        print("written", city)
    build_seoul(data_dir, out_dir)
    print("written seoul")
