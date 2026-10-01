#!/usr/bin/env python3
"""
fetch_nwp_data.py - download NWP forecasts and observations for the
bike-demand forecast-error calibration (Seoul, London, Washington DC).

Run this on a computer with normal internet access:

    pip install requests eccodes        # eccodes only needed for the GEFS part
    python fetch_nwp_data.py            # everything, ~4 h budget for GEFS
    python fetch_nwp_data.py --check    # 1 test request per source, then exit

Everything is resumable: stop with Ctrl-C and start again, finished pieces
are skipped. At the end it writes  nwp_data/nwp_extracts.zip  (tens of MB)
- that zip is what goes back to Claude for the analysis.

Sources (all free, no account):
  1. Open-Meteo Single Runs API - ECMWF IFS HRES 9 km, every 00/12 UTC run
     since 2024-03-14, lead 0-192 h, 6 variables incl. visibility.
     https://open-meteo.com/en/docs/single-runs-api   (CC-BY 4.0, non-commercial)
  2. Open-Meteo Historical Weather API - ERA5 family reanalysis at the same
     points, for the data years and for 2024-2026 (reference / "truth" that
     matches how the London and DC training covariates were built).
     https://open-meteo.com/en/docs/historical-weather-api
  3. NOAA ISD hourly station observations: Seoul 47108, London Heathrow
     03772, Washington Reagan National 72405-13743.
     https://www.ncei.noaa.gov/data/global-hourly/access/
  4. NOAA GEFSv12 reforecast, control member c00, 00 UTC runs 2011-2018 (the
     years of the bike data), lead 3-180 h every 3 h, nearest 0.25 deg point.
     https://noaa-gefs-retrospective.s3.amazonaws.com  (NOAA Open Data)

Raw downloads are kept under nwp_data/raw (GEFS GRIB subsets can be large,
see --discard-grib). Every GEFS byte range is logged with its SHA-256 in
nwp_data/extracted/gefs_manifest.csv so each value can be re-downloaded.
"""

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import shutil
import signal
import sys
import threading
import time
import zipfile


def utcnow():
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    sys.exit("Please install requests first:  pip install requests")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
# Station coordinates (forecast = nearest model grid point to the station)
CITIES = {
    "seoul": dict(lat=37.5714, lon=126.9658, isd="47108099999",
                  name="Seoul (KMA ASOS 108 / WMO 47108)",
                  data_start="2017-12-01", data_end="2018-11-30"),
    "london": dict(lat=51.4790, lon=-0.4490, isd="03772099999",
                   name="London Heathrow (WMO 03772)",
                   data_start="2015-01-04", data_end="2017-01-03"),
    "washington": dict(lat=38.8483, lon=-77.0342, isd="72405013743",
                       name="Washington Reagan National (WMO 72405 / WBAN 13743)",
                       data_start="2011-01-01", data_end="2012-12-31"),
}

OM_SINGLE_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
OM_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ISD_URL = "https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{station}.csv"
GEFS_URL = ("https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast/"
            "{y}/{ymdh}/c00/Days:1-10/{var}_{ymdh}_c00.grib2")

OM_SINGLE_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                  "shortwave_radiation", "precipitation", "visibility"]
OM_ARCHIVE_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                   "shortwave_radiation", "precipitation"]      # archive has no visibility
OM_REPLICA_VARS = OM_ARCHIVE_VARS + ["rain", "snowfall"]

ECMWF_FIRST_RUN = dt.datetime(2024, 3, 14, 0)
SINGLE_RUN_HOURS = 192          # request lead 0..192 h of each run
LAG_DAYS_FOR_OBS = 13           # last run used = today - 13 days (192 h + ERA5 delay)

# GEFS: fields = (file variable, GRIB name, GRIB level text)
GEFS_FIELDS = [
    ("tmp_2m", "TMP", "2 m above ground"),
    ("spfh_2m", "SPFH", "2 m above ground"),
    ("pres_sfc", "PRES", "surface"),
    ("ugrd_hgt", "UGRD", "10 m above ground"),
    ("vgrd_hgt", "VGRD", "10 m above ground"),
    ("dswrf_sfc", "DSWRF", "surface"),
    ("apcp_sfc", "APCP", "surface"),
]
GEFS_MAX_LEAD = 180             # 168 h horizon + up to 12 h model-run age

# Open-Meteo free tier: 600/min, 5000/h, 10000/day (weighted calls).
OM_SECONDS_PER_CALL = 0.8       # -> at most 4500 weighted calls per hour

UA = {"User-Agent": "nwp-error-calibration-research-script/1.0 (non-commercial)"}

STOP = threading.Event()
LOG_LOCK = threading.Lock()
LOGFILE = None


def log(msg):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    with LOG_LOCK:
        print(line, flush=True)
        if LOGFILE:
            with open(LOGFILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def _sleep_interruptible(seconds):
    end = time.time() + seconds
    while time.time() < end:
        if STOP.is_set():
            raise KeyboardInterrupt
        time.sleep(min(5.0, end - time.time()))


def http_get(session, url, params=None, headers=None, timeout=120, tries=6,
             ok_status=(200, 206)):
    """GET with retries/backoff. Returns the response (also for 400/403/404,
    the caller decides) or raises after `tries` failed attempts. Rate-limit
    answers (429) are waited out and do not count as failed attempts."""
    wait = 5
    last = RuntimeError(f"no response from {url}")
    attempt = 0
    while attempt < tries:
        if STOP.is_set():
            raise KeyboardInterrupt
        try:
            r = session.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in ok_status:
                return r
            if r.status_code == 429:
                txt = (r.text or "").lower()
                if "daily" in txt:
                    now = utcnow()
                    nxt = (now + dt.timedelta(days=1)).replace(hour=0, minute=5, second=0,
                                                               microsecond=0)
                    log(f"  Open-Meteo DAILY limit reached - waiting until {nxt:%H:%M} UTC "
                        "(or press Ctrl-C and simply run the script again later)")
                    _sleep_interruptible((nxt - now).total_seconds())
                elif "hour" in txt:
                    log("  Open-Meteo hourly limit reached - waiting 10 min")
                    _sleep_interruptible(600)
                else:
                    log("  HTTP 429 (rate limit) - waiting 60 s")
                    _sleep_interruptible(60)
                continue
            if r.status_code in (400, 403, 404):
                return r
            last = RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        except requests.RequestException as e:
            last = e
        attempt += 1
        _sleep_interruptible(wait)
        wait = min(wait * 2, 300)
    raise last


def atomic_write(path, data, mode="wb"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, mode) as f:
        f.write(data)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# 1+2. Open-Meteo (single runs + archive), strictly sequential & rate-limited
# --------------------------------------------------------------------------
class OMLimiter:
    """Thread-safe spacing of (weighted) Open-Meteo calls."""

    def __init__(self):
        self.next_ok = 0.0
        self.lock = threading.Lock()

    def wait(self, weight):
        with self.lock:
            now = time.time()
            start = max(now, self.next_ok)
            self.next_ok = start + OM_SECONDS_PER_CALL * weight
        if start > now:
            time.sleep(start - now)


def om_weight(n_vars, n_days):
    """Open-Meteo counts >10 variables or >2 weeks as multiple calls."""
    return max(1.0, n_vars / 10.0) * max(1.0, n_days / 14.0)


def om_request(session, limiter, url, params, weight):
    limiter.wait(weight)
    r = http_get(session, url, params=params, headers=UA, timeout=180)
    try:
        js = r.json()
    except ValueError:
        js = {"error": True, "reason": f"HTTP {r.status_code}, non-JSON: {r.text[:200]}"}
    if r.status_code != 200 and "error" not in js:
        js = {"error": True, "reason": f"HTTP {r.status_code}: {r.text[:200]}"}
    return js


def single_run_list(end_run):
    runs = []
    t = ECMWF_FIRST_RUN
    while t <= end_run:
        runs.append(t)
        t += dt.timedelta(hours=12)
    return runs


def fetch_single_run(session, limiter, city, c, run, outdir):
    path = os.path.join(outdir, "raw", "openmeteo_single_runs", city,
                        f"{run:%Y%m%d%H}.json")
    if os.path.exists(path):
        return "cached"
    # The Single Runs API rejects start_hour/end_hour; forecast_days=10 returns
    # the whole 10-day run (the analysis keeps leads 1-192 h).
    params = dict(latitude=c["lat"], longitude=c["lon"], models="ecmwf_ifs",
                  hourly=",".join(OM_SINGLE_VARS), run=f"{run:%Y-%m-%dT%H:%M}",
                  forecast_days=10, wind_speed_unit="ms", cell_selection="nearest",
                  elevation="nan", timezone="GMT")
    js = om_request(session, limiter, OM_SINGLE_URL, params, 1.0)
    if js.get("error") and "forecast_days" in str(js.get("reason", "")):
        params.pop("forecast_days")
        js = om_request(session, limiter, OM_SINGLE_URL, params, 1.0)
    js["_request"] = {k: str(v) for k, v in params.items()}
    js["_url"] = OM_SINGLE_URL
    js["_downloaded_utc"] = utcnow().isoformat()
    if js.get("error"):
        # keep the error message (the run is retried on the next start)
        atomic_write(path.replace(".json", ".error.json"),
                     json.dumps(js).encode())
        return "error: " + str(js.get("reason"))[:120]
    atomic_write(path, json.dumps(js).encode())
    stale = path.replace(".json", ".error.json")
    if os.path.exists(stale):
        os.remove(stale)
    return "ok"


def year_chunks(start, end):
    """Split [start, end] (dates) into calendar-year chunks."""
    out = []
    s = start
    while s <= end:
        e = min(dt.date(s.year, 12, 31), end)
        out.append((s, e))
        s = e + dt.timedelta(days=1)
    return out


def fetch_archive(session, limiter, city, c, tag, model, start, end, variables,
                  outdir, raw_grid=True):
    results = []
    for s, e in year_chunks(start, end):
        path = os.path.join(outdir, "raw", "openmeteo_archive", city,
                            f"{tag}_{s:%Y%m%d}_{e:%Y%m%d}.json")
        if os.path.exists(path):
            results.append("cached"); continue
        params = dict(latitude=c["lat"], longitude=c["lon"],
                      start_date=f"{s:%Y-%m-%d}", end_date=f"{e:%Y-%m-%d}",
                      hourly=",".join(variables), wind_speed_unit="ms",
                      timezone="GMT")
        if model:
            params["models"] = model
        if raw_grid:
            params["cell_selection"] = "nearest"
            params["elevation"] = "nan"
        w = om_weight(len(variables), (e - s).days + 1)
        js = om_request(session, limiter, OM_ARCHIVE_URL, params, w)
        js["_request"] = {k: str(v) for k, v in params.items()}
        js["_url"] = OM_ARCHIVE_URL
        js["_downloaded_utc"] = utcnow().isoformat()
        if js.get("error"):
            log(f"  archive {city} {tag} {s}..{e}: ERROR {js.get('reason')}")
            results.append("error"); continue
        atomic_write(path, json.dumps(js).encode())
        results.append("ok")
    return results


def run_openmeteo(outdir, end_run, do_archive=True, do_single=True):
    session = requests.Session()
    limiter = OMLimiter()
    today = dt.date.today()
    if do_archive:
        log("Open-Meteo archive (reanalysis reference) ...")
        for city, c in CITIES.items():
            d0 = dt.date.fromisoformat(c["data_start"])
            d1 = dt.date.fromisoformat(c["data_end"]) + dt.timedelta(days=8)
            r0 = ECMWF_FIRST_RUN.date()
            r1 = min(end_run.date() + dt.timedelta(days=9), today - dt.timedelta(days=6))
            jobs = [
                # same source family as the London/DC training covariates
                ("datayears_era5seamless", "era5_seamless", d0, d1, OM_ARCHIVE_VARS, True),
                ("recent_era5seamless", "era5_seamless", r0, r1, OM_ARCHIVE_VARS, True),
                # IFS-analysis based best match (2017+), higher resolution
                ("recent_bestmatch", "best_match", r0, r1, OM_ARCHIVE_VARS, True),
                # default settings, to check against the columns in your datasets
                ("datayears_default_replica", None, d0, d1, OM_REPLICA_VARS, False),
            ]
            for tag, model, s, e, vars_, raw in jobs:
                if STOP.is_set():
                    return
                res = fetch_archive(session, limiter, city, c, tag, model, s, e,
                                    vars_, outdir, raw_grid=raw)
                log(f"  {city:10s} {tag:28s} {res}")
    if do_single:
        runs = single_run_list(end_run)
        n = len(runs) * len(CITIES)
        log(f"Open-Meteo single runs: ECMWF IFS 9 km, {len(runs)} runs x "
            f"{len(CITIES)} cities = {n} requests "
            f"(~{n * OM_SECONDS_PER_CALL / 3600:.1f} h because of the free-tier rate limit)")
        done = 0
        t0 = time.time()
        errors = 0
        tl = threading.local()

        def one(args):
            run, city = args
            if STOP.is_set():
                return run, city, "stopped"
            if not hasattr(tl, "s"):
                tl.s = requests.Session()
            try:
                return run, city, fetch_single_run(tl.s, limiter, city, CITIES[city], run,
                                                   outdir)
            except KeyboardInterrupt:
                return run, city, "stopped"
            except Exception as e:
                return run, city, f"error {type(e).__name__}: {str(e)[:100]}"

        jobs = [(run, city) for run in runs for city in CITIES]
        # 3 connections share one limiter: the free-tier rate limit sets the pace,
        # not the (slower) archive response time.
        with ThreadPoolExecutor(max_workers=3) as ex:
            for run, city, st in ex.map(one, jobs):
                done += 1
                if st.startswith("error"):
                    errors += 1
                    log(f"  {city} {run:%Y-%m-%d %H}Z: {st}")
                if done % 150 == 0:
                    el = time.time() - t0
                    eta = el / done * (n - done)
                    log(f"  single runs {done}/{n}  ({errors} errors)  "
                        f"ETA {eta / 3600:.1f} h")
        log(f"Open-Meteo single runs finished ({errors} errors).")


# --------------------------------------------------------------------------
# 3. NOAA ISD station data
# --------------------------------------------------------------------------
def isd_years(c, end_run):
    d0 = dt.date.fromisoformat(c["data_start"])
    d1 = dt.date.fromisoformat(c["data_end"]) + dt.timedelta(days=8)
    years = set(range(d0.year, d1.year + 1))
    years |= set(range(ECMWF_FIRST_RUN.year, dt.date.today().year + 1))
    return sorted(years)


def run_isd(outdir, end_run):
    session = requests.Session()
    log("NOAA ISD station data ...")
    for city, c in CITIES.items():
        for y in isd_years(c, end_run):
            path = os.path.join(outdir, "raw", "isd", f"{c['isd']}_{y}.csv")
            if os.path.exists(path) and y < dt.date.today().year:
                continue
            url = ISD_URL.format(year=y, station=c["isd"])
            try:
                r = http_get(session, url, headers=UA, timeout=300)
            except Exception as e:
                log(f"  ISD {city} {y}: FAILED {e}")
                continue
            if r.status_code != 200:
                log(f"  ISD {city} {y}: HTTP {r.status_code} (no file)")
                continue
            atomic_write(path, r.content)
            log(f"  ISD {city} {y}: {len(r.content) / 1e6:.1f} MB")


# --------------------------------------------------------------------------
# 4. GEFSv12 reforecast (control member), byte-range GRIB subsets
# --------------------------------------------------------------------------
IDX_TIME = re.compile(r"^(?:(\d+)-)?(\d+) (hour|day) (?:(acc|ave|max|min) )?fcst$")


def parse_idx(text):
    """wgrib2-style index -> list of dicts with byte ranges and step info."""
    rows = []
    lines = [l for l in text.splitlines() if l.strip()]
    for i, line in enumerate(lines):
        p = line.split(":")
        if len(p) < 6:
            continue
        off = int(p[1])
        nxt = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else None
        ft = p[5].strip()
        m = IDX_TIME.match(ft)
        if ft == "anl":
            s0 = s1 = 0; kind = "instant"
        elif m:
            mult = 24 if m.group(3) == "day" else 1
            s1 = int(m.group(2)) * mult
            s0 = int(m.group(1)) * mult if m.group(1) else s1
            kind = m.group(4) or "instant"
        else:
            continue
        rows.append(dict(n=int(p[0]), start=off, end=nxt, var=p[3], level=p[4],
                         step0=s0, step1=s1, kind=kind, line=line))
    return rows


def merge_ranges(rows):
    """Merge adjacent byte ranges to reduce the number of HTTP requests."""
    rows = sorted(rows, key=lambda r: r["start"])
    groups = []
    for r in rows:
        if groups and groups[-1][-1]["end"] is not None and \
                groups[-1][-1]["end"] + 1 == r["start"]:
            groups[-1].append(r)
        else:
            groups.append([r])
    return groups


class GribDecoder:
    """Nearest-grid-point values from single GRIB2 messages (eccodes or pygrib).

    For regular lat/lon grids the nearest point is computed from the grid
    definition (exact, fast); other grids fall back to eccodes' find_nearest.
    Decoding is serialised with a lock (ecCodes is not guaranteed thread-safe).
    """
    LOCK = threading.Lock()

    def __init__(self):
        self.backend = None
        try:
            import eccodes  # noqa
            self.ec = eccodes
            self.backend = "eccodes"
        except Exception:
            try:
                import pygrib  # noqa
                import numpy  # noqa
                self.pg = pygrib
                self.np = numpy
                self.backend = "pygrib"
            except Exception:
                self.backend = None

    @staticmethod
    def _regular_index(g, lat, lon):
        """g: dict of grid keys. Returns (flat index, grid lat, grid lon)."""
        ni, nj = g["Ni"], g["Nj"]
        la0, lo0 = g["la0"], g["lo0"]
        di, dj = g["di"], g["dj"]
        if g["jpos"]:
            j = int(round((lat - la0) / dj))
            glat = la0 + j * dj
        else:
            j = int(round((la0 - lat) / dj))
            glat = la0 - j * dj
        if g["ineg"]:
            i = int(round(((lo0 - lon) % 360.0) / di))
            glon = lo0 - i * di
        else:
            i = int(round(((lon - lo0) % 360.0) / di))
            glon = lo0 + i * di
        if i >= ni:          # wrap-around at the date line
            i -= ni
        if not (0 <= i < ni and 0 <= j < nj):
            raise ValueError("point outside grid")
        glon = ((glon + 180.0) % 360.0) - 180.0
        return j * ni + i, glat, glon

    def points(self, msg_bytes, pts):
        """pts: {name: (lat, lon)} -> ({name: (value, glat, glon, dist_km)}, meta)."""
        with self.LOCK:
            if self.backend == "eccodes":
                return self._points_eccodes(msg_bytes, pts)
            if self.backend == "pygrib":
                return self._points_pygrib(msg_bytes, pts)
        raise RuntimeError("No GRIB decoder: pip install eccodes  (or pygrib)")

    def _points_eccodes(self, msg_bytes, pts):
        ec = self.ec
        gid = ec.codes_new_from_message(msg_bytes)
        try:
            meta = dict(shortName=ec.codes_get(gid, "shortName"),
                        stepRange=str(ec.codes_get(gid, "stepRange")),
                        gridType=ec.codes_get(gid, "gridType"))
            out = {}
            if meta["gridType"] == "regular_ll":
                g = dict(Ni=ec.codes_get(gid, "Ni"), Nj=ec.codes_get(gid, "Nj"),
                         la0=ec.codes_get(gid, "latitudeOfFirstGridPointInDegrees"),
                         lo0=ec.codes_get(gid, "longitudeOfFirstGridPointInDegrees"),
                         di=ec.codes_get(gid, "iDirectionIncrementInDegrees"),
                         dj=ec.codes_get(gid, "jDirectionIncrementInDegrees"),
                         jpos=ec.codes_get(gid, "jScansPositively") == 1,
                         ineg=ec.codes_get(gid, "iScansNegatively") == 1)
                vals = ec.codes_get_values(gid)
                for name, (la, lo) in pts.items():
                    k, gla, glo = self._regular_index(g, la, lo)
                    out[name] = (float(vals[k]), gla, glo, haversine(la, lo, gla, glo))
            else:
                for name, (la, lo) in pts.items():
                    nn = ec.codes_grib_find_nearest(gid, la, lo)[0]
                    out[name] = (float(nn["value"]), float(nn["lat"]),
                                 float(nn["lon"]), float(nn["distance"]))
            return out, meta
        finally:
            ec.codes_release(gid)

    def _points_pygrib(self, msg_bytes, pts):
        m = self.pg.fromstring(msg_bytes)
        meta = dict(shortName=m.shortName, stepRange=str(m.stepRange),
                    gridType=m.gridType)
        la, lo = m.latlons()
        vals = m.values
        out = {}
        for name, (pla, plo) in pts.items():
            d = haversine_np(self.np, la, lo, pla, plo)
            k = int(self.np.argmin(d))
            out[name] = (float(vals.flat[k]), float(la.flat[k]),
                         float(((lo.flat[k] + 180) % 360) - 180), float(d.flat[k]))
        return out, meta


def haversine(la1, lo1, la2, lo2):
    import math
    la1, lo1, la2, lo2 = map(math.radians, (la1, lo1, la2, lo2))
    a = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def haversine_np(np, la, lo, pla, plo):
    la1, lo1, la2, lo2 = map(np.radians, (la, lo, pla, plo))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


GEFS_POINT_HEADER = ["city", "init_utc", "file_var", "step_start_h", "step_end_h",
                     "kind", "value", "grid_lat", "grid_lon"]
MANIFEST_HEADER = ["init_utc", "url", "byte_start", "byte_end", "n_messages",
                   "sha256", "idx_lines"]


def city_for_date(d):
    for city, c in CITIES.items():
        if dt.date.fromisoformat(c["data_start"]) <= d <= dt.date.fromisoformat(c["data_end"]):
            return city
    return None


def expected_step_range(x):
    return str(x["step1"]) if x["kind"] == "instant" else f"{x['step0']}-{x['step1']}"


def gefs_one_run(init, outdir, decoder, keep_grib, session):
    ymdh = f"{init:%Y%m%d%H}"
    done_flag = os.path.join(outdir, "extracted", "gefs_points", f"{ymdh}.done")
    if os.path.exists(done_flag):
        return "cached", 0
    city = city_for_date(init.date())
    pts = {city: (CITIES[city]["lat"], CITIES[city]["lon"])}
    rows_out, manifest, nbytes, mismatches = [], [], 0, []
    for fvar, gvar, glev in GEFS_FIELDS:
        url = GEFS_URL.format(y=init.year, ymdh=ymdh, var=fvar)
        r = http_get(session, url + ".idx", headers=UA, timeout=60)
        if r.status_code != 200:
            return f"missing idx {fvar} (HTTP {r.status_code})", nbytes
        idx = parse_idx(r.text)
        sel = [x for x in idx if x["var"] == gvar and x["level"] == glev
               and 0 < x["step1"] <= GEFS_MAX_LEAD]
        if not sel:
            return f"no messages for {fvar} in idx", nbytes
        grib_parts = []
        for grp in merge_ranges(sel):
            b0 = grp[0]["start"]
            b1 = grp[-1]["end"]
            rng = f"bytes={b0}-{'' if b1 is None else b1}"
            rr = http_get(session, url, headers=dict(UA, Range=rng), timeout=300)
            if rr.status_code not in (200, 206):
                return f"range fetch failed {fvar} HTTP {rr.status_code}", nbytes
            data = rr.content
            if b1 is not None and len(data) != b1 - b0 + 1:
                return f"short read {fvar}: {len(data)} of {b1 - b0 + 1} bytes", nbytes
            nbytes += len(data)
            manifest.append([init.isoformat(), url, b0, b1 if b1 is not None else "",
                             len(grp), hashlib.sha256(data).hexdigest(),
                             " | ".join(x["line"] for x in grp)])
            grib_parts.append(data)
            # split into single GRIB messages using the idx offsets
            for x in grp:
                s = x["start"] - b0
                e = (x["end"] - b0 + 1) if x["end"] is not None else len(data)
                vals, meta = decoder.points(data[s:e], pts)
                grib_range = "-".join(t.rstrip("h") for t in str(meta["stepRange"]).split("-"))
                if grib_range != expected_step_range(x):
                    mismatches.append(f"{fvar} idx={expected_step_range(x)} "
                                      f"grib={meta['stepRange']}")
                for pname, (v, gla, glo, dist) in vals.items():
                    rows_out.append([pname, init.strftime("%Y-%m-%dT%H:%M"), fvar,
                                     x["step0"], x["step1"], x["kind"], repr(v),
                                     round(gla, 4), round(glo, 4)])
        if keep_grib:
            atomic_write(os.path.join(outdir, "raw", "gefs", f"{init:%Y}", ymdh,
                                      f"{fvar}_{ymdh}_c00_sel.grib2"),
                         b"".join(grib_parts))
    if mismatches:
        # never silently accept values whose time window we cannot confirm
        return "step mismatch between idx and GRIB: " + "; ".join(mismatches[:3]), nbytes
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(GEFS_POINT_HEADER)
    w.writerows(rows_out)
    atomic_write(os.path.join(outdir, "extracted", "gefs_points", f"{ymdh}.csv"),
                 buf.getvalue().encode())
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(MANIFEST_HEADER)
    w.writerows(manifest)
    atomic_write(os.path.join(outdir, "extracted", "gefs_manifest", f"{ymdh}.csv"),
                 buf.getvalue().encode())
    atomic_write(done_flag, b"ok\n")
    return "ok", nbytes


def gefs_schedule():
    """All 00 UTC init dates of the three data periods, ordered so that
    every 4th day comes first, then the days in between (denser passes)."""
    per_pass = {0: [], 2: [], 1: [], 3: []}
    for city, c in CITIES.items():
        d0 = dt.date.fromisoformat(c["data_start"])
        d1 = dt.date.fromisoformat(c["data_end"])
        d = d0
        i = 0
        while d <= d1:
            per_pass[i % 4].append(dt.datetime(d.year, d.month, d.day, 0))
            d += dt.timedelta(days=1)
            i += 1
    out = []
    for k in (0, 2, 1, 3):
        out += sorted(per_pass[k])
    return out


def run_gefs(outdir, hours_budget, workers, keep_grib, t_start):
    decoder = GribDecoder()
    if decoder.backend is None:
        log("GEFS: no GRIB decoder found -> skipped. Install with:  pip install eccodes")
        return
    log(f"GEFS reforecast (GRIB decoder: {decoder.backend}), budget "
        f"{hours_budget:.1f} h, {workers} parallel downloads")
    sched = gefs_schedule()
    todo = [d for d in sched if not os.path.exists(os.path.join(
        outdir, "extracted", "gefs_points", f"{d:%Y%m%d%H}.done"))]
    log(f"  {len(sched)} runs in the data periods, {len(todo)} still to do")
    tl = threading.local()

    def sess():
        if not hasattr(tl, "s"):
            tl.s = requests.Session()
        return tl.s

    total_bytes = 0
    n_ok = 0
    t0 = time.time()
    keep = [keep_grib]

    def job(init):
        if STOP.is_set() or (time.time() - t_start) / 3600 > hours_budget:
            return init, "skipped (budget)", 0
        if keep[0]:
            free = shutil.disk_usage(outdir).free / 1e9
            if free < 20:
                keep[0] = False
                log(f"  only {free:.0f} GB free disk -> no longer keeping GEFS GRIB "
                    "subsets (byte ranges + SHA-256 stay in the manifest)")
        try:
            st, nb = gefs_one_run(init, outdir, decoder, keep[0], sess())
        except KeyboardInterrupt:
            return init, "stopped", 0
        except Exception as e:
            st, nb = f"error {type(e).__name__}: {str(e)[:150]}", 0
        return init, st, nb

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(job, d) for d in todo]
        for f in as_completed(futs):
            init, st, nb = f.result()
            total_bytes += nb
            if st == "ok":
                n_ok += 1
                if n_ok % 20 == 0:
                    el = time.time() - t0
                    rate = total_bytes / el / 1e6
                    log(f"  GEFS {n_ok} runs done, {total_bytes / 1e9:.1f} GB, "
                        f"{rate:.1f} MB/s, {n_ok / el * 3600:.0f} runs/h")
            elif not st.startswith("skipped") and st != "cached":
                log(f"  GEFS {init:%Y-%m-%d}: {st}")
    log(f"GEFS finished: {n_ok} new runs this session, {total_bytes / 1e9:.1f} GB")


# --------------------------------------------------------------------------
# Packing: the small zip that goes back for the analysis
# --------------------------------------------------------------------------
ISD_KEEP_COLS = ["STATION", "DATE", "SOURCE", "REPORT_TYPE", "QUALITY_CONTROL",
                 "LATITUDE", "LONGITUDE", "ELEVATION", "NAME", "WND", "VIS", "TMP",
                 "DEW", "AA1", "AA2", "AA3", "AA4", "AJ1", "MW1", "MW2", "AU1",
                 "AU2", "AW1", "AW2", "GF1"]


def pack(outdir):
    log("Packing nwp_extracts.zip ...")
    zpath = os.path.join(outdir, "nwp_extracts.zip")
    with zipfile.ZipFile(zpath + ".part", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        # Open-Meteo raw JSON (small)
        for sub in ("openmeteo_single_runs", "openmeteo_archive"):
            root = os.path.join(outdir, "raw", sub)
            for dp, _, fns in os.walk(root):
                for fn in sorted(fns):
                    if fn.endswith(".json"):
                        full = os.path.join(dp, fn)
                        z.write(full, os.path.relpath(full, outdir))
        # ISD: trimmed to the relevant columns (raw strings, no parsing)
        root = os.path.join(outdir, "raw", "isd")
        if os.path.isdir(root):
            for fn in sorted(os.listdir(root)):
                if not fn.endswith(".csv"):
                    continue
                with open(os.path.join(root, fn), newline="", encoding="utf-8",
                          errors="replace") as f:
                    rd = csv.DictReader(f)
                    cols = [c for c in ISD_KEEP_COLS if c in (rd.fieldnames or [])]
                    buf = io.StringIO()
                    w = csv.writer(buf)
                    w.writerow(cols)
                    for row in rd:
                        w.writerow([row.get(c, "") for c in cols])
                z.writestr(f"isd_trimmed/{fn}", buf.getvalue())
        # GEFS point values + manifests
        root = os.path.join(outdir, "extracted", "gefs_points")
        if os.path.isdir(root):
            for fn in sorted(os.listdir(root)):
                if fn.endswith(".csv"):
                    z.write(os.path.join(root, fn), f"gefs_points/{fn}")
        # log + this script
        if LOGFILE and os.path.exists(LOGFILE):
            z.write(LOGFILE, "fetch_log.txt")
        z.write(os.path.abspath(__file__), "fetch_nwp_data.py")
        z.writestr("environment.json", json.dumps(dict(
            python=sys.version, platform=sys.platform,
            packed_utc=utcnow().isoformat(),
            requests=requests.__version__), indent=1))
    os.replace(zpath + ".part", zpath)
    log(f"Done: {zpath}  ({os.path.getsize(zpath) / 1e6:.1f} MB) - attach this file.")


# --------------------------------------------------------------------------
# Connectivity check
# --------------------------------------------------------------------------
def check(outdir):
    s = requests.Session()
    ok = True
    c = CITIES["london"]
    print("Checking sources (1 small request each) ...")
    try:
        p = dict(latitude=c["lat"], longitude=c["lon"], models="ecmwf_ifs",
                 hourly=",".join(OM_SINGLE_VARS), run="2024-06-01T12:00",
                 forecast_days=10, wind_speed_unit="ms", cell_selection="nearest",
                 elevation="nan", timezone="GMT")
        js = s.get(OM_SINGLE_URL, params=p, headers=UA, timeout=60).json()
        if js.get("error"):
            print("  Open-Meteo single runs : ERROR", js.get("reason")); ok = False
        else:
            h = js["hourly"]
            nn = {v: sum(x is not None for x in h.get(v, [])) for v in OM_SINGLE_VARS}
            first = next((t for t, v in zip(h["time"], h["temperature_2m"]) if v is not None), None)
            last = [t for t, v in zip(h["time"], h["temperature_2m"]) if v is not None][-1:]
            print("  Open-Meteo single runs : OK, run 2024-06-01 12Z has data from", first,
                  "to", last[0] if last else None, "| values per variable:", nn)
    except Exception as e:
        print("  Open-Meteo single runs : FAILED", e); ok = False
    try:
        p = dict(latitude=c["lat"], longitude=c["lon"], start_date="2015-06-01",
                 end_date="2015-06-01", hourly=",".join(OM_ARCHIVE_VARS),
                 models="era5_seamless", wind_speed_unit="ms")
        js = s.get(OM_ARCHIVE_URL, params=p, headers=UA, timeout=60).json()
        print("  Open-Meteo archive     :", "ERROR " + str(js.get("reason")) if js.get("error")
              else f"OK, {len(js['hourly']['time'])} hours")
    except Exception as e:
        print("  Open-Meteo archive     : FAILED", e); ok = False
    try:
        r = s.get(ISD_URL.format(year=2015, station=c["isd"]), headers=dict(UA, Range="bytes=0-2000"),
                  timeout=60)
        print("  NOAA ISD               :", "OK" if r.status_code in (200, 206)
              else f"HTTP {r.status_code}", r.text.splitlines()[0][:100])
    except Exception as e:
        print("  NOAA ISD               : FAILED", e); ok = False
    try:
        url = GEFS_URL.format(y=2015, ymdh="2015060100", var="tmp_2m") + ".idx"
        r = s.get(url, headers=UA, timeout=60)
        rows = parse_idx(r.text) if r.status_code == 200 else []
        print("  GEFS reforecast idx    :", f"OK, {len(rows)} messages, first:",
              rows[0]["line"] if rows else f"HTTP {r.status_code}")
        if rows:
            x = rows[1]
            rr = s.get(url[:-4], headers=dict(UA, Range=f"bytes={x['start']}-{x['end']}"), timeout=120)
            dec = GribDecoder()
            if dec.backend:
                vals, meta = dec.points(rr.content, {"london": (c["lat"], c["lon"])})
                print(f"  GEFS decode ({dec.backend:7s}) : OK, message {len(rr.content)/1e6:.2f} MB,",
                      meta, vals)
            else:
                print("  GEFS decode            : no decoder -> pip install eccodes"); ok = False
    except Exception as e:
        print("  GEFS reforecast        : FAILED", e); ok = False
    print("All sources OK." if ok else "Some sources failed - see above.")
    return ok


# --------------------------------------------------------------------------
def main():
    global LOGFILE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="nwp_data", help="output folder (default nwp_data)")
    ap.add_argument("--hours", type=float, default=4.0,
                    help="time budget for the GEFS part in hours (default 4). "
                         "Open-Meteo and ISD always complete.")
    ap.add_argument("--workers", type=int, default=6, help="parallel GEFS downloads")
    ap.add_argument("--discard-grib", action="store_true",
                    help="do not keep GEFS GRIB subsets on disk (manifest keeps "
                         "URL + byte range + SHA-256 for exact re-download)")
    ap.add_argument("--only", default="isd,archive,single,gefs",
                    help="comma list of parts: isd,archive,single,gefs")
    ap.add_argument("--check", action="store_true", help="test each source and exit")
    ap.add_argument("--pack-only", action="store_true", help="only (re)build the zip")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    LOGFILE = os.path.join(a.out, "fetch_log.txt")
    if a.check:
        sys.exit(0 if check(a.out) else 1)
    if a.pack_only:
        pack(a.out); return

    def handler(sig, frm):
        if STOP.is_set():
            sys.exit(1)
        log("Stopping after current downloads (Ctrl-C again to abort hard) ...")
        STOP.set()
    signal.signal(signal.SIGINT, handler)

    parts = set(x.strip() for x in a.only.split(","))
    t_start = time.time()
    today = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    end_run = today - dt.timedelta(days=LAG_DAYS_FOR_OBS) + dt.timedelta(hours=12)
    log(f"Start. Output: {os.path.abspath(a.out)}; ECMWF runs "
        f"{ECMWF_FIRST_RUN:%Y-%m-%d} .. {end_run:%Y-%m-%d %H}Z")

    if "isd" in parts:
        run_isd(a.out, end_run)
    gefs_thread = None
    if "gefs" in parts:
        gefs_thread = threading.Thread(target=run_gefs, args=(
            a.out, a.hours, a.workers, not a.discard_grib, t_start), daemon=True)
        gefs_thread.start()
    if "archive" in parts or "single" in parts:
        try:
            run_openmeteo(a.out, end_run, "archive" in parts, "single" in parts)
        except KeyboardInterrupt:
            pass
    if gefs_thread:
        while gefs_thread.is_alive():
            gefs_thread.join(timeout=1.0)
    pack(a.out)
    log(f"Total time {(time.time() - t_start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
