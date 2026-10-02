"""
Measured NWP forecast-error model (per city).

Degrades observed weather covariates with forecast errors measured for each
city from real ECMWF IFS HRES 9 km forecasts (1,828 runs, Mar 2024 - Sep 2026,
Open-Meteo Single Runs API). The calibration files are built by
weather/nwp/build_nwp_calibration.py; see weather_methodology.md.

Default setting (fresh_forecast=True, remove_bias=True): an operator with an
up-to-date, locally corrected weather forecast.

How one test window is degraded
-------------------------------
1. Lead times.
   fresh_forecast=True (default): the weather forecast starts when the demand
   forecast is issued (one hour before the first test hour), so row i gets
   lead time i + 1 hours in every city. The errors are replayed from runs
   whose start hour (00 or 12 UTC) is closest to that issue time of day
   (at most 6 h apart).
   fresh_forecast=False: the newest ECMWF run available at the issue time is
   used (runs start at 00 and 12 UTC and are available `delay_h` = 6 h
   later); row i gets that run's lead time, 6-17 h + i + 1.
   Lead times count rows (time steps), not clock time, so a daylight-saving
   change inside the window does not shift them. The UTC issue time (first
   hour - 1 h) only selects the replayed run (start hour, day of year).
2. Temperature, humidity, wind speed: one real ECMWF run of the city is drawn
   (that start hour; start date within +/- `season_days` of the test window's
   day of year, any year) and its actual errors at those lead times are
   added: X' = X + e(run, lead). Growth with lead time, hour-to-hour
   persistence and the links between the three variables are those of the
   real forecasts.
   remove_bias=True (default): the average error of all candidate runs
   (same start hour and season) at each lead time is subtracted first, i.e.
   the systematic lean that a locally corrected forecast would not have
   (e.g. Seoul -1.5 C against the city-centre station) is removed:
   X' = X + e(run, lead) - mean_runs e(lead).
3. Solar radiation: X' = X * (1 + b(lead) + s(lead) * z), z AR(1) with the
   measured lag-1 correlation; b = 0 if remove_bias; 0 at night; capped at
   the training-fold cap.
4. Precipitation (total precipitation in mm, one decision per hour): miss rate
   and false-alarm ratio per lead time; misses and false alarms persist from
   hour to hour (latent AR(1) with the measured correlation); hits get the
   measured lognormal amount error (mean-preserving if remove_bias); false
   alarms the measured amount distribution. The false-alarm ratio is
   converted into a probability per dry hour with the wet-hour share of the
   training fold. Degraded amounts are capped at the training fold's maximum
   of the column (or the hour's measured amount if larger), as solar
   radiation is capped. seasonal_rain=True (default): miss rate, false-alarm ratio
   and hit amount error are those of the time of year (12 monthly bins,
   interpolated linearly by the day of year of the forecast start); False:
   year-round values. Reference: Seoul = the station, London and Washington =
   ERA5 (see build_nwp_calibration.py).
5. Visibility: lognormal error in log space, persistent (AR(1));
   mean-preserving if remove_bias. For a city whose covariate is capped
   (Seoul 20 km, Washington 16 km), an hour at the cap stays at the cap unless
   the forecast falls below it, which happens with the measured probability.
6. Snow depth (since 2 Oct 2026): persistence, i.e. the value of the last
   training hour (the issue time) for the whole window; no forecast errors of
   snow depth were measured. Snow depth is not precipitation (no event draw,
   not in the wet-hour share). There is no rain/snow phase correction.

remove_bias does not change event rates: precipitation misses and false
alarms, false-alarm amounts, and visibility falling below its cap stay as
measured, except: rain_frequency_unbiased=True (config default since 2 Oct
2026): false alarms are set so that the forecast is wet as often as the
observations (frequency bias 1; the false-alarm ratio then equals the miss
rate), i.e. the rain-frequency bias of the raw forecast is removed as well. Against the Seoul station the
raw forecast is wet about twice as often as observed; against ERA5 (London,
Washington) about 0.8-0.9 times.

noise_scale multiplies every error magnitude (additive errors, including
their bias if it is kept; the log errors of visibility and precipitation
amounts; the relative solar error). Detection (misses, false alarms,
visibility falling below the cap) and false-alarm amounts are not scaled.
The random numbers do not depend on noise_scale (common random numbers
across scenarios).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

MAX_LEAD = 192
TQW_TYPES = ("temperature", "humidity", "wind_speed")


def to_utc(timestamps, tz):
    """Naive local timestamps -> naive UTC. tz None = timestamps already UTC.
    Daylight saving: an ambiguous (repeated) hour is read as daylight-saving
    time (the first occurrence, the one load_and_prepare_data keeps), a
    non-existent hour is shifted forward. Used for the issue time (run
    selection) only; lead times count rows."""
    t = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if tz is None:
        return t
    loc = t.tz_localize(tz, ambiguous=np.ones(len(t), dtype=bool),
                        nonexistent="shift_forward")
    return loc.tz_convert("UTC").tz_localize(None)


def run_init_and_leads(times_utc, run_hours=(0, 12), delay_h=6):
    """
    NWP run used for a forecast of the hours `times_utc` (consecutive rows),
    and the lead time of every row. Issue time = first hour - 1 h (UTC); run =
    newest start in `run_hours` (UTC) with start + delay_h <= issue time.
    Row i (0-based) gets lead time run age + i + 1, run age = issue time - run
    start: counted in rows, not clock time (a daylight-saving change inside
    the window does not shift it). Only times_utc[0] is used.

    Returns (init: pd.Timestamp, leads: np.ndarray of int hours)
    """
    times_utc = pd.DatetimeIndex(times_utc)
    issue = times_utc[0] - pd.Timedelta(hours=1)
    latest = issue - pd.Timedelta(hours=delay_h)
    day = latest.normalize()
    starts = [day + pd.Timedelta(hours=h) for h in sorted(run_hours)]
    starts += [s - pd.Timedelta(days=1) for s in starts]
    init = max(s for s in starts if s <= latest)
    age = int(round((issue - init) / pd.Timedelta(hours=1)))
    leads = age + np.arange(1, len(times_utc) + 1)
    if leads.min() < 1 or leads.max() > MAX_LEAD:
        raise ValueError(f"lead times {leads.min()}..{leads.max()} h outside 1..{MAX_LEAD} h")
    return init, leads


def fresh_start_and_leads(times_utc, run_hours=(0, 12)):
    """
    Fresh forecast: the weather forecast starts at the issue time (first hour
    - 1 h), so row i (0-based) gets lead time i + 1: counted in rows, not
    clock time (a daylight-saving change inside the window does not shift
    it). Only times_utc[0] is used: the errors are replayed from runs whose
    start hour in `run_hours` (UTC) is closest to the issue time of day; on a
    tie the earlier start (before the issue time) is used.

    Returns (issue: pd.Timestamp, start_hour: int, leads: np.ndarray of int hours)
    """
    times_utc = pd.DatetimeIndex(times_utc)
    issue = times_utc[0] - pd.Timedelta(hours=1)
    leads = np.arange(1, len(times_utc) + 1)
    if leads.min() < 1 or leads.max() > MAX_LEAD:
        raise ValueError(f"lead times {leads.min()}..{leads.max()} h outside 1..{MAX_LEAD} h")
    t = issue.hour + issue.minute / 60.0
    # hours since each run start (0..24); smallest = closest start before t
    since = {h: (t - h) % 24 for h in run_hours}
    dist = {h: min(d, 24 - d) for h, d in since.items()}
    start_hour = min(run_hours, key=lambda h: (dist[h], since[h]))
    return issue, int(start_hour), leads


def ar1(n, phi, rng):
    """Standard-normal AR(1) series of length n with lag-1 correlation phi."""
    eps = rng.standard_normal(n)
    z = np.empty(n)
    if n == 0:
        return z
    phi = float(np.clip(phi, -0.999, 0.999))
    z[0] = eps[0]
    c = np.sqrt(1.0 - phi * phi)
    for i in range(1, n):
        z[i] = phi * z[i - 1] + c * eps[i]
    return z


class NWPErrorModel:
    """Per-city measured NWP error model loaded from a calibration .npz file."""

    def __init__(self, path):
        self.path = Path(path)
        z = np.load(self.path, allow_pickle=False)
        self.run_init = pd.DatetimeIndex(z["run_init_utc"].astype("datetime64[ns]"))
        self.tqw_errors = z["tqw_errors"].astype(float)          # [run, lead 0..192, 3]
        self.solar_bias = z["solar_bias_rel"]
        self.solar_sd = z["solar_sd_rel"]
        self.solar_phi = float(z["solar_phi"])
        self.miss_rate = z["precip_miss_rate"]
        self.far = z["precip_far"]
        self.hit_mean_log = z["precip_hit_mean_log"]
        self.hit_sd_log = z["precip_hit_sd_log"]
        self.fa_mean_log = float(z["precip_fa_mean_log"])
        self.fa_sd_log = float(z["precip_fa_sd_log"])
        self.rho_miss = float(z["precip_rho_miss"])
        self.rho_fa = float(z["precip_rho_fa"])
        self.hit_phi = float(z["precip_hit_phi"])
        # per time of year [12, lead] (calibration files from 2 Oct 2026 on)
        if "precip_miss_rate_seasonal" in z.files:
            self.season_doy = z["precip_season_doy"].astype(float)
            self.miss_rate_seasonal = z["precip_miss_rate_seasonal"]
            self.far_seasonal = z["precip_far_seasonal"]
            self.hit_mean_log_seasonal = z["precip_hit_mean_log_seasonal"]
            self.hit_sd_log_seasonal = z["precip_hit_sd_log_seasonal"]
        else:
            self.season_doy = None
        cap = float(z["vis_cap_km"])
        self.vis_cap = None if np.isnan(cap) else cap
        self.vis_below_mean_log = float(z["vis_below_mean_log"])
        self.vis_below_sd_log = float(z["vis_below_sd_log"])
        self.vis_at_cap_p_below = float(z["vis_at_cap_p_below"])
        self.vis_at_cap_depth_mean_log = float(z["vis_at_cap_depth_mean_log"])
        self.vis_at_cap_depth_sd_log = float(z["vis_at_cap_depth_sd_log"])
        self.vis_phi = float(z["vis_phi"])
        self.summary = str(z["summary_json"])
        self._doy = self.run_init.dayofyear.to_numpy()
        self._hour = self.run_init.hour.to_numpy()

    # ------------------------------------------------------------------
    def candidate_runs(self, start_hour, dayofyear, leads, season_days=30):
        """Runs starting at `start_hour` (UTC), start day of year within
        +/- season_days of `dayofyear` (any year) and complete errors at
        `leads`. The window is widened (x2) until a run is found."""
        complete = np.isfinite(self.tqw_errors[:, leads, :]).all(axis=(1, 2))
        same_hour = self._hour == start_hour
        d = np.abs(self._doy - dayofyear)
        d = np.minimum(d, 366 - d)
        days = season_days
        while True:
            idx = np.flatnonzero(complete & same_hour & (d <= days))
            if len(idx) or days >= 183:
                break
            days *= 2
        if len(idx) == 0:
            raise RuntimeError(f"no complete ECMWF run for start hour {start_hour} UTC "
                               f"and lead times {leads.min()}..{leads.max()} h in {self.path.name}")
        return idx, days

    # ------------------------------------------------------------------
    def degrade(self, df, times_utc, column_mapping, degradation_params, seed,
                noise_scale=1.0, run_hours=(0, 12), delay_h=6,
                season_days=30, fresh_forecast=True, remove_bias=True, seasonal_rain=True,
                rain_frequency_unbiased=False):
        """
        Degrade one test window. Returns (degraded DataFrame, info dict).
        df rows must be the consecutive test hours, times_utc their UTC times.
        fresh_forecast / remove_bias / seasonal_rain / rain_frequency_unbiased:
        see the module docstring.
        """
        if noise_scale < 0:
            raise ValueError(f"noise_scale must be >= 0, got {noise_scale}")
        rng = np.random.default_rng(seed)
        n = len(df)
        if fresh_forecast:
            start, start_hour, leads = fresh_start_and_leads(times_utc, run_hours)
        else:
            start, leads = run_init_and_leads(times_utc, run_hours, delay_h)
            start_hour = start.hour
        cand, days = self.candidate_runs(start_hour, start.dayofyear, leads, season_days)
        k = int(cand[rng.integers(len(cand))])
        e_tqw = self.tqw_errors[k][leads, :]                      # [n, 3]
        if remove_bias:
            # average error of the candidate runs (same start hour and season)
            e_tqw = e_tqw - self.tqw_errors[cand][:, leads, :].mean(axis=0)

        # Random numbers, always drawn in the same order (independent of the
        # columns present and of noise_scale)
        z_solar = ar1(n, self.solar_phi, rng)
        z_vis = ar1(n, self.vis_phi, rng)
        u_miss = ndtr(ar1(n, self.rho_miss, rng))
        u_fa = ndtr(ar1(n, self.rho_fa, rng))
        z_hit = ar1(n, self.hit_phi, rng)
        z_fa_amount = rng.standard_normal(n)

        s = float(noise_scale)
        out = df.copy()
        precip_cols = [c for c, t in column_mapping.items()
                       if t == "precipitation" and c in df.columns]
        for col, vtype in column_mapping.items():
            if col not in df.columns or vtype == "precipitation":
                continue
            x = df[col].to_numpy(dtype=float)
            if vtype in TQW_TYPES:
                y = x + s * e_tqw[:, TQW_TYPES.index(vtype)]
                if vtype == "humidity":
                    y = np.clip(y, 0.0, 100.0)
                elif vtype == "wind_speed":
                    y = np.maximum(y, 0.0)
            elif vtype == "solar_radiation":
                bias = 0.0 if remove_bias else self.solar_bias[leads]
                rel = bias + self.solar_sd[leads] * z_solar
                y = np.where(x > 0, x * (1.0 + s * rel), 0.0)
                y = np.clip(y, 0.0, degradation_params["solar_cap"])
            elif vtype == "visibility":
                y = self._visibility(x, z_vis, s, degradation_params.get("visibility_max"),
                                     remove_bias)
            elif vtype == "snow_depth":
                # persistence: snow depth at the issue time (last training
                # hour) for the whole window; no snow-depth errors measured
                persist = degradation_params.get("persist", {})
                if col not in persist:
                    raise ValueError(f"degradation_params has no persistence value for '{col}'")
                y = np.full(n, persist[col], dtype=float)
            else:
                raise ValueError(f"Unknown variable type: '{vtype}'")
            out[col] = y.astype(float)

        if precip_cols:
            if "wet_fraction" not in degradation_params:
                raise ValueError("degradation_params has no 'wet_fraction'")
            fa_col = precip_cols[0]   # one total-precipitation column per city
            params = self.rain_params(start.dayofyear, seasonal_rain)
            p = self._precipitation(df[precip_cols], leads, degradation_params["wet_fraction"],
                                    u_miss, u_fa, z_hit, z_fa_amount, s, fa_col, remove_bias, params,
                                    rain_frequency_unbiased, degradation_params.get("precip_max"))
            for c in precip_cols:
                out[c] = p[c].to_numpy(dtype=float)

        info = dict(fresh_forecast=bool(fresh_forecast), remove_bias=bool(remove_bias),
                    seasonal_rain=bool(seasonal_rain and self.season_doy is not None),
                    rain_frequency_unbiased=bool(rain_frequency_unbiased),
                    forecast_start_utc=str(start), start_hour_utc=int(start_hour),
                    lead_first=int(leads[0]), lead_last=int(leads[-1]),
                    replayed_run_utc=str(self.run_init[k]), n_candidate_runs=int(len(cand)),
                    season_window_days=int(days))
        return out, info

    # ------------------------------------------------------------------
    def season_weights(self, dayofyear):
        """Weights of the 12 monthly bins for a day of year: linear
        interpolation between the two nearest bin centres (circular)."""
        c = self.season_doy
        d = (float(dayofyear) - c + 182.5) % 365.0 - 182.5      # signed distance
        after = np.flatnonzero(d >= 0)
        i = after[np.argmin(d[after])]                              # nearest centre before
        j = (i + 1) % len(c)
        gap = (c[j] - c[i]) % 365.0
        w = np.zeros(len(c))
        w[j] = d[i] / gap
        w[i] = 1.0 - w[j]
        return w

    def rain_params(self, dayofyear=None, seasonal=True):
        """(miss_rate, far, hit_mean_log, hit_sd_log) per lead: of the time of
        year if seasonal and the calibration has seasonal values, else the
        year-round values."""
        if not seasonal or self.season_doy is None or dayofyear is None:
            return self.miss_rate, self.far, self.hit_mean_log, self.hit_sd_log
        w = self.season_weights(dayofyear)
        return tuple(w @ a for a in (self.miss_rate_seasonal, self.far_seasonal,
                                     self.hit_mean_log_seasonal, self.hit_sd_log_seasonal))

    def false_alarm_prob(self, lead, wet_fraction, miss_rate=None, far=None,
                         frequency_unbiased=False):
        """P(false alarm | dry hour) from the measured false-alarm ratio and
        miss rate at this lead time and the wet-hour share p of the training
        fold: FAR / (1 - FAR) * POD * p / (1 - p) (reproduces the FAR)."""
        p = float(wet_fraction)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"wet_fraction must be in [0, 1], got {wet_fraction}")
        if p in (0.0, 1.0):
            return 0.0
        miss = (self.miss_rate if miss_rate is None else miss_rate)[lead]
        if frequency_unbiased:
            # false alarms = misses: forecast wet as often as observed
            return min(miss * p / (1.0 - p), 1.0)
        far = (self.far if far is None else far)[lead]
        return min(far / (1.0 - far) * (1.0 - miss) * p / (1.0 - p), 1.0)

    def _precipitation(self, precip, leads, wet_fraction, u_miss, u_fa, z_hit,
                       z_fa_amount, s, fa_col, remove_bias=False, params=None,
                       frequency_unbiased=False, caps=None):
        miss_rate, far, hit_mean_log, hit_sd_log = params or self.rain_params(seasonal=False)
        vals = precip.to_numpy(dtype=float)
        out = np.zeros_like(vals)
        fa_idx = list(precip.columns).index(fa_col)
        for i, L in enumerate(leads):
            if np.all(vals[i] == 0):
                if u_fa[i] < self.false_alarm_prob(L, wet_fraction, miss_rate, far,
                                                   frequency_unbiased):
                    out[i, fa_idx] = np.exp(self.fa_mean_log + self.fa_sd_log * z_fa_amount[i])
            elif u_miss[i] >= miss_rate[L]:
                sd = s * hit_sd_log[L]
                if remove_bias:   # mean-preserving: E[exp(log_err)] = 1
                    log_err = sd * z_hit[i] - 0.5 * sd * sd
                else:
                    log_err = s * hit_mean_log[L] + sd * z_hit[i]
                out[i] = vals[i] * np.exp(log_err)
            # else: missed event, forecast stays 0
        # cap: training-fold maximum of the column, never below the hour's
        # measured amount (the lognormal amount error has no upper limit)
        for j, c in enumerate(precip.columns):
            if caps and caps.get(c) is not None:
                out[:, j] = np.minimum(out[:, j], np.maximum(caps[c], vals[:, j]))
        return pd.DataFrame(out, index=precip.index, columns=precip.columns)

    def _visibility(self, x, z, s, vis_max, remove_bias=False):
        sd = s * self.vis_below_sd_log
        if remove_bias:   # mean-preserving: E[exp(log_err)] = 1
            y = x * np.exp(sd * z - 0.5 * sd * sd)
        else:
            y = x * np.exp(s * self.vis_below_mean_log + sd * z)
        C = self.vis_cap
        if C is not None:
            at = x >= C - 1e-6
            # at the cap: below it with the measured probability; the depth
            # below the cap, -log(forecast / C), is lognormal and follows the
            # same latent variable (low u = deep), so it persists in time
            u = ndtr(z)
            below = at & (u < self.vis_at_cap_p_below)
            q = ndtri(np.clip(u / max(self.vis_at_cap_p_below, 1e-12), 1e-12, 1 - 1e-12))
            depth = np.exp(self.vis_at_cap_depth_mean_log - self.vis_at_cap_depth_sd_log * q)
            y_at = C * np.exp(-s * depth)
            y = np.where(at, np.where(below, y_at, C), y)
            y = np.minimum(y, C)
        elif vis_max is not None:
            y = np.minimum(y, vis_max)
        return np.maximum(y, 0.0)


_CACHE = {}


def load_error_model(path):
    """Load (and cache) the calibration file of a city."""
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    key = str(path)
    if key not in _CACHE:
        if not path.exists():
            raise FileNotFoundError(
                f"NWP calibration file not found: {path}. Build it with "
                f"weather/nwp/build_nwp_calibration.py")
        _CACHE[key] = NWPErrorModel(path)
    return _CACHE[key]
