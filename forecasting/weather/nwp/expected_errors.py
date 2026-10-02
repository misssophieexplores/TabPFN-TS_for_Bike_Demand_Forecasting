"""
Errors applied by the default measured model (fresh forecast, bias removed,
seasonal rain; rain: year-round values plus 15 January and 15 July)
at 6/24/48/168 h ahead, from the calibration files, plus the measured bias
that is removed. Typical error = mean absolute error after subtracting the
average error of the run's pool (same start hour, start day of year within
+/- 30 days), as the model does. Usage (from forecasting/):
    python weather/nwp/expected_errors.py      -> expected_errors.json
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from weather.nwp_error_model import NWPErrorModel

BASE = str(Path(__file__).resolve().parent / "calibration") + "/"
from config import ForecastConfig  # noqa: E402
UNBIASED = ForecastConfig().nwp_rain_frequency_unbiased
out = {}
for city in ["seoul", "london", "washington"]:
    m = NWPErrorModel(BASE + f"{city}.npz")
    doy, hour = m._doy, m._hour
    res = {}
    for L in [6, 24, 48, 168]:
        r = {}
        for v, name in enumerate(["T", "RH", "WS"]):
            e = m.tqw_errors[:, L, v]
            ok = np.isfinite(e)
            deb = []
            for k in np.flatnonzero(ok):
                d = np.abs(doy - doy[k]); d = np.minimum(d, 366 - d)
                pool = ok & (hour == hour[k]) & (d <= 30)
                deb.append(e[k] - e[pool].mean())
            deb = np.array(deb)
            r[f"{name}_mae_new"] = float(np.abs(deb).mean())
            r[f"{name}_mae_raw"] = float(np.abs(e[ok]).mean())
            r[f"{name}_lean"] = float(e[ok].mean())
            r[f"{name}_n"] = int(ok.sum())
        r["GHI_relmae_new"] = float(np.sqrt(2 / np.pi) * m.solar_sd[L])
        r["GHI_bias_removed"] = float(m.solar_bias[L])
        # applied false-alarm ratio: = miss rate with nwp_rain_frequency_unbiased
        # (config default); the measured ratio is kept as far_measured
        r["miss"] = float(m.miss_rate[L]); r["far_measured"] = float(m.far[L])
        r["far"] = r["miss"] if UNBIASED else r["far_measured"]
        r["pcv"] = float(np.sqrt(np.exp(m.hit_sd_log[L] ** 2) - 1))
        if m.season_doy is not None:      # time of year: 15 Jan and 15 Jul
            for name, k in (("jan", 0), ("jul", 6)):
                r[f"miss_{name}"] = float(m.miss_rate_seasonal[k, L])
                r[f"far_{name}"] = (r[f"miss_{name}"] if UNBIASED
                                    else float(m.far_seasonal[k, L]))
                r[f"pcv_{name}"] = float(np.sqrt(np.exp(m.hit_sd_log_seasonal[k, L] ** 2) - 1))
        r["vcv"] = float(np.sqrt(np.exp(m.vis_below_sd_log ** 2) - 1))
        r["vbelow"] = float(m.vis_at_cap_p_below) if m.vis_cap is not None else None
        res[str(L)] = r
    out[city] = res
json.dump(out, open("expected_errors.json", "w"), indent=1)
for c in out:
    for L in ["6", "24", "48", "168"]:
        r = out[c][L]
        print(c, L, " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
