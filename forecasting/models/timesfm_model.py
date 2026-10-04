"""
TimesFM forecasters.
forecasting/models/timesfm_model.py

Starts a single persistent server process (.timesfm_venv) on first call,
reuses it across all folds — model loads once, not once per fold.

TimesFMForecaster           — with weather covariates
TimesFMForecaster_NoWeather — univariate

The server (run_timesfm_server.py) runs TimesFM 2.5 zero-shot. Covariates
(context + horizon) go into TimesFM's in-context linear regression
(forecast_with_covariates, "xreg + timesfm"). No timestamps are used.
"""
import atexit
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from models.base import BaseForecaster

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
_VENV_PYTHON = _REPO_ROOT / ".timesfm_venv" / "bin" / "python"
_SERVER      = _REPO_ROOT / "forecasting" / "run_timesfm_server.py"

# ---------------------------------------------------------------------------
# Persistent server process (one instance for the whole pipeline run)
# ---------------------------------------------------------------------------
_proc = None
_lock = threading.Lock()
TIMESFM_VERSION: Optional[str] = None  # reported by the server at startup
# Server stderr goes to a temporary file (deleted when closed), so that a
# server that fails to start or dies mid-run is reported with its own error
# message instead of an empty reply.
_stderr_file = None


def _server_stderr_tail(n_lines: int = 30) -> str:
    """Last lines the server wrote to stderr ('' if none)."""
    if _stderr_file is None:
        return ""
    try:
        _stderr_file.seek(0)
        text = _stderr_file.read().decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return ""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    return "\nTimesFM server stderr (last lines):\n" + "\n".join(lines[-n_lines:])


def _get_proc():
    global _proc, TIMESFM_VERSION, _stderr_file
    with _lock:
        if _proc is None or _proc.poll() is not None:
            if _stderr_file is not None:
                _stderr_file.close()
            _stderr_file = tempfile.TemporaryFile()
            _proc = subprocess.Popen(
                [str(_VENV_PYTHON), str(_SERVER)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=_stderr_file,
                text=True,
            )
            ready = _proc.stdout.readline().strip()
            if not ready.startswith("READY|"):
                raise RuntimeError(f"TimesFM server failed to start: {ready!r}{_server_stderr_tail()}")
            TIMESFM_VERSION = ready.split("|", 1)[1]
    return _proc


@atexit.register
def _cleanup():
    global _proc
    if _proc and _proc.poll() is None:
        _proc.terminate()


# ---------------------------------------------------------------------------
# Shared inference helper
# ---------------------------------------------------------------------------
def _run_timesfm(
    y_train: np.ndarray,
    horizon: int,
    X_train: Optional[pd.DataFrame],
    X_test:  Optional[pd.DataFrame],
) -> np.ndarray:
    y_train = np.asarray(y_train, dtype=float)
    # The server finds the horizon rows by their NaN target, so the context
    # must not contain NaN/inf.
    if not np.all(np.isfinite(y_train)):
        raise ValueError("y_train contains NaN/inf; TimesFM server would misread the horizon")

    y_col = np.concatenate([y_train, np.full(horizon, np.nan)])
    df    = pd.DataFrame({"y": y_col})

    if X_train is not None and X_test is not None:
        if len(X_train) != len(y_train) or len(X_test) < horizon:
            raise ValueError(
                f"Covariate lengths do not match: X_train={len(X_train)} "
                f"(y_train={len(y_train)}), X_test={len(X_test)} (horizon={horizon})"
            )
        cov_train = X_train.reset_index(drop=True)
        cov_test  = X_test.reset_index(drop=True).iloc[:horizon]
        covariates = pd.concat([cov_train, cov_test], ignore_index=True)
        for col in covariates.columns:
            df[col] = covariates[col].values

    fd_in,  in_path  = tempfile.mkstemp(suffix=".parquet")
    fd_out, out_path = tempfile.mkstemp(suffix=".parquet")
    os.close(fd_in)
    os.close(fd_out)
    try:
        df.to_parquet(in_path, index=False)

        proc = _get_proc()
        proc.stdin.write(f"{in_path}|{out_path}\n")
        proc.stdin.flush()
        response = proc.stdout.readline().strip()

        if response != f"OK|{out_path}":
            # an empty reply means the server died: add its own error output
            tail = "" if response.startswith("ERROR|") else _server_stderr_tail()
            raise RuntimeError(f"TimesFM server error: {response!r}{tail}")

        y_pred = pd.read_parquet(out_path)["y_pred"].values
    finally:
        for p in (in_path, out_path):
            try:
                os.remove(p)
            except OSError:
                pass

    if len(y_pred) != horizon:
        raise RuntimeError(f"TimesFM returned {len(y_pred)} values, expected {horizon}")
    return y_pred


# ---------------------------------------------------------------------------
# With covariates
# ---------------------------------------------------------------------------
class TimesFMForecaster(BaseForecaster):

    needs_datetime = False

    def __init__(self):
        super().__init__("TimesFM", use_covariates=True)
        self._y_train = None
        self._X_train = None

    def fit(self, y_train: np.ndarray, X_train: Optional[pd.DataFrame] = None) -> None:
        self._y_train = y_train
        self._X_train = X_train
        self._is_fitted = True

    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before predict()")
        if self._X_train is None or X_future is None:
            raise ValueError("TimesFMForecaster requires X_train and X_future (covariates)")
        return _run_timesfm(self._y_train, horizon, self._X_train, X_future)

    def reset(self) -> None:
        super().reset()
        self._y_train = None
        self._X_train = None


# ---------------------------------------------------------------------------
# No covariates
# ---------------------------------------------------------------------------
class TimesFMForecaster_NoWeather(BaseForecaster):

    needs_datetime = False

    def __init__(self):
        super().__init__("TimesFM_NoWeather", use_covariates=False)
        self._y_train = None

    def fit(self, y_train: np.ndarray, X_train: Optional[pd.DataFrame] = None) -> None:
        self._y_train = y_train
        self._is_fitted = True

    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before predict()")
        return _run_timesfm(self._y_train, horizon, None, None)

    def reset(self) -> None:
        super().reset()
        self._y_train = None
