"""
TimesFM forecasters.
forecasting/models/timesfm_model.py

Starts a single persistent server process (.timesfm_venv) on first call,
reuses it across all folds — model loads once, not once per fold.

TimesFMForecaster           — with weather covariates
TimesFMForecaster_NoWeather — univariate
"""
import atexit
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


def _get_proc():
    global _proc
    with _lock:
        if _proc is None or _proc.poll() is not None:
            _proc = subprocess.Popen(
                [str(_VENV_PYTHON), str(_SERVER)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            ready = _proc.stdout.readline().strip()
            if ready != "READY":
                raise RuntimeError(f"TimesFM server failed to start: {ready}")
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
    y_col = np.concatenate([y_train, np.full(horizon, np.nan)])
    df    = pd.DataFrame({"y": y_col})

    if X_train is not None and X_test is not None:
        cov_train = X_train.reset_index(drop=True)
        cov_test  = X_test.reset_index(drop=True).iloc[:horizon]
        covariates = pd.concat([cov_train, cov_test], ignore_index=True)
        for col in covariates.columns:
            df[col] = covariates[col].values

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as fin, \
         tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as fout:
        in_path, out_path = fin.name, fout.name

    df.to_parquet(in_path, index=False)

    proc = _get_proc()
    proc.stdin.write(f"{in_path}|{out_path}\n")
    proc.stdin.flush()
    response = proc.stdout.readline().strip()

    if response.startswith("ERROR"):
        raise RuntimeError(f"TimesFM server error: {response}")

    return pd.read_parquet(out_path)["y_pred"].values[:horizon]


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