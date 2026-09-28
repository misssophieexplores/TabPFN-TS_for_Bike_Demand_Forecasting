"""
Code provenance for tuning files and results.

Every tuning JSON and every results row records the git commit of the code that
produced it, so each number in the paper can be traced to one code version.
Tuning JSONs, aggregated results rows and the W&B config also record the
versions of the forecasting libraries (incl. TimesFM in .timesfm_venv).

git_dirty = True means tracked files had uncommitted changes when the run
started; the commit then does not fully identify the code. Untracked files
(data, results) are ignored.
"""
import platform
import subprocess
from functools import lru_cache
from importlib import metadata
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TIMESFM_PYTHON = _REPO_ROOT / ".timesfm_venv" / "bin" / "python"

# Libraries whose version changes forecasts (main environment)
_LIBRARIES = [
    "numpy", "pandas", "scikit-learn", "statsmodels", "pmdarima", "xgboost",
    "optuna", "prophet", "neuralprophet", "torch", "tabpfn", "tabpfn-time-series",
]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(_REPO_ROOT), *args],
        capture_output=True, text=True, timeout=10, check=True,
    ).stdout.strip()


@lru_cache(maxsize=1)
def get_code_version() -> dict:
    """{'git_commit': <full hash or 'unknown'>, 'git_dirty': bool or None}"""
    try:
        commit = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    except Exception:
        return {"git_commit": "unknown", "git_dirty": None}
    if dirty:
        print(f"WARNING: uncommitted changes in tracked files; results are tagged "
              f"git_dirty=True (commit {commit[:10]}). Commit before paper runs.")
    return {"git_commit": commit, "git_dirty": dirty}


def _timesfm_version():
    """timesfm version in the separate .timesfm_venv (None if not available)."""
    if not _TIMESFM_PYTHON.exists():
        return None
    try:
        return subprocess.run(
            [str(_TIMESFM_PYTHON), "-c",
             "from importlib import metadata; print(metadata.version('timesfm'))"],
            capture_output=True, text=True, timeout=120, check=True,
        ).stdout.strip() or None
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_library_versions() -> dict:
    """{library: version or None if not installed}, plus python and timesfm."""
    versions = {"python": platform.python_version()}
    for lib in _LIBRARIES:
        try:
            versions[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            versions[lib] = None
    versions["timesfm (.timesfm_venv)"] = _timesfm_version()
    return versions


def get_provenance() -> dict:
    """Git commit/dirty flag plus library versions (for tuning JSONs, W&B)."""
    return {**get_code_version(), "library_versions": get_library_versions()}
