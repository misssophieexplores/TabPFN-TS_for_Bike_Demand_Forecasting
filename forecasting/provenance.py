"""
Code provenance for tuning files and results.

Every tuning JSON and every results row records the git commit of the code that
produced it, so each number in the paper can be traced to one code version.
Tuning JSONs, aggregated results rows and the W&B config also record the
versions of the forecasting libraries (incl. TimesFM in .timesfm_venv).

git_dirty = True means tracked files had uncommitted changes when the run
started; the commit then does not fully identify the code. Untracked files
(data, results) are ignored.

Without git (e.g. on the cluster, where the repo is copied without its .git
folder), git_commit holds a code ID instead: "code:<16 hex>", a SHA-256 of all
.py files and weather-error calibration files (.npz) in forecasting/ (without
testing/, __pycache__ and hidden folders; line endings normalised). The same files always give the same ID. To find the
commit of a result, run on the laptop after committing:
    python forecasting/provenance.py
and compare the printed code ID with the result's git_commit.
"""
import hashlib
import json
import os
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


_CODE_DIR = _REPO_ROOT / "forecasting"
_CODE_EXCLUDED_DIRS = {"testing", "__pycache__"}
_CODE_SUFFIXES = {".py", ".npz"}   # code and weather-error calibration data


def code_id() -> str:
    """
    "code:<16 hex>": SHA-256 over the relative paths and contents of all .py
    and .npz files in forecasting/ (sorted; testing/, __pycache__ and hidden
    folders excluded; CRLF -> LF in .py files). Identifies the code without git.
    """
    h = hashlib.sha256()
    files = sorted(
        f for f in _CODE_DIR.rglob("*")
        if f.is_file() and f.suffix in _CODE_SUFFIXES
        and not any(part in _CODE_EXCLUDED_DIRS or part.startswith(".")
                   for part in f.relative_to(_CODE_DIR).parts[:-1])
    )
    for f in files:
        h.update(f.relative_to(_CODE_DIR).as_posix().encode() + b"\0")
        data = f.read_bytes()
        if f.suffix == ".py":
            data = data.replace(b"\r\n", b"\n")
        h.update(data + b"\0")
    return "code:" + h.hexdigest()[:16]


@lru_cache(maxsize=1)
def get_code_version() -> dict:
    """
    {'git_commit': <full commit hash>, 'git_dirty': bool} with git, otherwise
    {'git_commit': code_id(), 'git_dirty': None} (no git, or no .git folder).
    """
    try:
        # The repo must be this project itself, not a parent folder that happens to be a repo
        if Path(_git("rev-parse", "--show-toplevel")).resolve() != _REPO_ROOT:
            raise RuntimeError("not the project repository")
        commit = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    except Exception:
        return {"git_commit": code_id(), "git_dirty": None}
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


def check_output_dir(output_dir) -> Path:
    """
    Create output_dir and check that a file can be written there. The tuning
    scripts call this before any tuning work: with a read-only folder the
    params files were otherwise lost only after hours of tuning.
    """
    out = Path(output_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / f".write_test_{os.getpid()}"
        probe.write_text("ok")
        probe.unlink()
    except OSError as e:
        raise PermissionError(
            f"Cannot write to {out.resolve()} ({e}). The tuning results would be "
            f"lost: pass --output-dir with a writable folder."
        ) from e
    return out


def save_params_json(params: dict, output_file) -> Path:
    """
    Write a tuning result with its provenance. The complete JSON is printed
    to the log first, between BEGIN/END marker lines, so the file can be
    restored exactly from the log if writing it fails.
    """
    output_file = Path(output_file)
    text = json.dumps({**params, "provenance": get_provenance()}, indent=2)
    print(f"----- BEGIN PARAMS JSON {output_file.name} -----\n{text}\n"
          f"----- END PARAMS JSON {output_file.name} -----", flush=True)
    output_file.write_text(text)
    print(f"\nResults saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    # On the laptop: match results from the cluster (git_commit = "code:...") to a commit
    print(f"Code ID: {code_id()}")
    try:
        print(f"Commit:  {_git('rev-parse', 'HEAD')}"
              f"{' (uncommitted changes)' if _git('status', '--porcelain', '--untracked-files=no') else ''}")
    except Exception:
        print("Commit:  no git repository here")
