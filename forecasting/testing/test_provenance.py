"""
Tests for provenance.py: code ID used as git_commit when git cannot read the
project repository (cluster copy without .git).

Run with: pytest forecasting/testing/test_provenance.py -v
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # forecasting/

import pytest

import provenance


def _make_project(root: Path) -> Path:
    code = root / "forecasting"
    (code / "models").mkdir(parents=True)
    (code / "testing").mkdir()
    (code / "__pycache__").mkdir()
    (code / ".cache").mkdir()
    (code / "config.py").write_text("x = 1\n")
    (code / "models" / "m.py").write_text("y = 2\n")
    (code / "testing" / "test_a.py").write_text("t = 0\n")
    (code / "__pycache__" / "c.py").write_text("junk\n")
    (code / ".cache" / "d.py").write_text("junk\n")
    (code / "notes.md").write_text("docs\n")
    return code


@pytest.fixture
def project(tmp_path, monkeypatch):
    code = _make_project(tmp_path)
    monkeypatch.setattr(provenance, "_REPO_ROOT", tmp_path.resolve())
    monkeypatch.setattr(provenance, "_CODE_DIR", code)
    provenance.get_code_version.cache_clear()
    yield code
    provenance.get_code_version.cache_clear()


def test_code_id_stable_and_format(project):
    a, b = provenance.code_id(), provenance.code_id()
    assert a == b and a.startswith("code:") and len(a) == len("code:") + 16


def test_code_id_changes_with_code(project):
    before = provenance.code_id()
    (project / "models" / "m.py").write_text("y = 3\n")
    assert provenance.code_id() != before


def test_code_id_changes_with_new_file_and_calibration(project):
    before = provenance.code_id()
    (project / "new.py").write_text("z = 1\n")
    after_py = provenance.code_id()
    (project / "cal.npz").write_bytes(b"\x00\x01")
    assert len({before, after_py, provenance.code_id()}) == 3


def test_code_id_ignores_tests_cache_docs_and_line_endings(project):
    before = provenance.code_id()
    (project / "testing" / "test_a.py").write_text("t = 1\n")
    (project / "__pycache__" / "c.py").write_text("other\n")
    (project / ".cache" / "d.py").write_text("other\n")
    (project / "notes.md").write_text("changed docs\n")
    (project / "config.py").write_bytes(b"x = 1\r\n")
    assert provenance.code_id() == before


def test_without_git_repository_uses_code_id(project):
    v = provenance.get_code_version()
    assert v == {"git_commit": provenance.code_id(), "git_dirty": None}


def test_with_git_repository_uses_commit(project, tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)
    try:
        git("init", "-q")
        git("add", "-A")
        git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not available")
    v = provenance.get_code_version()
    assert len(v["git_commit"]) == 40 and v["git_dirty"] is False
