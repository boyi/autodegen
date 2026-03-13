from __future__ import annotations

import subprocess
from pathlib import Path

from autodegen.git_ops import get_changed_files, get_current_commit_hash, git_add, git_commit, git_reset_hard


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], cwd=repo, check=True, text=True, capture_output=True)
    return out.stdout.strip()


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Tester")
    (repo / "a.txt").write_text("a\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-m", "init")
    return repo


def test_git_add_stages_a_file(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    f = repo / "a.txt"
    f.write_text("changed\n")
    git_add(Path("a.txt"), repo)
    staged = _git(repo, "diff", "--cached", "--name-only")
    assert staged == "a.txt"


def test_git_commit_creates_commit_with_message(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "a.txt").write_text("changed\n")
    git_add(Path("a.txt"), repo)
    git_commit("my message", repo)
    msg = _git(repo, "log", "-1", "--pretty=%s")
    assert msg == "my message"


def test_git_reset_hard_reverts_last_commit(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    old_head = get_current_commit_hash(repo)
    (repo / "a.txt").write_text("changed\n")
    git_add(Path("a.txt"), repo)
    git_commit("next", repo)
    git_reset_hard(1, repo)
    assert get_current_commit_hash(repo) == old_head


def test_get_changed_files_returns_paths(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "a.txt").write_text("changed\n")
    changed = get_changed_files(repo)
    assert changed == ["a.txt"]


def test_get_current_commit_hash_non_empty(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    head = get_current_commit_hash(repo)
    assert len(head) >= 7
