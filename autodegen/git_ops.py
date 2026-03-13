from __future__ import annotations

import subprocess
from pathlib import Path


def _run_git(args: list[str], repo_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def git_add(filepath: Path, repo_dir: Path | None = None) -> None:
    repo = repo_dir or Path(".")
    _run_git(["add", filepath.as_posix()], repo)


def git_commit(message: str, repo_dir: Path | None = None) -> None:
    repo = repo_dir or Path(".")
    _run_git(["commit", "-m", message], repo)


def git_reset_hard(n: int = 1, repo_dir: Path | None = None) -> None:
    repo = repo_dir or Path(".")
    _run_git(["reset", "--hard", f"HEAD~{n}"], repo)


def get_changed_files(repo_dir: Path | None = None) -> list[str]:
    repo = repo_dir or Path(".")
    out = _run_git(["diff", "--name-only", "HEAD"], repo).stdout.strip()
    return [line.strip() for line in out.splitlines() if line.strip()]


def get_current_commit_hash(repo_dir: Path | None = None) -> str:
    repo = repo_dir or Path(".")
    return _run_git(["rev-parse", "HEAD"], repo).stdout.strip()
