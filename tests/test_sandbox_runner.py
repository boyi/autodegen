from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from autodegen.config import AppConfig
from autodegen.sandbox.runner import DockerRunner, LocalRunner, SystemdRunner, create_sandbox_runner


def test_local_runner_executes_script_and_captures_output(tmp_path: Path) -> None:
    script = tmp_path / "ok.py"
    script.write_text("print('hello')\n")
    res = LocalRunner().execute_strategy(script, tmp_path, tmp_path)
    assert res.success is True
    assert "hello" in res.stdout
    assert res.exit_code == 0


def test_local_runner_respects_timeout(tmp_path: Path) -> None:
    script = tmp_path / "sleep.py"
    script.write_text("import time\ntime.sleep(2)\n")
    with pytest.raises(subprocess.TimeoutExpired):
        LocalRunner().execute_strategy(script, tmp_path, tmp_path, timeout_seconds=1)


def test_local_runner_returns_nonzero_exit_code(tmp_path: Path) -> None:
    script = tmp_path / "bad.py"
    script.write_text("raise SystemExit(5)\n")
    res = LocalRunner().execute_strategy(script, tmp_path, tmp_path)
    assert res.success is False
    assert res.exit_code == 5


def test_docker_runner_raises_not_implemented(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        DockerRunner().execute_strategy(tmp_path / "s.py", tmp_path, tmp_path)


def test_systemd_runner_raises_not_implemented(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        SystemdRunner().execute_strategy(tmp_path / "s.py", tmp_path, tmp_path)


def test_create_sandbox_runner_local() -> None:
    runner = create_sandbox_runner(AppConfig(sandbox_backend="local"))
    assert isinstance(runner, LocalRunner)


def test_create_sandbox_runner_docker() -> None:
    runner = create_sandbox_runner(AppConfig(sandbox_backend="docker"))
    assert isinstance(runner, DockerRunner)


def test_create_sandbox_runner_systemd() -> None:
    runner = create_sandbox_runner(AppConfig(sandbox_backend="systemd"))
    assert isinstance(runner, SystemdRunner)


def test_create_sandbox_runner_unknown_raises() -> None:
    with pytest.raises(ValueError):
        create_sandbox_runner(AppConfig(sandbox_backend="wat"))
