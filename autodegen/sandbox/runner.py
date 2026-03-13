from __future__ import annotations

import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from autodegen.config import AppConfig


@dataclass(slots=True)
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    wall_time_seconds: float


class SandboxRunner(ABC):
    @abstractmethod
    def execute_strategy(
        self,
        strategy_path: Path,
        data_path: Path,
        output_path: Path,
        timeout_seconds: int = 900,
    ) -> ExecutionResult:
        raise NotImplementedError


class LocalRunner(SandboxRunner):
    """v0: runs strategy eval in a subprocess (no Docker/systemd yet)."""

    def execute_strategy(
        self,
        strategy_path: Path,
        data_path: Path,
        output_path: Path,
        timeout_seconds: int = 900,
    ) -> ExecutionResult:
        output_path.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        proc = subprocess.run(
            ["python", str(strategy_path)],
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        wall = time.monotonic() - start
        return ExecutionResult(
            success=proc.returncode == 0,
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
            wall_time_seconds=wall,
        )


class DockerRunner(SandboxRunner):
    """Stub for Docker backend."""

    def execute_strategy(
        self,
        strategy_path: Path,
        data_path: Path,
        output_path: Path,
        timeout_seconds: int = 900,
    ) -> ExecutionResult:
        raise NotImplementedError("Docker runner not implemented in v0")


class SystemdRunner(SandboxRunner):
    """Stub for systemd backend."""

    def execute_strategy(
        self,
        strategy_path: Path,
        data_path: Path,
        output_path: Path,
        timeout_seconds: int = 900,
    ) -> ExecutionResult:
        raise NotImplementedError("Systemd runner not implemented in v0")


def create_sandbox_runner(config: AppConfig) -> SandboxRunner:
    if config.sandbox_backend == "local":
        return LocalRunner()
    if config.sandbox_backend == "docker":
        return DockerRunner()
    if config.sandbox_backend == "systemd":
        return SystemdRunner()
    raise ValueError(f"Unknown sandbox backend: {config.sandbox_backend}")
