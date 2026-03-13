from __future__ import annotations

import subprocess
from pathlib import Path

import polars as pl
import pytest

from autodegen.agent_loop import AgentLoop, CommitValidationError, main
from autodegen.config import AppConfig


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    (r / "autodegen" / "sandbox").mkdir(parents=True)
    (r / "autodegen" / "oracle").mkdir(parents=True)
    (r / "data" / "hyperliquid" / "BTC-USDT").mkdir(parents=True)

    (r / "config.md").write_text(
        """# cfg\n- pairs: BTC/USDT\n- exchanges: hyperliquid\n- walk_forward_folds: 4\n- validation_pct: 15%\n- test_pct: 15%\n- sandbox_backend: local\n- min_trades: 0\n"""
    )
    (r / "program.md").write_text(
        """# autodegen program.md\n\n## Current Best Score\nbest_composite: 0.00\nbest_strategy: none\n"""
    )
    (r / "autodegen" / "oracle" / "stub.py").write_text("x=1\n")
    (r / "autodegen" / "sandbox" / "strategy.py").write_text(
        """from autodegen.sandbox.strategy import Strategy\nfrom autodegen.oracle.backtest import Signal\nclass T(Strategy):\n    name = 't'\n    def __init__(self):\n        super().__init__()\n        self.parameters={'a':1}\n    def on_bar(self, bar, portfolio):\n        return []\n"""
    )

    ts = pl.datetime_range(
        pl.datetime(2025, 1, 1, 0, 0, time_zone="UTC"),
        pl.datetime(2025, 1, 5, 3, 0, time_zone="UTC"),
        interval="1h",
        eager=True,
    )
    df = pl.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0 + i for i in range(len(ts))],
            "high": [101.0 + i for i in range(len(ts))],
            "low": [99.0 + i for i in range(len(ts))],
            "close": [100.5 + i for i in range(len(ts))],
            "volume": [1000.0 for _ in range(len(ts))],
        }
    )
    df.write_parquet(r / "data" / "hyperliquid" / "BTC-USDT" / "2025-01.parquet")

    subprocess.run(["git", "init"], cwd=r, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=r, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=r, check=True)
    subprocess.run(["git", "add", "."], cwd=r, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=r, check=True)
    return r


def _loop(repo: Path, dry_run: bool = False) -> AgentLoop:
    return AgentLoop(
        repo,
        AppConfig(
            sandbox_backend="local",
            min_trades=0,
            walk_forward_folds=4,
            exchanges=["hyperliquid"],
            pairs=["BTC/USDT"],
        ),
        dry_run=dry_run,
    )


def test_build_context_reads_files(repo: Path) -> None:
    (repo / "results.tsv").write_text("a\nb\nc\n")
    ctx = _loop(repo).build_context()
    assert "program_md" in ctx
    assert "strategy_py" in ctx
    assert ctx["results_tail"] == ["a", "b", "c"]


def test_write_strategy_writes_to_correct_path(repo: Path) -> None:
    lp = _loop(repo)
    lp.write_strategy("# new\n")
    assert (repo / "autodegen" / "sandbox" / "strategy.py").read_text() == "# new\n"


def test_validate_and_commit_succeeds_with_only_strategy_changed(repo: Path) -> None:
    lp = _loop(repo)
    (repo / "autodegen" / "sandbox" / "strategy.py").write_text("# changed\n")
    lp.validate_and_commit("h1")
    msg = subprocess.run(["git", "log", "-1", "--pretty=%s"], cwd=repo, check=True, text=True, capture_output=True).stdout.strip()
    assert msg.startswith("hypothesis: h1 | experiment_id:")


def test_validate_and_commit_fails_when_other_files_changed(repo: Path) -> None:
    lp = _loop(repo)
    (repo / "autodegen" / "sandbox" / "strategy.py").write_text("# changed\n")
    (repo / "program.md").write_text("changed\n")
    with pytest.raises(CommitValidationError):
        lp.validate_and_commit("h")


def test_decide_keep_or_revert_keeps_on_improvement(repo: Path) -> None:
    lp = _loop(repo)
    res = lp.run_eval()
    res.hard_gates_pass = True
    res.validation_pass = True
    res.composite_score = 0.9
    kept = lp.decide_keep_or_revert(res, "h")
    assert kept is True
    assert lp.best_score == 0.9


def test_decide_keep_or_revert_reverts_when_not_improved(repo: Path) -> None:
    lp = _loop(repo)
    (repo / "autodegen" / "sandbox" / "strategy.py").write_text(
        "from autodegen.sandbox.strategy import Strategy\n"
        "class T2(Strategy):\n"
        "    name='t2'\n"
        "    def __init__(self):\n"
        "        super().__init__(); self.parameters={'a':1}\n"
        "    def on_bar(self, bar, portfolio):\n"
        "        return []\n"
    )
    lp.validate_and_commit("h")
    res = lp.run_eval()
    res.hard_gates_pass = False
    res.validation_pass = False
    res.composite_score = 0.0
    kept = lp.decide_keep_or_revert(res, "h")
    assert kept is False


def test_run_one_iteration_completes(repo: Path) -> None:
    lp = _loop(repo)
    lp.run_one_iteration()
    assert (repo / "results.tsv").exists()


def test_once_flag_runs_and_exits(monkeypatch, repo: Path) -> None:
    monkeypatch.setattr("sys.argv", ["agent_loop", "--repo-dir", str(repo), "--once", "--dry-run"])
    main()


def test_error_handling_bad_strategy_does_not_crash(repo: Path) -> None:
    lp = _loop(repo)
    (repo / "autodegen" / "sandbox" / "strategy.py").write_text("def broken(:\n")
    lp.run_one_iteration()
    text = (repo / "results.tsv").read_text()
    assert "error" in text


def test_results_tsv_gets_new_row_each_iteration(repo: Path) -> None:
    lp = _loop(repo)
    lp.run_one_iteration()
    n1 = len((repo / "results.tsv").read_text().splitlines())
    lp.run_one_iteration()
    n2 = len((repo / "results.tsv").read_text().splitlines())
    assert n2 > n1


def test_config_hot_reload_between_iterations(repo: Path) -> None:
    lp = _loop(repo)
    lp.run_one_iteration()
    (repo / "config.md").write_text((repo / "config.md").read_text() + "\n- walk_forward_folds: 3\n")
    lp.run_one_iteration()
    assert lp.config.walk_forward_folds == 3
