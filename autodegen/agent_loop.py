from __future__ import annotations

import argparse
import hashlib
import importlib.util
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from autodegen.config import AppConfig, config_hash, load_config
from autodegen.git_ops import get_changed_files, git_add, git_commit, git_reset_hard
from autodegen.oracle.backtest import BacktestConfig, load_bars, run_backtest
from autodegen.oracle.evaluate import (
    Metrics,
    composite_score,
    compute_data_snapshot_id,
    compute_metrics,
    compute_oracle_hash,
    passes_hard_gates,
    split_data_three_way,
    walk_forward_eval,
    write_result_row,
)
from autodegen.sandbox.runner import create_sandbox_runner
from autodegen.sandbox.strategy import Strategy

logger = logging.getLogger(__name__)


class CommitValidationError(RuntimeError):
    pass


@dataclass(slots=True)
class EvalResult:
    composite_score: float
    walk_forward_metrics: Metrics
    hard_gates_pass: bool
    hard_gates_reason: str
    validation_metrics: Metrics | None
    validation_pass: bool
    strategy_hash: str
    oracle_hash: str
    data_snapshot_id: str
    config_hash: str


class AgentLoop:
    def __init__(self, repo_dir: Path, config: AppConfig, dry_run: bool = False):
        self.repo_dir = repo_dir
        self.config = config
        self.dry_run = dry_run
        self.sandbox_runner = create_sandbox_runner(config)
        self.best_score = 0.0

    def run_forever(self) -> None:
        while True:
            try:
                self.run_one_iteration()
            except KeyboardInterrupt:
                break
            except Exception as exc:  # pragma: no cover
                logger.error("Loop error: %s", exc)

    def run_one_iteration(self) -> None:
        self.config = load_config(self.repo_dir / "config.md")
        context = self.build_context()
        hypothesis, strategy_code = self.propose_experiment(context)

        try:
            self.write_strategy(strategy_code)
            self.validate_and_commit(hypothesis)
            result = self.run_eval()
            kept = self.decide_keep_or_revert(result, hypothesis)
            self.log_result(result, hypothesis, kept=kept)
        except CommitValidationError as exc:
            self.git_reset_if_uncommitted()
            self.log_error(hypothesis, str(exc), status="error")
        except subprocess.TimeoutExpired as exc:
            self.safe_reset()
            self.log_error(hypothesis, str(exc), status="timeout")
        except Exception as exc:  # noqa: BLE001
            logger.error("Experiment failed: %s", exc)
            self.safe_reset()
            self.log_error(hypothesis, str(exc), status="error")

    def build_context(self) -> dict[str, str | list[str]]:
        program = (self.repo_dir / "program.md").read_text() if (self.repo_dir / "program.md").exists() else ""
        strategy = (self.repo_dir / "autodegen" / "sandbox" / "strategy.py").read_text()
        config_md = (self.repo_dir / "config.md").read_text() if (self.repo_dir / "config.md").exists() else ""
        results_path = self.repo_dir / "results.tsv"
        tail: list[str] = []
        if results_path.exists():
            tail = results_path.read_text().splitlines()[-10:]
        return {
            "program_md": program,
            "strategy_py": strategy,
            "config_md": config_md,
            "results_tail": tail,
        }

    def propose_experiment(self, context: dict[str, str | list[str]]) -> tuple[str, str]:
        # TODO: Replace with LLM agent call
        return "baseline test", str(context["strategy_py"])

    def write_strategy(self, code: str) -> None:
        (self.repo_dir / "autodegen" / "sandbox" / "strategy.py").write_text(code)

    def validate_and_commit(self, hypothesis: str) -> None:
        if self.dry_run:
            return
        changed = get_changed_files(self.repo_dir)
        if len(changed) != 1:
            raise CommitValidationError(f"Expected exactly 1 changed file, got {len(changed)}: {changed}")
        if changed[0] != "autodegen/sandbox/strategy.py":
            raise CommitValidationError(
                f"Changed file is {changed[0]}, expected autodegen/sandbox/strategy.py"
            )
        git_add(Path("autodegen/sandbox/strategy.py"), self.repo_dir)
        git_commit(f"hypothesis: {hypothesis} | experiment_id: {uuid4()}", self.repo_dir)

    def _load_strategy_class(self) -> type[Strategy]:
        strategy_path = self.repo_dir / "autodegen" / "sandbox" / "strategy.py"
        spec = importlib.util.spec_from_file_location("agent_strategy_module", strategy_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to load strategy module")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        candidates: list[type[Strategy]] = []
        for value in vars(mod).values():
            if isinstance(value, type) and issubclass(value, Strategy) and value is not Strategy:
                candidates.append(value)
        if not candidates:
            from autodegen.sandbox.strategy import ReferenceStrategy

            return ReferenceStrategy
        return candidates[-1]

    def run_eval(self) -> EvalResult:
        exchange = self.config.exchanges[0] if self.config.exchanges else "hyperliquid"
        pair = self.config.pairs[0] if self.config.pairs else "BTC/USDT"
        bars = load_bars(self.repo_dir / "data", exchange, pair)
        if len(bars) < 16:
            raise ValueError("Not enough bars for evaluation")

        validation_pct = self.config.validation_pct / 100 if self.config.validation_pct > 1 else self.config.validation_pct
        test_pct = self.config.test_pct / 100 if self.config.test_pct > 1 else self.config.test_pct

        wf_data, validation_data, _test_holdout = split_data_three_way(
            bars,
            validation_pct=validation_pct,
            test_pct=test_pct,
        )

        strategy_cls = self._load_strategy_class()
        wf = walk_forward_eval(strategy_cls, wf_data, n_folds=self.config.walk_forward_folds)
        gates_pass, gates_reason = passes_hard_gates(wf)

        strategy_obj = strategy_cls()
        score = composite_score(wf, strategy_obj)

        validation_metrics: Metrics | None = None
        validation_pass = False
        if gates_pass and validation_data:
            bt = run_backtest(strategy_cls(), validation_data, BacktestConfig())
            validation_metrics = compute_metrics(bt)
            validation_pass = (
                validation_metrics.sharpe >= self.config.min_sharpe
                and validation_metrics.max_drawdown <= (self.config.max_drawdown_tolerance / 100.0)
                and validation_metrics.trade_count >= self.config.min_trades
            )

        strategy_hash = hashlib.sha256((self.repo_dir / "autodegen" / "sandbox" / "strategy.py").read_bytes()).hexdigest()
        oracle_hash = compute_oracle_hash(self.repo_dir / "autodegen" / "oracle")
        data_snapshot_id = compute_data_snapshot_id(self.repo_dir / "data")
        cfg_hash = config_hash(self.repo_dir / "config.md")

        fold_metrics = wf.fold_results[0] if wf.fold_results else Metrics(0.0, 0.0, 0.0, 0.0, 0, 0.0)

        return EvalResult(
            composite_score=score,
            walk_forward_metrics=fold_metrics,
            hard_gates_pass=gates_pass,
            hard_gates_reason=gates_reason,
            validation_metrics=validation_metrics,
            validation_pass=validation_pass,
            strategy_hash=strategy_hash,
            oracle_hash=oracle_hash,
            data_snapshot_id=data_snapshot_id,
            config_hash=cfg_hash,
        )

    def decide_keep_or_revert(self, result: EvalResult, hypothesis: str) -> bool:
        keep = (
            result.composite_score > self.best_score
            and result.hard_gates_pass
            and result.validation_pass
        )
        if keep:
            self.best_score = result.composite_score
            self._update_program_best(result.composite_score)
            return True

        if not self.dry_run:
            git_reset_hard(1, self.repo_dir)
        return False

    def _update_program_best(self, score: float) -> None:
        path = self.repo_dir / "program.md"
        if not path.exists():
            return
        lines = path.read_text().splitlines()
        out: list[str] = []
        for line in lines:
            if line.startswith("best_composite:"):
                out.append(f"best_composite: {score:.2f}")
            else:
                out.append(line)
        path.write_text("\n".join(out) + "\n")

    def log_result(self, result: EvalResult, hypothesis: str, kept: bool) -> None:
        write_result_row(
            self.repo_dir / "results.tsv",
            datetime.now(UTC),
            hypothesis,
            result.composite_score,
            result.walk_forward_metrics,
            kept,
            result.strategy_hash,
            result.oracle_hash,
            result.data_snapshot_id,
            result.config_hash,
            status="ok",
            error="",
        )

    def log_error(self, hypothesis: str, error: str, status: str) -> None:
        write_result_row(
            self.repo_dir / "results.tsv",
            datetime.now(UTC),
            hypothesis,
            0.0,
            Metrics(0.0, 0.0, 0.0, 0.0, 0, 0.0),
            False,
            "",
            "",
            "",
            config_hash(self.repo_dir / "config.md"),
            status=status,
            error=error,
        )

    def git_reset_if_uncommitted(self) -> None:
        if self.dry_run:
            return
        changed = get_changed_files(self.repo_dir)
        if changed:
            subprocess.run(["git", "reset", "--hard"], cwd=self.repo_dir, check=True)

    def safe_reset(self) -> None:
        if self.dry_run:
            return
        try:
            git_reset_hard(1, self.repo_dir)
        except Exception:  # noqa: BLE001
            subprocess.run(["git", "reset", "--hard"], cwd=self.repo_dir, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="autodegen agent loop")
    parser.add_argument("--repo-dir", type=Path, default=Path("."))
    parser.add_argument("--once", action="store_true", help="Run one iteration then exit")
    parser.add_argument("--dry-run", action="store_true", help="Don't commit or modify git")
    args = parser.parse_args()

    config = load_config(args.repo_dir / "config.md")
    loop = AgentLoop(args.repo_dir, config, dry_run=args.dry_run)

    if args.once:
        loop.run_one_iteration()
    else:
        loop.run_forever()


if __name__ == "__main__":
    main()
