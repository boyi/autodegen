# autodegen program.md

## Objective
Discover trading strategies that pass walk-forward validation + validation holdout.
Composite score target: > 0.50. Elite: > 0.70.

## Current Best Score
best_composite: 0.00
best_strategy: none

## Research Directives
- Current focus: momentum, mean reversion
- Avoid: HFT, orderbook microstructure, sub-30m bar strategies
- Complexity budget: 8 parameters max (enforced in score)

## Constraints (DO NOT VIOLATE)
- Edit ONLY autodegen/sandbox/strategy.py
- Do not modify autodegen/oracle/ directory
- strategy.parameters must contain ALL tunable values
- No hardcoded numeric constants in method bodies outside self.parameters (AST lint enforces this)
- strategy.name must be unique per experiment
- Minimum bar timeframe: 30m. No tick data, no L2 orderbook features.

## Loop Protocol
1. Read config.md + program.md + strategy.py + last 10 lines results.tsv
2. State hypothesis in one sentence
3. Edit strategy.py
4. Run eval
5. Respond: hypothesis | score | outcome
