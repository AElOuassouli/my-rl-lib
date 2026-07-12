# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Off-policy algorithm: n-step tree backup, with metrics collector support (episode reward/steps, TD error, value change).
- Auto-generated API documentation (MkDocs + Material + mkdocstrings).
- Example script demonstrating n-step tree backup + MLflow on Windy Grid World.

### Fixed

- `n_step_tree_backup`: the training loop never advanced its time step, causing an infinite loop for episodes longer than one step.
- `Policy.get_actions_probabilities_given_state`: fixed a state/action key mix-up that raised a spurious mypy type error and could mask a missing-state lookup.
- `compute_n_step_tree_backup_return`: guarded against `None` rewards and fixed action-probability dicts being iterated as `(key, value)` pairs instead of via `.items()`.
- `compute_n_step_tree_backup_return`: the backward recursion's loop index `k` was never decremented, so every iteration reused the same step instead of walking backward from `min(t, T-1)` to `tau+1`. This produced incorrect returns for any `n > 1`. Also removed a leftover debug `print` that flooded stdout on multi-step returns.

### Changed

- `n_step_tree_backup`: `behavior_policy` is now updated in-place from the same learned action-values as the target policy at every `tau` step. Previously it stayed frozen at its initial values for the whole run, so a fixed, uninformed policy kept exploring the environment even late in training — on Windy Grid World this could make every episode take tens of thousands of steps. Correctness is unaffected: the tree-backup return only ever reads from the target policy.
- `n_step_tree_backup`: now tracks per-episode state visits and forwards `ContextKey.STATE_VISITS`/`VALUE_FUNCTION`/`ENVIRONMENT` to the metrics collector (matching `q_learning`), so state/value heatmaps, the greedy-agent animation, and the registered-model artifact are actually produced instead of being silently skipped.

### Removed

- Unused `WebDashboardVisualizer` hook from the n-step SARSA training loops (on-policy and off-policy); it was never implemented and is superseded by the MLflow metrics backend.

## [0.1.0] - 2026-07-02

### Added

- On-policy algorithms: SARSA, Expected SARSA, n-step SARSA.
- Off-policy algorithms: Q-Learning, Double Q-Learning, n-step off-policy SARSA (with importance sampling).
- Windy Grid World environment (abstract `Environment` base class + concrete implementation).
- Epsilon-greedy and Greedy policies.
- Action-state value function `Q(s, a)` storage with configurable initialisation strategies.
- Modular metrics system with an in-memory backend and an optional MLflow backend for asynchronous metric and artifact logging (scalars, heatmaps, GIF animations).
- Example scripts demonstrating SARSA + MLflow, Q-Learning + MLflow, and metrics tracking.
