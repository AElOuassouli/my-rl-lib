# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Auto-generated API documentation (MkDocs + Material + mkdocstrings).

## [0.1.0] - 2026-07-02

### Added

- On-policy algorithms: SARSA, Expected SARSA, n-step SARSA.
- Off-policy algorithms: Q-Learning, Double Q-Learning, n-step off-policy SARSA (with importance sampling).
- Windy Grid World environment (abstract `Environment` base class + concrete implementation).
- Epsilon-greedy and Greedy policies.
- Action-state value function `Q(s, a)` storage with configurable initialisation strategies.
- Modular metrics system with an in-memory backend and an optional MLflow backend for asynchronous metric and artifact logging (scalars, heatmaps, GIF animations).
- Example scripts demonstrating SARSA + MLflow, Q-Learning + MLflow, and metrics tracking.
