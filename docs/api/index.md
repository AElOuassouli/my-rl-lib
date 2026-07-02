# API Reference

The library is generic over the environment's state and action types. Two
type variables, `StateT` and `ActionT` (defined in `my_rl_lib.types`), thread
that generality through environments, values, policies, and the learning
algorithms so a concrete environment (e.g. a grid world with
`tuple[int, int]` states) yields fully-typed algorithms.

| Package | Contents |
|---|---|
| [Environments](environments.md) | Abstract environment base class + Windy Grid World |
| [Learning &mdash; On-Policy](learning-on-policy.md) | SARSA, Expected SARSA, n-step SARSA |
| [Learning &mdash; Off-Policy](learning-off-policy.md) | Q-Learning, Double Q-Learning, n-step off-policy SARSA |
| [Policies](policies.md) | Epsilon-greedy, Greedy, importance sampling |
| [Values](values.md) | Q(s, a) storage and initialisation strategies |
| [Metrics](metrics.md) | Collector, handlers, and the MLflow backend |
