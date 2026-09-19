# Changelog

## 0.3.1

- Replace the supervised-model leaderboard on each successful `train()` call,
  so earlier models cannot override a newly selected algorithm.
- Preserve the last successful models, fitted preprocessing, and score labels
  when retraining fails. Failed initial training no longer exposes partial models.
- Return a parameter error when no algorithms are selected or the supervised
  fold count is outside `2..=number_of_training_rows`.
- Make regression `only(...)` replace previous `only(...)` and `skip(...)` choices.
- Shorten the README around a tested quickstart, move detailed preprocessing
  recipes into the cookbook, and remove obsolete feature and capability claims.
