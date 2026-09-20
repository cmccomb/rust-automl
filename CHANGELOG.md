# Changelog

## 0.3.2

- Add versioned JSON `save`/`load` for nine regression families and their fitted
  preprocessing. Loaded models predict and can be saved again without training data.
- Reject unsupported XGBoost persistence before replacing existing artifacts.
- Save atomically and default new Unix artifacts to owner-only permissions.
- Preserve floating-point category values exactly across JSON decoding.
- Persist SVR support vectors, kernels, coefficients, bias, and input width,
  including valid constant models with no support vectors.
- Retain 0.3.1 retraining behavior and add round trips across kernels, KNN
  distance/search settings, preprocessing variants, and f32/f64 models.

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
