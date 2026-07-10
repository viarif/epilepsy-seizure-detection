# Feature Selection Report (Step 6)

Ranking only. `val`/`test` untouched; everything below is computed on `split=='train'`.

## Setup

- Rows used (after negative subsampling): **295,806** (14,086 positive)
- Negatives per positive: `20`
- RF: `400` trees, `class_weight=balanced`, `max_features=sqrt`, `min_samples_leaf=5`
- Permutation importance: `average_precision`, `10` repeats, on a **patient-disjoint** hold-out scored on **real (non-augmented) windows only**
- RF fit rows: 153,598 (real+augmented, 7,190 augmented); hold-out real rows: 140,098 (1,086 seizure)
- Held-out patients: `['chb01', 'chb02', 'chb04', 'chb07', 'chb10', 'chb18']`
- Correlation/redundancy also measured on **real windows only**; augmented copies are used for the RF *fit* only
- Correlation clusters at |Spearman rho| >= `0.9`: **21**

## Why this method, not raw `feature_importances_`

Impurity importance is split across correlated features and several of these 28 are redundant by construction (`hjorth_activity == std**2`; the 6 ratios are functions of the FZ-CZ band powers; `broadband ~= sum(bands)`). So we rank on **permutation importance** (honest, cross-patient) and use **Spearman clustering** to keep only one representative per redundant group.

## Full ranking (by permutation importance)

| rk | feature | perm mean | perm std | gini | cluster | rep |
|---:|---|---:|---:|---:|---:|:--:|
| 1 | `FZ-CZ_std` | 0.22565 | 0.00611 | 0.0449 | 3 | * |
| 2 | `T7-P7_delta_power` | 0.16678 | 0.00474 | 0.0844 | 1 | * |
| 3 | `T7-P7_theta_power` | 0.15341 | 0.00350 | 0.0744 | 9 | * |
| 4 | `T8-P8_beta_power` | 0.15099 | 0.00613 | 0.0357 | 8 | * |
| 5 | `T8-P8_theta_power` | 0.14009 | 0.00397 | 0.0382 | 11 | * |
| 6 | `T8-P8_broadband_power` | 0.12717 | 0.00591 | 0.0341 | 12 | * |
| 7 | `T8-P8_gamma_low_power` | 0.11590 | 0.00553 | 0.0402 | 8 |  |
| 8 | `T8-P8_delta_power` | 0.08569 | 0.00458 | 0.0379 | 2 | * |
| 9 | `T7-P7_beta_power` | 0.06964 | 0.00528 | 0.0496 | 7 | * |
| 10 | `FZ-CZ_peak_to_peak` | 0.06832 | 0.00420 | 0.0414 | 3 |  |
| 11 | `FZ-CZ_theta_power` | 0.05171 | 0.00407 | 0.0171 | 5 | * |
| 12 | `FZ-CZ_hjorth_mobility` | 0.03847 | 0.00472 | 0.0404 | 19 | * |
| 13 | `T7-P7_broadband_power` | 0.02964 | 0.00336 | 0.0565 | 10 | * |
| 14 | `FZ-CZ_beta_power` | 0.02680 | 0.00440 | 0.0366 | 6 | * |
| 15 | `FZ-CZ_zero_crossing_rate` | 0.02241 | 0.00310 | 0.0320 | 19 |  |
| 16 | `T7-P7_gamma_low_power` | 0.02221 | 0.00278 | 0.0373 | 7 |  |
| 17 | `FZ-CZ_beta_gamma_ratio` | 0.01800 | 0.00190 | 0.0434 | 14 | * |
| 18 | `FZ-CZ_gamma_low_power` | 0.01373 | 0.00126 | 0.0298 | 13 | * |
| 19 | `FZ-CZ_delta_power` | 0.00306 | 0.00283 | 0.0302 | 4 | * |
| 20 | `FZ-CZ_skewness` | 0.00040 | 0.00066 | 0.0114 | 21 | * |
| 21 | `FZ-CZ_hjorth_activity` | -0.00000 | 0.00000 | 0.0000 | 3 |  |
| 22 | `FZ-CZ_theta_beta_ratio` | -0.00007 | 0.00053 | 0.0131 | 16 | * |
| 23 | `FZ-CZ_broadband_power` | -0.00508 | 0.00163 | 0.0280 | 4 |  |
| 24 | `FZ-CZ_delta_theta_ratio` | -0.00710 | 0.00083 | 0.0187 | 18 | * |
| 25 | `FZ-CZ_slow_fast_ratio` | -0.00815 | 0.00068 | 0.0258 | 17 | * |
| 26 | `FZ-CZ_delta_relative_power` | -0.01025 | 0.00105 | 0.0256 | 18 |  |
| 27 | `FZ-CZ_gamma_relative_power` | -0.01107 | 0.00156 | 0.0378 | 15 | * |
| 28 | `FZ-CZ_hjorth_complexity` | -0.01183 | 0.00134 | 0.0355 | 20 | * |

## Highly correlated pairs (|rho| >= 0.95)

| rho | feature A | feature B |
|---:|---|---|
| +1.000 | `FZ-CZ_std` | `FZ-CZ_hjorth_activity` |
| +0.973 | `FZ-CZ_std` | `FZ-CZ_peak_to_peak` |
| +0.973 | `FZ-CZ_hjorth_activity` | `FZ-CZ_peak_to_peak` |
| +0.953 | `FZ-CZ_delta_power` | `FZ-CZ_broadband_power` |

## De-correlated shortlist (21 representatives)

Redundancy already removed. Importance was scored on **real (non-augmented) held-out windows only**, so the ranking reflects the true seizure distribution, not the training augmentation. Do **not** pad to a fixed count: keep only representatives whose permutation importance is **positive and stable** (mean > std), then let the validation sweep pick the final `k` among those.

| pick | feature | perm | perm std | gini | cluster | stable |
|---:|---|---:|---:|---:|---:|:--:|
| 1 | `FZ-CZ_std` | 0.22565 | 0.00611 | 0.0449 | 3 | yes |
| 2 | `T7-P7_delta_power` | 0.16678 | 0.00474 | 0.0844 | 1 | yes |
| 3 | `T7-P7_theta_power` | 0.15341 | 0.00350 | 0.0744 | 9 | yes |
| 4 | `T8-P8_beta_power` | 0.15099 | 0.00613 | 0.0357 | 8 | yes |
| 5 | `T8-P8_theta_power` | 0.14009 | 0.00397 | 0.0382 | 11 | yes |
| 6 | `T8-P8_broadband_power` | 0.12717 | 0.00591 | 0.0341 | 12 | yes |
| 7 | `T8-P8_delta_power` | 0.08569 | 0.00458 | 0.0379 | 2 | yes |
| 8 | `T7-P7_beta_power` | 0.06964 | 0.00528 | 0.0496 | 7 | yes |
| 9 | `FZ-CZ_theta_power` | 0.05171 | 0.00407 | 0.0171 | 5 | yes |
| 10 | `FZ-CZ_hjorth_mobility` | 0.03847 | 0.00472 | 0.0404 | 19 | yes |
| 11 | `T7-P7_broadband_power` | 0.02964 | 0.00336 | 0.0565 | 10 | yes |
| 12 | `FZ-CZ_beta_power` | 0.02680 | 0.00440 | 0.0366 | 6 | yes |
| 13 | `FZ-CZ_beta_gamma_ratio` | 0.01800 | 0.00190 | 0.0434 | 14 | yes |
| 14 | `FZ-CZ_gamma_low_power` | 0.01373 | 0.00126 | 0.0298 | 13 | yes |
| 15 | `FZ-CZ_delta_power` | 0.00306 | 0.00283 | 0.0302 | 4 | yes |
| 16 | `FZ-CZ_skewness` | 0.00040 | 0.00066 | 0.0114 | 21 |  |
| 17 | `FZ-CZ_theta_beta_ratio` | -0.00007 | 0.00053 | 0.0131 | 16 |  |
| 18 | `FZ-CZ_delta_theta_ratio` | -0.00710 | 0.00083 | 0.0187 | 18 |  |
| 19 | `FZ-CZ_slow_fast_ratio` | -0.00815 | 0.00068 | 0.0258 | 17 |  |
| 20 | `FZ-CZ_gamma_relative_power` | -0.01107 | 0.00156 | 0.0378 | 15 |  |
| 21 | `FZ-CZ_hjorth_complexity` | -0.01183 | 0.00134 | 0.0355 | 20 |  |

## Files

- `feature_ranking.csv` -- per-feature scores + cluster + rep flag
- `correlation_matrix.csv` -- 28x28 Spearman rho
- `clusters.txt` -- cluster membership, reps marked
- `ranking.png`, `dendrogram.png`, `correlation_heatmap.png`