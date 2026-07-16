# tsx_ws_smoke leaderboard

| horizon | target | model | uses_covariates | n_covariates | covariates | predictor_id | mean_crps | n_scores | n_predictions | skipped_origins | dir_precision_up | dir_recall_up | dir_f1_up | dir_accuracy | dir_roc_auc_prob_up | dir_n_eval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | tsx_logret_1b | darts_kalman | False | 0 | — | darts_kalman | 0.00546 | 3 | 3 | 0 | 0.50000 | 1.00000 | 0.66667 | 0.66667 | 0.50000 | 3 |
| 1 | tsx_logret_1b | darts_ets | False | 0 | — | darts_ets | 0.00552 | 3 | 3 | 0 | 0.33333 | 1.00000 | 0.50000 | 0.33333 | 0.50000 | 3 |
| 1 | tsx_logret_1b | darts_autoarima | False | 0 | — | darts_autoarima | 0.00560 | 3 | 3 | 0 | 0.33333 | 1.00000 | 0.50000 | 0.33333 | 1.00000 | 3 |
| 1 | tsx_logret_1b | darts_lightgbm | False | 0 | — | darts_lightgbm | 0.00619 | 3 | 3 | 0 | 0.33333 | 1.00000 | 0.50000 | 0.33333 | 0.50000 | 3 |
| 1 | tsx_logret_1b | darts_lightgbm_cov | False | 0 | — | darts_lightgbm_cov | 0.00630 | 3 | 3 | 0 | 0.33333 | 1.00000 | 0.50000 | 0.33333 | 0.50000 | 3 |
| 1 | tsx_logret_1b | last_value_naive | False | 0 | — | last_value_naive | 0.01146 | 3 | 3 | 0 | 0.50000 | 1.00000 | 0.66667 | 0.66667 | 0.75000 | 3 |
| 5 | tsx_logret_5b | darts_lightgbm_cov | False | 0 | — | darts_lightgbm_cov | 0.00640 | 3 | 3 | 0 | 0.66667 | 1.00000 | 0.80000 | 0.66667 | 1.00000 | 3 |
| 5 | tsx_logret_5b | darts_kalman | False | 0 | — | darts_kalman | 0.00687 | 3 | 3 | 0 | 0.00000 | 0.00000 | 0.00000 | 0.33333 | 0.50000 | 3 |
| 5 | tsx_logret_5b | darts_lightgbm | False | 0 | — | darts_lightgbm | 0.00736 | 3 | 3 | 0 | 0.50000 | 0.50000 | 0.50000 | 0.33333 | 0.50000 | 3 |
| 5 | tsx_logret_5b | darts_autoarima | False | 0 | — | darts_autoarima | 0.00825 | 3 | 3 | 0 | 0.50000 | 0.50000 | 0.50000 | 0.33333 | 0.50000 | 3 |
| 5 | tsx_logret_5b | darts_ets | False | 0 | — | darts_ets | 0.01140 | 3 | 3 | 0 | 1.00000 | 0.50000 | 0.66667 | 0.66667 | 0.50000 | 3 |
| 5 | tsx_logret_5b | last_value_naive | False | 0 | — | last_value_naive | 0.01203 | 3 | 3 | 0 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 3 |
| 21 | tsx_logret_21b | darts_lightgbm_cov | False | 0 | — | darts_lightgbm_cov | 0.01058 | 3 | 3 | 0 | 1.00000 | 1.00000 | 1.00000 | 1.00000 |  | 3 |
| 21 | tsx_logret_21b | darts_lightgbm | False | 0 | — | darts_lightgbm | 0.01280 | 3 | 3 | 0 | 1.00000 | 1.00000 | 1.00000 | 1.00000 |  | 3 |
| 21 | tsx_logret_21b | darts_autoarima | False | 0 | — | darts_autoarima | 0.01992 | 3 | 3 | 0 | 1.00000 | 0.66667 | 0.80000 | 0.66667 |  | 3 |
| 21 | tsx_logret_21b | darts_kalman | False | 0 | — | darts_kalman | 0.02165 | 3 | 3 | 0 | 1.00000 | 0.66667 | 0.80000 | 0.66667 |  | 3 |
| 21 | tsx_logret_21b | darts_ets | False | 0 | — | darts_ets | 0.02304 | 3 | 3 | 0 | 1.00000 | 0.66667 | 0.80000 | 0.66667 |  | 3 |
| 21 | tsx_logret_21b | last_value_naive | False | 0 | — | last_value_naive | 0.02996 | 3 | 3 | 0 | 1.00000 | 0.66667 | 0.80000 | 0.66667 |  | 3 |
