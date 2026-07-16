# sp500_ws_smoke leaderboard

| horizon | target | model | uses_covariates | n_covariates | covariates | predictor_id | mean_crps | n_scores | n_predictions | skipped_origins | dir_precision_up | dir_recall_up | dir_f1_up | dir_accuracy | dir_roc_auc_prob_up | dir_n_eval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | sp500_logret_1b | darts_autoarima | False | 0 | — | darts_autoarima | 0.00318 | 3 | 3 | 0 | 0.50000 | 1.00000 | 0.66667 | 0.66667 | 1.00000 | 3 |
| 1 | sp500_logret_1b | darts_ets | False | 0 | — | darts_ets | 0.00327 | 3 | 3 | 0 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 3 |
| 1 | sp500_logret_1b | last_value_naive | False | 0 | — | last_value_naive | 0.01170 | 3 | 3 | 0 | 0.33333 | 1.00000 | 0.50000 | 0.33333 | 0.50000 | 3 |
| 5 | sp500_logret_5b | darts_autoarima | False | 0 | — | darts_autoarima | 0.00968 | 3 | 3 | 0 | 0.50000 | 0.50000 | 0.50000 | 0.33333 | 0.00000 | 3 |
| 5 | sp500_logret_5b | darts_ets | False | 0 | — | darts_ets | 0.01394 | 3 | 3 | 0 | 0.50000 | 0.50000 | 0.50000 | 0.33333 | 0.00000 | 3 |
| 5 | sp500_logret_5b | last_value_naive | False | 0 | — | last_value_naive | 0.01931 | 3 | 3 | 0 | 0.50000 | 0.50000 | 0.50000 | 0.33333 | 0.25000 | 3 |
| 21 | sp500_logret_21b | darts_autoarima | False | 0 | — | darts_autoarima | 0.01547 | 3 | 3 | 0 | 0.50000 | 0.50000 | 0.50000 | 0.33333 | 0.50000 | 3 |
| 21 | sp500_logret_21b | darts_ets | False | 0 | — | darts_ets | 0.02270 | 3 | 3 | 0 | 0.66667 | 1.00000 | 0.80000 | 0.66667 | 0.50000 | 3 |
| 21 | sp500_logret_21b | last_value_naive | False | 0 | — | last_value_naive | 0.02684 | 3 | 3 | 0 | 0.66667 | 1.00000 | 0.80000 | 0.66667 | 0.50000 | 3 |
