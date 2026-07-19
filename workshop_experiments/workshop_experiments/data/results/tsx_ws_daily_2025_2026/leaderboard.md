# tsx_ws_daily_2025_2026 leaderboard

| horizon | target | model | uses_covariates | n_covariates | covariates | predictor_id | mean_crps | n_scores | n_predictions | skipped_origins | dir_precision_up | dir_recall_up | dir_f1_up | dir_accuracy | dir_roc_auc_prob_up | dir_n_eval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | tsx_logret_1b | llmp_quantile_grid_tsx_ws[gemini-3.1-flash-lite-preview] | False | 0 | — | llmp_quantile_grid_tsx_ws[gemini-3.1-flash-lite-preview] | 0.00483 | 365 | 365 | 13 | 0.61299 | 0.97309 | 0.75217 | 0.60822 | 0.50767 | 365 |
| 1 | tsx_logret_1b | llmp_quantile_grid_tsx_ws_cov[gemini-3.1-flash-lite-preview] | True | 11 | tsx_vix_level_l1b, tsx_vix_log_ret_1b_l1b, tsx_wti_oil_log_ret_1b_l1b, tsx_gold_log_ret_1b_l1b, tsx_usdcad_log_ret_1b_l1b, tsx_sp500_log_ret_1b_l1b, tsx_us10y_level_l1b, tsx_boc_policy_rate_l1b, tsx_goc10y_level_l1b, tsx_ca_cpi_mom_logdiff_l1b, tsx_ca_unemployment_l1b | llmp_quantile_grid_tsx_ws_cov[gemini-3.1-flash-lite-preview] | 0.00486 | 365 | 365 | 13 | 0.62222 | 0.87892 | 0.72862 | 0.60000 | 0.54402 | 365 |
| 1 | tsx_logret_1b | darts_lightgbm_cov | False | 0 | — | darts_lightgbm_cov | 0.00488 | 365 | 365 | 13 | 0.61565 | 0.81166 | 0.70019 | 0.57534 | 0.48772 | 365 |
| 1 | tsx_logret_1b | darts_lightgbm | False | 0 | — | darts_lightgbm | 0.00491 | 365 | 365 | 13 | 0.59286 | 0.74439 | 0.66004 | 0.53151 | 0.44553 | 365 |
| 1 | tsx_logret_1b | darts_ets | False | 0 | — | darts_ets | 0.00512 | 365 | 365 | 13 | 0.61509 | 0.73094 | 0.66803 | 0.55616 | 0.51402 | 365 |
| 1 | tsx_logret_1b | darts_kalman | False | 0 | — | darts_kalman | 0.00512 | 365 | 365 | 13 | 0.61111 | 0.49327 | 0.54591 | 0.49863 | 0.53313 | 365 |
| 1 | tsx_logret_1b | darts_autoarima | False | 0 | — | darts_autoarima | 0.00520 | 365 | 365 | 13 | 0.62766 | 0.52915 | 0.57421 | 0.52055 | 0.51544 | 365 |
| 1 | tsx_logret_1b | last_value_naive | False | 0 | — | last_value_naive | 0.00952 | 365 | 365 | 13 | 0.59641 | 0.59641 | 0.59641 | 0.50685 | 0.47988 | 365 |
| 5 | tsx_logret_5b | darts_lightgbm | False | 0 | — | darts_lightgbm | 0.01000 | 365 | 365 | 13 | 0.66234 | 0.83607 | 0.73913 | 0.60548 | 0.49248 | 365 |
| 5 | tsx_logret_5b | darts_lightgbm_cov | False | 0 | — | darts_lightgbm_cov | 0.01001 | 365 | 365 | 13 | 0.65443 | 0.87705 | 0.74956 | 0.60822 | 0.44279 | 365 |
| 5 | tsx_logret_5b | darts_kalman | False | 0 | — | darts_kalman | 0.01039 | 365 | 365 | 13 | 0.66667 | 0.66393 | 0.66530 | 0.55342 | 0.49925 | 365 |
| 5 | tsx_logret_5b | llmp_quantile_grid_tsx_ws_cov[gemini-3.1-flash-lite-preview] | True | 11 | tsx_vix_level_l1b, tsx_vix_log_ret_1b_l1b, tsx_wti_oil_log_ret_1b_l1b, tsx_gold_log_ret_1b_l1b, tsx_usdcad_log_ret_1b_l1b, tsx_sp500_log_ret_1b_l1b, tsx_us10y_level_l1b, tsx_boc_policy_rate_l1b, tsx_goc10y_level_l1b, tsx_ca_cpi_mom_logdiff_l1b, tsx_ca_unemployment_l1b | llmp_quantile_grid_tsx_ws_cov[gemini-3.1-flash-lite-preview] | 0.01041 | 365 | 365 | 13 | 0.66429 | 0.76230 | 0.70992 | 0.58356 | 0.51328 | 365 |
| 5 | tsx_logret_5b | llmp_quantile_grid_tsx_ws[gemini-3.1-flash-lite-preview] | False | 0 | — | llmp_quantile_grid_tsx_ws[gemini-3.1-flash-lite-preview] | 0.01049 | 365 | 365 | 13 | 0.69799 | 0.85246 | 0.76753 | 0.65479 | 0.54786 | 365 |
| 5 | tsx_logret_5b | darts_autoarima | False | 0 | — | darts_autoarima | 0.01085 | 365 | 365 | 13 | 0.64865 | 0.49180 | 0.55944 | 0.48219 | 0.44906 | 365 |
| 5 | tsx_logret_5b | darts_ets | False | 0 | — | darts_ets | 0.01510 | 365 | 365 | 13 | 0.67200 | 0.68852 | 0.68016 | 0.56712 | 0.51477 | 365 |
| 5 | tsx_logret_5b | last_value_naive | False | 0 | — | last_value_naive | 0.01904 | 365 | 365 | 13 | 0.66942 | 0.66393 | 0.66667 | 0.55616 | 0.50139 | 365 |
| 21 | tsx_logret_21b | darts_lightgbm_cov | False | 0 | — | darts_lightgbm_cov | 0.01945 | 364 | 364 | 14 | 0.80263 | 0.85915 | 0.82993 | 0.72527 | 0.49406 | 364 |
| 21 | tsx_logret_21b | darts_lightgbm | False | 0 | — | darts_lightgbm | 0.01957 | 364 | 364 | 14 | 0.78134 | 0.94366 | 0.85486 | 0.75000 | 0.46923 | 364 |
| 21 | tsx_logret_21b | darts_autoarima | False | 0 | — | darts_autoarima | 0.02189 | 364 | 364 | 14 | 0.77576 | 0.45070 | 0.57016 | 0.46978 | 0.47711 | 364 |
| 21 | tsx_logret_21b | darts_kalman | False | 0 | — | darts_kalman | 0.02200 | 364 | 364 | 14 | 0.78967 | 0.75352 | 0.77117 | 0.65110 | 0.54349 | 364 |
| 21 | tsx_logret_21b | llmp_quantile_grid_tsx_ws_cov[gemini-3.1-flash-lite-preview] | True | 11 | tsx_vix_level_l1b, tsx_vix_log_ret_1b_l1b, tsx_wti_oil_log_ret_1b_l1b, tsx_gold_log_ret_1b_l1b, tsx_usdcad_log_ret_1b_l1b, tsx_sp500_log_ret_1b_l1b, tsx_us10y_level_l1b, tsx_boc_policy_rate_l1b, tsx_goc10y_level_l1b, tsx_ca_cpi_mom_logdiff_l1b, tsx_ca_unemployment_l1b | llmp_quantile_grid_tsx_ws_cov[gemini-3.1-flash-lite-preview] | 0.02557 | 364 | 364 | 14 | 0.78099 | 0.66549 | 0.71863 | 0.59341 | 0.52865 | 364 |
| 21 | tsx_logret_21b | llmp_quantile_grid_tsx_ws[gemini-3.1-flash-lite-preview] | False | 0 | — | llmp_quantile_grid_tsx_ws[gemini-3.1-flash-lite-preview] | 0.02557 | 364 | 364 | 14 | 0.77820 | 0.72887 | 0.75273 | 0.62637 | 0.46998 | 364 |
| 21 | tsx_logret_21b | darts_ets | False | 0 | — | darts_ets | 0.02821 | 364 | 364 | 14 | 0.78676 | 0.75352 | 0.76978 | 0.64835 | 0.53438 | 364 |
| 21 | tsx_logret_21b | last_value_naive | False | 0 | — | last_value_naive | 0.03503 | 364 | 364 | 14 | 0.78467 | 0.75704 | 0.77061 | 0.64835 | 0.50977 | 364 |
