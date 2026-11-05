# #21 대화 간 이력 재검증 (validation 전용)

test 자료 접근·최종 평가 재실행 없이 기존 표본과 사용자 분리를 유지했다.
네 조건 모두 같은 3개 설정·조기 종료 규칙으로 train 내부에서 선택했다. PCA k∈{8,16,32}도 내부 선택이다.
전체 train에서 선택 반복 수로 재학습했다. validation은 보고 전용이며 분위수를 정렬하지 않았다.

## 주지표 결론

- B_cross_hist − B: -0.072577, 95% CI [-0.256117, 0.095218]. 95% CI가 0을 포함해 개선을 확인하지 못했다.
- C_emb_PCA_cross_hist − C_emb_PCA: -0.051924, 95% CI [-0.163734, 0.069330]. 95% CI가 0을 포함해 개선을 확인하지 못했다.

## 인과적 시각과 사용자 정의

- 사용자 키는 기존 분리와 동일한 top-level hashed_ip. header는 결합하지 않는다(공유 IP·IP 변경 한계).
- UTC assistant timestamp는 응답 완료 시각이다. 현재 응답 시각은 입력으로 쓰지 않는다.
- 각 행의 기준 시각은 같은 대화의 앞선 assistant 완료 시각이다. 요청 시각의 보수적 하한이다.
- user timestamp가 검증된 제출 시각이 아니므로 사용하지 않는다. 첫 턴·시각 비정상 행은 기준 시각/이력이 결측이다.
- 같은 split에서 기준 시각보다 엄격히 먼저 완료된 다른 원본 대화 그룹만 쓴다. 진행 중 대화·같은 시각·현재 중복 그룹은 제외한다.
- 중복 그룹은 사용자별 가장 먼저 완료된 원본 행 하나로 집계한다. 평균/중앙값은 고정 표본의 유효 응답들만 포함한다.
- EWMA α=0.3, 직전 대화 평균/중앙값, EWMA/log1p EWMA, 완료 대화 수, 초 간격, UTC 시간대/요일, 결측 지표를 추가했다.
- header(accept-language/user-agent), country/state(IP 지역), model(기록 모델)은 결측 집계만 감사하고 모델 입력에는 넣지 않는다.
- 이력 사용자 = validation 내 적어도 한 행에 인과적 이력이 있는 사용자. 이 사용자의 첫 턴도 해당 사용자 표에 포함한다.
- validation 이력은 순차 관측을 가정한다. 앞선 validation 정답은 완료 후 이력으로만 사용하며 학습·선택에는 사용하지 않는다.

## 내부 선택

| 조건 | PCA k | 내부 평균 pinball | P10/P50/P90 반복 수 |
|---|---:|---:|---|
| B | 0 | 43.5825 | 403/244/279 |
| C_emb_PCA | 32 | 40.9279 | 331/499/390 |
| B_cross_hist | 0 | 43.4828 | 387/413/326 |
| C_emb_PCA_cross_hist | 32 | 40.7699 | 320/500/465 |

## validation 평균 pinball

| 사용자 구분 | 턴 | 가중 | B | C_emb_PCA | B_cross_hist | C_emb_PCA_cross_hist |
|---|---|---|---:|---:|---:|---:|
| all_users | first_turn | row | 72.1246 | 64.4535 | 72.1172 | 64.3571 |
| all_users | first_turn | user | 62.4703 | 56.1819 | 62.4914 | 56.1044 |
| all_users | followup_turn | row | 36.3316 | 34.7969 | 36.2191 | 34.7722 |
| all_users | followup_turn | user | 42.0518 | 40.3187 | 42.0583 | 40.3160 |
| all_users | all | row | 49.9285 | 46.0628 | 49.8559 | 46.0108 |
| all_users | all | user | 55.8758 | 51.2102 | 55.9401 | 51.2364 |
| users_with_history | first_turn | row | 74.8335 | 65.7261 | 74.7068 | 65.6628 |
| users_with_history | first_turn | user | 61.7931 | 55.9204 | 61.7392 | 55.8886 |
| users_with_history | followup_turn | row | 35.5615 | 34.1391 | 35.3558 | 34.0006 |
| users_with_history | followup_turn | user | 41.2142 | 39.6703 | 40.9315 | 39.3213 |
| users_with_history | all | row | 49.4154 | 45.2819 | 49.2375 | 45.1699 |
| users_with_history | all | user | 49.5557 | 46.2154 | 49.4385 | 46.1401 |
| users_without_history | first_turn | row | 67.0721 | 62.0800 | 67.2871 | 61.9220 |
| users_without_history | first_turn | user | 62.7135 | 56.2758 | 62.7615 | 56.1818 |
| users_without_history | followup_turn | row | 38.4311 | 36.5903 | 38.5728 | 36.8760 |
| users_without_history | followup_turn | user | 42.6514 | 40.7829 | 42.8649 | 41.0281 |
| users_without_history | all | row | 51.1316 | 47.8934 | 51.3058 | 47.9823 |
| users_without_history | all | user | 58.1273 | 52.9896 | 58.2563 | 53.0520 |

## 사용자 bootstrap 95% CI (차이 < 0이면 개선)

| 사용자 구분 | 턴 | 가중 | 차이 | 점추정 | 95% CI |
|---|---|---|---|---:|---|
| all_users | first_turn | row | B_cross_hist-B | -0.0075 | [-0.2265, 0.1886] |
| all_users | first_turn | row | C_emb_PCA_cross_hist-C_emb_PCA | -0.0964 | [-0.3408, 0.1446] |
| all_users | first_turn | user | B_cross_hist-B | 0.0211 | [-0.0684, 0.1167] |
| all_users | first_turn | user | C_emb_PCA_cross_hist-C_emb_PCA | -0.0775 | [-0.2750, 0.1151] |
| all_users | followup_turn | row | B_cross_hist-B | -0.1125 | [-0.3721, 0.1282] |
| all_users | followup_turn | row | C_emb_PCA_cross_hist-C_emb_PCA | -0.0247 | [-0.2004, 0.1218] |
| all_users | followup_turn | user | B_cross_hist-B | 0.0065 | [-0.2641, 0.3026] |
| all_users | followup_turn | user | C_emb_PCA_cross_hist-C_emb_PCA | -0.0027 | [-0.2353, 0.2300] |
| all_users | all | row | B_cross_hist-B | -0.0726 | [-0.2561, 0.0952] |
| all_users | all | row | C_emb_PCA_cross_hist-C_emb_PCA | -0.0519 | [-0.1637, 0.0693] |
| all_users | all | user | B_cross_hist-B | 0.0644 | [-0.0558, 0.1779] |
| all_users | all | user | C_emb_PCA_cross_hist-C_emb_PCA | 0.0262 | [-0.1261, 0.1961] |
| users_with_history | first_turn | row | B_cross_hist-B | -0.1267 | [-0.4383, 0.1467] |
| users_with_history | first_turn | row | C_emb_PCA_cross_hist-C_emb_PCA | -0.0633 | [-0.4191, 0.2667] |
| users_with_history | first_turn | user | B_cross_hist-B | -0.0539 | [-0.1955, 0.0707] |
| users_with_history | first_turn | user | C_emb_PCA_cross_hist-C_emb_PCA | -0.0318 | [-0.2679, 0.2101] |
| users_with_history | followup_turn | row | B_cross_hist-B | -0.2057 | [-0.5557, 0.1231] |
| users_with_history | followup_turn | row | C_emb_PCA_cross_hist-C_emb_PCA | -0.1385 | [-0.3646, 0.0456] |
| users_with_history | followup_turn | user | B_cross_hist-B | -0.2827 | [-0.7518, 0.2011] |
| users_with_history | followup_turn | user | C_emb_PCA_cross_hist-C_emb_PCA | -0.3490 | [-0.8024, 0.0591] |
| users_with_history | all | row | B_cross_hist-B | -0.1778 | [-0.4338, 0.0468] |
| users_with_history | all | row | C_emb_PCA_cross_hist-C_emb_PCA | -0.1120 | [-0.2590, 0.0393] |
| users_with_history | all | user | B_cross_hist-B | -0.1171 | [-0.3402, 0.1059] |
| users_with_history | all | user | C_emb_PCA_cross_hist-C_emb_PCA | -0.0753 | [-0.2841, 0.1350] |
| users_without_history | first_turn | row | B_cross_hist-B | 0.2150 | [0.0701, 0.3609] |
| users_without_history | first_turn | row | C_emb_PCA_cross_hist-C_emb_PCA | -0.1581 | [-0.4380, 0.1084] |
| users_without_history | first_turn | user | B_cross_hist-B | 0.0480 | [-0.0620, 0.1641] |
| users_without_history | first_turn | user | C_emb_PCA_cross_hist-C_emb_PCA | -0.0940 | [-0.3435, 0.1519] |
| users_without_history | followup_turn | row | B_cross_hist-B | 0.1417 | [-0.0761, 0.3685] |
| users_without_history | followup_turn | row | C_emb_PCA_cross_hist-C_emb_PCA | 0.2857 | [0.1038, 0.4683] |
| users_without_history | followup_turn | user | B_cross_hist-B | 0.2135 | [-0.0846, 0.5866] |
| users_without_history | followup_turn | user | C_emb_PCA_cross_hist-C_emb_PCA | 0.2451 | [-0.0115, 0.5104] |
| users_without_history | all | row | B_cross_hist-B | 0.1742 | [0.0390, 0.3191] |
| users_without_history | all | row | C_emb_PCA_cross_hist-C_emb_PCA | 0.0889 | [-0.0639, 0.2531] |
| users_without_history | all | user | B_cross_hist-B | 0.1290 | [0.0006, 0.2664] |
| users_without_history | all | user | C_emb_PCA_cross_hist-C_emb_PCA | 0.0623 | [-0.1272, 0.2727] |

## 전체 지표 (P10/P50/P90 고유 순서 유지)

| 사용자 구분 | 턴 | 가중 | 조건 | 행/사용자 | 평균 pinball | P10 loss | P50 loss | P90 loss | P50 MAE | coverage | 부호 있는 폭 | 교차율 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all_users | first_turn | row | B | 4908/1726 | 72.124630 | 33.300349 | 114.440223 | 68.633318 | 228.880446 | 0.766911 | 654.808344 | 0.000000 |
| all_users | first_turn | row | C_emb_PCA | 4908/1726 | 64.453527 | 32.925397 | 100.133189 | 60.301996 | 200.266378 | 0.733496 | 555.458747 | 0.005501 |
| all_users | first_turn | row | B_cross_hist | 4908/1726 | 72.117168 | 33.243937 | 114.479247 | 68.628319 | 228.958494 | 0.764466 | 655.100245 | 0.000000 |
| all_users | first_turn | row | C_emb_PCA_cross_hist | 4908/1726 | 64.357145 | 32.749903 | 100.688001 | 59.633532 | 201.376002 | 0.734923 | 553.016382 | 0.004075 |
| all_users | first_turn | user | B | 4908/1726 | 62.470312 | 29.039913 | 97.560885 | 60.810140 | 195.121769 | 0.793472 | 601.557581 | 0.000000 |
| all_users | first_turn | user | C_emb_PCA | 4908/1726 | 56.181915 | 27.499310 | 86.926274 | 54.120160 | 173.852548 | 0.771751 | 496.516804 | 0.010612 |
| all_users | first_turn | user | B_cross_hist | 4908/1726 | 62.491425 | 29.005501 | 97.565281 | 60.903493 | 195.130562 | 0.793262 | 602.073440 | 0.000000 |
| all_users | first_turn | user | C_emb_PCA_cross_hist | 4908/1726 | 56.104367 | 27.524659 | 87.050174 | 53.738267 | 174.100349 | 0.771946 | 493.311636 | 0.007348 |
| all_users | followup_turn | row | B | 8012/1093 | 36.331568 | 21.660933 | 53.985127 | 33.348644 | 107.970255 | 0.776460 | 340.598264 | 0.002496 |
| all_users | followup_turn | row | C_emb_PCA | 8012/1093 | 34.796918 | 20.786744 | 52.550548 | 31.053461 | 105.101096 | 0.771218 | 323.276834 | 0.008113 |
| all_users | followup_turn | row | B_cross_hist | 8012/1093 | 36.219104 | 21.778389 | 53.847668 | 33.031254 | 107.695336 | 0.774214 | 332.705518 | 0.002371 |
| all_users | followup_turn | row | C_emb_PCA_cross_hist | 8012/1093 | 34.772228 | 20.755836 | 52.505667 | 31.055182 | 105.011334 | 0.765227 | 321.393364 | 0.009486 |
| all_users | followup_turn | user | B | 8012/1093 | 42.051782 | 24.144210 | 62.445702 | 39.565436 | 124.891403 | 0.767895 | 364.403330 | 0.002068 |
| all_users | followup_turn | user | C_emb_PCA | 8012/1093 | 40.318735 | 23.334746 | 60.449948 | 37.171510 | 120.899896 | 0.763382 | 345.783381 | 0.004950 |
| all_users | followup_turn | user | B_cross_hist | 8012/1093 | 42.058275 | 24.214498 | 62.386871 | 39.573455 | 124.773742 | 0.767139 | 362.352137 | 0.003007 |
| all_users | followup_turn | user | C_emb_PCA_cross_hist | 8012/1093 | 40.316025 | 23.326880 | 60.443318 | 37.177875 | 120.886636 | 0.754772 | 342.848431 | 0.004990 |
| all_users | all | row | B | 12920/1736 | 49.928499 | 26.082470 | 76.950577 | 46.752451 | 153.901154 | 0.772833 | 459.959183 | 0.001548 |
| all_users | all | row | C_emb_PCA | 12920/1736 | 46.062757 | 25.397929 | 70.626059 | 42.164282 | 141.252118 | 0.756889 | 411.477207 | 0.007121 |
| all_users | all | row | B_cross_hist | 12920/1736 | 49.855922 | 26.133877 | 76.880159 | 46.553731 | 153.760319 | 0.770511 | 455.175589 | 0.001471 |
| all_users | all | row | C_emb_PCA_cross_hist | 12920/1736 | 46.010833 | 25.312096 | 70.808987 | 41.911416 | 141.617974 | 0.753715 | 409.381427 | 0.007430 |
| all_users | all | user | B | 12920/1736 | 55.875767 | 28.208048 | 86.206128 | 53.213126 | 172.412255 | 0.780525 | 507.042080 | 0.000879 |
| all_users | all | user | C_emb_PCA | 12920/1736 | 51.210220 | 26.696398 | 78.414870 | 48.519393 | 156.829739 | 0.768735 | 438.313339 | 0.007578 |
| all_users | all | user | B_cross_hist | 12920/1736 | 55.940122 | 28.217193 | 86.278851 | 53.324324 | 172.557702 | 0.780658 | 506.483389 | 0.001341 |
| all_users | all | user | C_emb_PCA_cross_hist | 12920/1736 | 51.236414 | 26.722831 | 78.513312 | 48.473099 | 157.026623 | 0.766617 | 435.340343 | 0.005221 |
| users_with_history | first_turn | row | B | 3195/456 | 74.833539 | 33.354806 | 119.469306 | 71.676503 | 238.938613 | 0.750235 | 674.222401 | 0.000000 |
| users_with_history | first_turn | row | C_emb_PCA | 3195/456 | 65.726084 | 33.909365 | 102.716474 | 60.552413 | 205.432948 | 0.711111 | 577.706181 | 0.002504 |
| users_with_history | first_turn | row | B_cross_hist | 3195/456 | 74.706825 | 33.256822 | 119.380374 | 71.483280 | 238.760748 | 0.747731 | 674.304708 | 0.000000 |
| users_with_history | first_turn | row | C_emb_PCA_cross_hist | 3195/456 | 65.662771 | 33.644093 | 103.502332 | 59.841889 | 207.004665 | 0.712363 | 575.396629 | 0.002504 |
| users_with_history | first_turn | user | B | 3195/456 | 61.793054 | 28.046184 | 97.099949 | 60.233030 | 194.199897 | 0.782471 | 611.723079 | 0.000000 |
| users_with_history | first_turn | user | C_emb_PCA | 3195/456 | 55.920404 | 27.640769 | 87.120100 | 53.000344 | 174.240200 | 0.739874 | 507.068924 | 0.004532 |
| users_with_history | first_turn | user | B_cross_hist | 3195/456 | 61.739202 | 27.983792 | 96.909522 | 60.324291 | 193.819045 | 0.783456 | 612.261156 | 0.000000 |
| users_with_history | first_turn | user | C_emb_PCA_cross_hist | 3195/456 | 55.888643 | 27.649256 | 87.629125 | 52.387548 | 175.258251 | 0.737561 | 504.181177 | 0.004240 |
| users_with_history | followup_turn | row | B | 5862/456 | 35.561532 | 21.330052 | 52.446053 | 32.908491 | 104.892107 | 0.783350 | 342.878723 | 0.002729 |
| users_with_history | followup_turn | row | C_emb_PCA | 5862/456 | 34.139147 | 20.465657 | 51.210371 | 30.741413 | 102.420742 | 0.777039 | 324.617472 | 0.009212 |
| users_with_history | followup_turn | row | B_cross_hist | 5862/456 | 35.355833 | 21.427558 | 52.286290 | 32.353652 | 104.572580 | 0.780280 | 331.704446 | 0.001024 |
| users_with_history | followup_turn | row | C_emb_PCA_cross_hist | 5862/456 | 34.000626 | 20.383952 | 51.079223 | 30.538704 | 102.158446 | 0.771409 | 322.734728 | 0.011088 |
| users_with_history | followup_turn | user | B | 5862/456 | 41.214201 | 23.291299 | 60.245374 | 40.105929 | 120.490749 | 0.779712 | 360.488609 | 0.001593 |
| users_with_history | followup_turn | user | C_emb_PCA | 5862/456 | 39.670277 | 22.345329 | 58.745073 | 37.920430 | 117.490145 | 0.766440 | 342.448669 | 0.005468 |
| users_with_history | followup_turn | user | B_cross_hist | 5862/456 | 40.931457 | 23.101901 | 60.065642 | 39.626828 | 120.131285 | 0.778814 | 351.178921 | 0.000859 |
| users_with_history | followup_turn | user | C_emb_PCA_cross_hist | 5862/456 | 39.321325 | 22.244054 | 58.611242 | 37.108680 | 117.222484 | 0.761033 | 339.481680 | 0.005942 |
| users_with_history | all | row | B | 9057/456 | 49.415354 | 25.571974 | 76.089566 | 46.584520 | 152.179132 | 0.771668 | 459.765446 | 0.001767 |
| users_with_history | all | row | C_emb_PCA | 9057/456 | 45.281939 | 25.208138 | 69.379964 | 41.257715 | 138.759927 | 0.753782 | 413.898517 | 0.006846 |
| users_with_history | all | row | B_cross_hist | 9057/456 | 49.237518 | 25.600518 | 75.954789 | 46.157247 | 151.909578 | 0.768798 | 452.562107 | 0.000662 |
| users_with_history | all | row | C_emb_PCA_cross_hist | 9057/456 | 45.169949 | 25.061676 | 69.572304 | 40.875866 | 139.144608 | 0.750580 | 411.865210 | 0.008060 |
| users_with_history | all | user | B | 9057/456 | 49.555685 | 25.449515 | 75.648150 | 47.569391 | 151.296300 | 0.779752 | 454.094702 | 0.001304 |
| users_with_history | all | user | C_emb_PCA | 9057/456 | 46.215403 | 24.610551 | 70.418645 | 43.617013 | 140.837290 | 0.757682 | 403.843976 | 0.005095 |
| users_with_history | all | user | B_cross_hist | 9057/456 | 49.438547 | 25.336949 | 75.520467 | 47.458224 | 151.040933 | 0.778984 | 449.106330 | 0.000617 |
| users_with_history | all | user | C_emb_PCA_cross_hist | 9057/456 | 46.140144 | 24.576172 | 70.629349 | 43.214911 | 141.258697 | 0.752567 | 401.078113 | 0.006189 |
| users_without_history | first_turn | row | B | 1713/1270 | 67.072112 | 33.198777 | 105.060234 | 62.957324 | 210.120469 | 0.798015 | 618.598236 | 0.000000 |
| users_without_history | first_turn | row | C_emb_PCA | 1713/1270 | 62.080020 | 31.090151 | 95.314978 | 59.834930 | 190.629956 | 0.775248 | 513.963971 | 0.011092 |
| users_without_history | first_turn | row | B_cross_hist | 1713/1270 | 67.287071 | 33.219905 | 105.337916 | 63.303392 | 210.675832 | 0.795680 | 619.281061 | 0.000000 |
| users_without_history | first_turn | row | C_emb_PCA_cross_hist | 1713/1270 | 61.921958 | 31.082106 | 95.438854 | 59.244916 | 190.877707 | 0.776999 | 511.273890 | 0.007005 |
| users_without_history | first_turn | user | B | 1713/1270 | 62.713485 | 29.396716 | 97.726386 | 61.017354 | 195.452772 | 0.797422 | 597.907607 | 0.000000 |
| users_without_history | first_turn | user | C_emb_PCA | 1713/1270 | 56.275811 | 27.448518 | 86.856680 | 54.522236 | 173.713359 | 0.783196 | 492.728012 | 0.012795 |
| users_without_history | first_turn | user | B_cross_hist | 1713/1270 | 62.761514 | 29.372350 | 97.800735 | 61.111458 | 195.601470 | 0.796783 | 598.415488 | 0.000000 |
| users_without_history | first_turn | user | C_emb_PCA_cross_hist | 1713/1270 | 56.181823 | 27.479921 | 86.842299 | 54.223250 | 173.684598 | 0.784292 | 489.408871 | 0.008465 |
| users_without_history | followup_turn | row | B | 2150/637 | 38.431081 | 22.563085 | 58.181430 | 34.548727 | 116.362860 | 0.757674 | 334.380566 | 0.001860 |
| users_without_history | followup_turn | row | C_emb_PCA | 2150/637 | 36.590337 | 21.662191 | 56.204555 | 31.904265 | 112.409110 | 0.755349 | 319.621566 | 0.005116 |
| users_without_history | followup_turn | row | B_cross_hist | 2150/637 | 38.572821 | 22.734933 | 58.104783 | 34.878745 | 116.209567 | 0.757674 | 335.434952 | 0.006047 |
| users_without_history | followup_turn | row | C_emb_PCA_cross_hist | 2150/637 | 36.876011 | 21.769784 | 56.394884 | 32.463365 | 112.789768 | 0.748372 | 317.736119 | 0.005116 |
| users_without_history | followup_turn | user | B | 2150/637 | 42.651370 | 24.754771 | 64.020818 | 39.178521 | 128.041636 | 0.759435 | 367.205705 | 0.002407 |
| users_without_history | followup_turn | user | C_emb_PCA | 2150/637 | 40.782937 | 24.043025 | 61.670393 | 36.635392 | 123.340786 | 0.761193 | 348.170554 | 0.004580 |
| users_without_history | followup_turn | user | B_cross_hist | 2150/637 | 42.864913 | 25.010957 | 64.048536 | 39.535248 | 128.097071 | 0.758781 | 370.350546 | 0.004546 |
| users_without_history | followup_turn | user | C_emb_PCA_cross_hist | 2150/637 | 41.028086 | 24.102028 | 61.754819 | 37.227409 | 123.509639 | 0.750290 | 345.258538 | 0.004309 |
| users_without_history | all | row | B | 3863/1280 | 51.131595 | 27.279352 | 78.969261 | 47.146171 | 157.938522 | 0.775563 | 460.413408 | 0.001035 |
| users_without_history | all | row | C_emb_PCA | 3863/1280 | 47.893424 | 25.842904 | 73.547593 | 44.289776 | 147.095185 | 0.764173 | 405.800324 | 0.007766 |
| users_without_history | all | row | B_cross_hist | 3863/1280 | 51.305803 | 27.384365 | 79.049737 | 47.483306 | 158.099474 | 0.774528 | 461.303030 | 0.003365 |
| users_without_history | all | row | C_emb_PCA_cross_hist | 3863/1280 | 47.982329 | 25.899219 | 73.708454 | 44.339316 | 147.416907 | 0.761067 | 403.558071 | 0.005954 |
| users_without_history | all | user | B | 3863/1280 | 58.127296 | 29.190776 | 89.967407 | 55.223706 | 179.934814 | 0.780800 | 525.904583 | 0.000728 |
| users_without_history | all | user | C_emb_PCA | 3863/1280 | 52.989624 | 27.439481 | 81.263525 | 50.265866 | 162.527049 | 0.772673 | 450.593049 | 0.008462 |
| users_without_history | all | user | B_cross_hist | 3863/1280 | 58.256309 | 29.243280 | 90.111525 | 55.414122 | 180.223051 | 0.781254 | 526.923966 | 0.001599 |
| users_without_history | all | user | C_emb_PCA_cross_hist | 3863/1280 | 53.051960 | 27.487579 | 81.321973 | 50.346328 | 162.643947 | 0.771622 | 447.546263 | 0.004876 |

## 실행·누수 검사

```json
{
  "inner_fit": {
    "partition": "inner_fit",
    "rows": 52674,
    "history_rows": 21626,
    "history_users": 1906,
    "unknown_anchor_rows": 18449,
    "eligible_completed_conversation_groups": 17132,
    "checked_history_links": 170991,
    "minimum_history_gap_seconds": 1.0,
    "violations": 0,
    "checks": {
      "same_partition_only": true,
      "same_user_only": true,
      "strictly_prior_completion": true,
      "current_source_and_group_excluded": true
    }
  },
  "inner_holdout": {
    "partition": "inner_holdout",
    "rows": 10524,
    "history_rows": 3303,
    "history_users": 455,
    "unknown_anchor_rows": 3932,
    "eligible_completed_conversation_groups": 3647,
    "checked_history_links": 12488,
    "minimum_history_gap_seconds": 1.0,
    "violations": 0,
    "checks": {
      "same_partition_only": true,
      "same_user_only": true,
      "strictly_prior_completion": true,
      "current_source_and_group_excluded": true
    }
  },
  "train": {
    "partition": "train",
    "rows": 81307,
    "history_rows": 36054,
    "history_users": 2782,
    "unknown_anchor_rows": 27887,
    "eligible_completed_conversation_groups": 25879,
    "checked_history_links": 361714,
    "minimum_history_gap_seconds": 1.0,
    "violations": 0,
    "checks": {
      "same_partition_only": true,
      "same_user_only": true,
      "strictly_prior_completion": true,
      "current_source_and_group_excluded": true
    }
  },
  "validation": {
    "partition": "validation",
    "rows": 12920,
    "history_rows": 4636,
    "history_users": 456,
    "unknown_anchor_rows": 4908,
    "eligible_completed_conversation_groups": 4639,
    "checked_history_links": 77348,
    "minimum_history_gap_seconds": 2.0,
    "violations": 0,
    "checks": {
      "same_partition_only": true,
      "same_user_only": true,
      "strictly_prior_completion": true,
      "current_source_and_group_excluded": true
    }
  }
}
```

```sh
/workspace/venvs/wildchat/bin/python wildchat/cross_history.py --prepared /workspace/data/wildchat-1m/derived/issue5-shard0 --source /workspace/data/wildchat-1m/data/train-00000-of-00014.parquet --embeddings /workspace/data/wildchat-1m/derived/issue19-categories/embeddings.npz --output wildchat/results --seed 42 --inner-fraction 0.8 --max-rounds 500 --early-stopping-rounds 40 --threads 4 --bootstrap-replicates 1000
```

- JSON에 메타데이터 감사, split 교집합, 모든 후보·선택 설정, 입력/코드 해시, 환경을 기록한다.
- 기존 46.3373은 B의 고정 설정으로 PCA만 선택한 #19 결과다. 이번 네 조건은 동일한 내부 재튜닝 절차이므로 기준 점수가 달라질 수 있다.
- 첫 샤드 고정 표본이며 반복 사용된 validation이다. CI는 고정 모델에 조건부이며 학습·선택 불확실성과 IP 식별 오차를 반영하지 않는다.
- 첫 턴의 과거 대화 활용 효과는 제출시각 부재로 검증할 수 없다. 실제 API 사용량·GPU 효과로 일반화하지 않는다.
