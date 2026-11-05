# #19 후속: MiniLM 임베딩 PCA 성분을 전역 B 특징으로 (validation 전용)

test 분할 파일은 읽지 않았다. PCA는 train에서만 적합(선택 단계는 내부 fit 행, validation용은 전체 purged train).
k는 #13 train 내부 홀드아웃 pinball로 골랐고 validation은 선택에 쓰지 않았다. 전역 B 하이퍼파라미터·반복 수 고정.

- 선택된 k: C_emb_PCA=32, C_feat_sim_PCA=32. 32성분 설명 분산 합(전체 train) 0.501.

## 내부 홀드아웃 pinball (k 선택)

| 변형 | k=8 | k=16 | k=32 |
|---|---:|---:|---:|
| C_emb_PCA | 41.5859 | 41.2704 | 40.9622 |
| C_feat_sim_PCA | 41.3486 | 41.0678 | 40.9960 |

## validation 평균 pinball (선택된 k)

| 턴 | 가중 | A | B | C_feat_sim | C_emb_PCA | C_feat_sim_PCA |
|---|---|---:|---:|---:|---:|---:|
| first_turn | row | 72.1761 | 72.1246 | 66.5699 | 64.8831 | 64.6747 |
| first_turn | user | 62.8032 | 62.4703 | 58.6292 | 56.6165 | 56.8043 |
| followup_turn | row | 51.5309 | 36.3316 | 35.4610 | 34.9765 | 34.8890 |
| followup_turn | user | 58.0626 | 42.0518 | 40.9417 | 40.4645 | 40.3708 |
| all | row | 59.3735 | 49.9285 | 47.2785 | 46.3373 | 46.2039 |
| all | user | 62.4498 | 55.8758 | 53.0072 | 51.5184 | 51.6103 |

## 보조: 모든 k의 validation (선택에 사용하지 않음)

| 턴 | 가중 | C_emb_PCA_k8 | C_emb_PCA_k16 | C_emb_PCA_k32 | C_feat_sim_PCA_k8 | C_feat_sim_PCA_k16 | C_feat_sim_PCA_k32 |
|---|---|---:|---:|---:|---:|---:|---:|
| first_turn | row | 67.4846 | 65.1044 | 64.8831 | 65.8451 | 64.5372 | 64.6747 |
| first_turn | user | 58.2722 | 56.9048 | 56.6165 | 57.5462 | 56.8858 | 56.8043 |
| followup_turn | row | 35.3210 | 35.0567 | 34.9765 | 35.1247 | 34.9824 | 34.8890 |
| followup_turn | user | 40.6910 | 40.5128 | 40.4645 | 40.6278 | 40.4562 | 40.3708 |
| all | row | 47.5392 | 46.4711 | 46.3373 | 46.7946 | 46.2096 | 46.2039 |
| all | user | 52.7144 | 51.7365 | 51.5184 | 52.1293 | 51.6639 | 51.6103 |

## 사용자 bootstrap 95% CI (음수 = 앞 모델이 좋음)

| 턴 | 가중 | 차이 | 점추정 | 95% CI |
|---|---|---|---:|---|
| first_turn | row | C_emb_PCA-B | -7.2415 | [-9.9057, -4.6526] |
| first_turn | row | C_emb_PCA-C_feat_sim | -1.6868 | [-3.0748, -0.2355] |
| first_turn | row | C_feat_sim_PCA-B | -7.4499 | [-10.0943, -4.7762] |
| first_turn | row | C_feat_sim_PCA-C_feat_sim | -1.8952 | [-3.1810, -0.5789] |
| first_turn | row | C_emb_PCA_k8-B | -4.6401 | [-6.7188, -2.6531] |
| first_turn | row | C_emb_PCA_k8-C_feat_sim | 0.9147 | [-0.3366, 2.1726] |
| first_turn | row | C_emb_PCA_k16-B | -7.0202 | [-9.6861, -4.3924] |
| first_turn | row | C_emb_PCA_k16-C_feat_sim | -1.4654 | [-2.7889, -0.2063] |
| first_turn | row | C_emb_PCA_k32-B | -7.2415 | [-9.9057, -4.6526] |
| first_turn | row | C_emb_PCA_k32-C_feat_sim | -1.6868 | [-3.0748, -0.2355] |
| first_turn | row | C_feat_sim_PCA_k8-B | -6.2796 | [-8.7183, -3.9410] |
| first_turn | row | C_feat_sim_PCA_k8-C_feat_sim | -0.7248 | [-1.8153, 0.3521] |
| first_turn | row | C_feat_sim_PCA_k16-B | -7.5874 | [-10.2860, -4.9706] |
| first_turn | row | C_feat_sim_PCA_k16-C_feat_sim | -2.0327 | [-3.2054, -0.8445] |
| first_turn | row | C_feat_sim_PCA_k32-B | -7.4499 | [-10.0943, -4.7762] |
| first_turn | row | C_feat_sim_PCA_k32-C_feat_sim | -1.8952 | [-3.1810, -0.5789] |
| first_turn | row | C_feat_sim-B | -5.5547 | [-7.5174, -3.6315] |
| first_turn | user | C_emb_PCA-B | -5.8538 | [-6.9890, -4.6960] |
| first_turn | user | C_emb_PCA-C_feat_sim | -2.0126 | [-2.8664, -1.2258] |
| first_turn | user | C_feat_sim_PCA-B | -5.6660 | [-6.7971, -4.4870] |
| first_turn | user | C_feat_sim_PCA-C_feat_sim | -1.8248 | [-2.5907, -1.1085] |
| first_turn | user | C_emb_PCA_k8-B | -4.1981 | [-5.3064, -3.1708] |
| first_turn | user | C_emb_PCA_k8-C_feat_sim | -0.3569 | [-1.2039, 0.4280] |
| first_turn | user | C_emb_PCA_k16-B | -5.5655 | [-6.6500, -4.4787] |
| first_turn | user | C_emb_PCA_k16-C_feat_sim | -1.7243 | [-2.5084, -0.9989] |
| first_turn | user | C_emb_PCA_k32-B | -5.8538 | [-6.9890, -4.6960] |
| first_turn | user | C_emb_PCA_k32-C_feat_sim | -2.0126 | [-2.8664, -1.2258] |
| first_turn | user | C_feat_sim_PCA_k8-B | -4.9241 | [-5.9709, -3.8429] |
| first_turn | user | C_feat_sim_PCA_k8-C_feat_sim | -1.0830 | [-1.6991, -0.5263] |
| first_turn | user | C_feat_sim_PCA_k16-B | -5.5845 | [-6.6223, -4.4359] |
| first_turn | user | C_feat_sim_PCA_k16-C_feat_sim | -1.7434 | [-2.3818, -1.1237] |
| first_turn | user | C_feat_sim_PCA_k32-B | -5.6660 | [-6.7971, -4.4870] |
| first_turn | user | C_feat_sim_PCA_k32-C_feat_sim | -1.8248 | [-2.5907, -1.1085] |
| first_turn | user | C_feat_sim-B | -3.8412 | [-4.7106, -2.9215] |
| followup_turn | row | C_emb_PCA-B | -1.3551 | [-1.8239, -0.9305] |
| followup_turn | row | C_emb_PCA-C_feat_sim | -0.4846 | [-0.7310, -0.2090] |
| followup_turn | row | C_feat_sim_PCA-B | -1.4426 | [-2.0073, -0.9616] |
| followup_turn | row | C_feat_sim_PCA-C_feat_sim | -0.5721 | [-0.7988, -0.3345] |
| followup_turn | row | C_emb_PCA_k8-B | -1.0106 | [-1.4105, -0.6523] |
| followup_turn | row | C_emb_PCA_k8-C_feat_sim | -0.1400 | [-0.3696, 0.1311] |
| followup_turn | row | C_emb_PCA_k16-B | -1.2749 | [-1.6987, -0.8544] |
| followup_turn | row | C_emb_PCA_k16-C_feat_sim | -0.4043 | [-0.6256, -0.1460] |
| followup_turn | row | C_emb_PCA_k32-B | -1.3551 | [-1.8239, -0.9305] |
| followup_turn | row | C_emb_PCA_k32-C_feat_sim | -0.4846 | [-0.7310, -0.2090] |
| followup_turn | row | C_feat_sim_PCA_k8-B | -1.2069 | [-1.6491, -0.7629] |
| followup_turn | row | C_feat_sim_PCA_k8-C_feat_sim | -0.3363 | [-0.5252, -0.1319] |
| followup_turn | row | C_feat_sim_PCA_k16-B | -1.3492 | [-1.8214, -0.9152] |
| followup_turn | row | C_feat_sim_PCA_k16-C_feat_sim | -0.4786 | [-0.6696, -0.2534] |
| followup_turn | row | C_feat_sim_PCA_k32-B | -1.4426 | [-2.0073, -0.9616] |
| followup_turn | row | C_feat_sim_PCA_k32-C_feat_sim | -0.5721 | [-0.7988, -0.3345] |
| followup_turn | row | C_feat_sim-B | -0.8706 | [-1.3845, -0.4797] |
| followup_turn | user | C_emb_PCA-B | -1.5872 | [-2.2690, -0.8434] |
| followup_turn | user | C_emb_PCA-C_feat_sim | -0.4772 | [-0.9432, -0.0119] |
| followup_turn | user | C_feat_sim_PCA-B | -1.6810 | [-2.3917, -0.9007] |
| followup_turn | user | C_feat_sim_PCA-C_feat_sim | -0.5709 | [-0.9876, -0.0993] |
| followup_turn | user | C_emb_PCA_k8-B | -1.3608 | [-1.9242, -0.7651] |
| followup_turn | user | C_emb_PCA_k8-C_feat_sim | -0.2507 | [-0.7015, 0.2011] |
| followup_turn | user | C_emb_PCA_k16-B | -1.5390 | [-2.1529, -0.8385] |
| followup_turn | user | C_emb_PCA_k16-C_feat_sim | -0.4289 | [-0.8603, 0.0093] |
| followup_turn | user | C_emb_PCA_k32-B | -1.5872 | [-2.2690, -0.8434] |
| followup_turn | user | C_emb_PCA_k32-C_feat_sim | -0.4772 | [-0.9432, -0.0119] |
| followup_turn | user | C_feat_sim_PCA_k8-B | -1.4240 | [-2.0454, -0.7163] |
| followup_turn | user | C_feat_sim_PCA_k8-C_feat_sim | -0.3139 | [-0.6657, 0.0935] |
| followup_turn | user | C_feat_sim_PCA_k16-B | -1.5956 | [-2.2579, -0.9034] |
| followup_turn | user | C_feat_sim_PCA_k16-C_feat_sim | -0.4855 | [-0.8629, -0.1043] |
| followup_turn | user | C_feat_sim_PCA_k32-B | -1.6810 | [-2.3917, -0.9007] |
| followup_turn | user | C_feat_sim_PCA_k32-C_feat_sim | -0.5709 | [-0.9876, -0.0993] |
| followup_turn | user | C_feat_sim-B | -1.1101 | [-1.6102, -0.5830] |
| all | row | C_emb_PCA-B | -3.5912 | [-4.7796, -2.5006] |
| all | row | C_emb_PCA-C_feat_sim | -0.9413 | [-1.5431, -0.3644] |
| all | row | C_feat_sim_PCA-B | -3.7246 | [-4.9306, -2.6131] |
| all | row | C_feat_sim_PCA-C_feat_sim | -1.0747 | [-1.6382, -0.5332] |
| all | row | C_emb_PCA_k8-B | -2.3893 | [-3.2693, -1.5736] |
| all | row | C_emb_PCA_k8-C_feat_sim | 0.2606 | [-0.2496, 0.7987] |
| all | row | C_emb_PCA_k16-B | -3.4574 | [-4.6748, -2.4142] |
| all | row | C_emb_PCA_k16-C_feat_sim | -0.8074 | [-1.3713, -0.3079] |
| all | row | C_emb_PCA_k32-B | -3.5912 | [-4.7796, -2.5006] |
| all | row | C_emb_PCA_k32-C_feat_sim | -0.9413 | [-1.5431, -0.3644] |
| all | row | C_feat_sim_PCA_k8-B | -3.1339 | [-4.1690, -2.2073] |
| all | row | C_feat_sim_PCA_k8-C_feat_sim | -0.4839 | [-0.9449, -0.0707] |
| all | row | C_feat_sim_PCA_k16-B | -3.7189 | [-4.9531, -2.6280] |
| all | row | C_feat_sim_PCA_k16-C_feat_sim | -1.0690 | [-1.6053, -0.5997] |
| all | row | C_feat_sim_PCA_k32-B | -3.7246 | [-4.9306, -2.6131] |
| all | row | C_feat_sim_PCA_k32-C_feat_sim | -1.0747 | [-1.6382, -0.5332] |
| all | row | C_feat_sim-B | -2.6500 | [-3.5620, -1.8474] |
| all | user | C_emb_PCA-B | -4.3574 | [-5.2726, -3.5068] |
| all | user | C_emb_PCA-C_feat_sim | -1.4888 | [-2.1046, -0.8927] |
| all | user | C_feat_sim_PCA-B | -4.2654 | [-5.2739, -3.3347] |
| all | user | C_feat_sim_PCA-C_feat_sim | -1.3969 | [-1.9961, -0.8280] |
| all | user | C_emb_PCA_k8-B | -3.1614 | [-4.0158, -2.3824] |
| all | user | C_emb_PCA_k8-C_feat_sim | -0.2928 | [-0.9008, 0.3181] |
| all | user | C_emb_PCA_k16-B | -4.1393 | [-4.9861, -3.3741] |
| all | user | C_emb_PCA_k16-C_feat_sim | -1.2707 | [-1.8940, -0.6822] |
| all | user | C_emb_PCA_k32-B | -4.3574 | [-5.2726, -3.5068] |
| all | user | C_emb_PCA_k32-C_feat_sim | -1.4888 | [-2.1046, -0.8927] |
| all | user | C_feat_sim_PCA_k8-B | -3.7465 | [-4.6230, -2.9103] |
| all | user | C_feat_sim_PCA_k8-C_feat_sim | -0.8779 | [-1.3616, -0.4157] |
| all | user | C_feat_sim_PCA_k16-B | -4.2119 | [-5.1010, -3.3640] |
| all | user | C_feat_sim_PCA_k16-C_feat_sim | -1.3433 | [-1.8734, -0.8853] |
| all | user | C_feat_sim_PCA_k32-B | -4.2654 | [-5.2739, -3.3347] |
| all | user | C_feat_sim_PCA_k32-C_feat_sim | -1.3969 | [-1.9961, -0.8280] |
| all | user | C_feat_sim-B | -2.8686 | [-3.5317, -2.1675] |

- 임베딩(embeddings.npz)·유사도(sims.jsonl)는 저장소 밖 데이터 디렉터리에 있다. validation은 이미 여러 번 본 기간이다.
