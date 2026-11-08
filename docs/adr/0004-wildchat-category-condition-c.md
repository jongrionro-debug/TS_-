# ADR-0004: 조건 C — 로컬 프롬프트 의미 분류 후 범주별 분위수 모델 (validation 전용)
- 상태: 채택 → Superseded by [ADR 0005](0005-freeinference-agentic-trace.md) (2026-11-08). 코드·결과는 태그 `wildchat-final`
- 날짜: 2026-11-05
- 관련: Issue #19

## 작성 전 질문
- 무엇이 바뀌는가? A/B에 의미 범주라는 새 입력 축과 새 선택 의존성(로컬 임베딩 모델)이 추가된다. PRD의 "의미 특징 미사용" 범위 밖이므로 ADR로 남긴다.

## 맥락
B(길이+이력)가 A를 이겼다(#13 val, #15 test 1회). 프롬프트 의미(코딩·번역 등)에 따라 응답 길이 분포가 다를 수 있다. test는 이미 사용했으므로 새 조건은 test로 평가할 수 없다. 유료 API는 쓰지 않는다.

## 결정
- 분류: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`(fastembed ONNX, CPU)로 현재 user 프롬프트 앞 600자를 임베딩하고, 7개 범주(coding, creative_writing, editing_summarization, translation, knowledge_qa, math_reasoning, chit_chat)의 고정 설명문 프로토타입과 최대 코사인 최근접으로 zero-shot 배정한다. 첫 샤드 고정 표본의 train+validation 전체 행을 분류하며 표본 추출은 없다. 라벨은 저장소 밖 ID→범주 파일만 저장하고 원문은 쓰지 않는다.
- 모델: C는 범주별로 B와 같은 31개 특징의 LightGBM 분위수 모델을 #13 규칙(사용자 분리, train 내부 홀드아웃 선택, 무정렬)으로 학습한다. C_fallback은 train 5,000행 미만 범주에 전역 B를, C_inner_select는 train 내부 홀드아웃에서 전역 B보다 나을 때만 범주 모델을 쓴다. 임계값은 결과 전 고정.
- 평가: validation만. test 분할 파일은 코드에서 열지 않는다.
- 의존성: `fastembed`는 별도 venv(`wildchat/requirements-classify.txt`)에만 둔다. 기존 고정 환경·#15 고정 파일은 바꾸지 않는다.

## 대안
- 소형 생성 LLM zero-shot: CPU에서 9만 행에 수 시간 이상, 출력 파싱 불안정.
- 사람 라벨+지도 분류기: 라벨링 비용. 재현성은 높다.
- 범주를 B의 특징으로 추가: 표본 분할 없이 공유 가능하나 이번 요청(범주별 별도 모델)과 다름. 후속 후보.

## 결과
- 의미 범주의 이득을 val에서 볼 수 있다. 다만 val은 #7·#13에서 반복 사용돼 낙관적이고 독립 test 확인이 없다. 분류 정확도는 검증하지 않았다.
- AGENTS.md 규칙 변화 없음(최종 test 재평가 금지 유지).

## 분류기 선택 결과 (추가)

- train 200행(validation 미포함, seed 42) 수기 라벨 대비 정확도: paraphrase-multilingual-MiniLM-L12-v2 프로토타입 0.32, multilingual-e5-large 0.235("query:" 접두어 미사용).
- Qwen2.5-0.5B-Instruct(GGUF Q4_K_M, llama-cpp CPU)는 행당 3초 이상으로 약 13.8만 행 분류가 비현실적이라 중단했다.
- 미세조정은 하지 않았다. 수기 라벨에 맞춘 프로토타입 튜닝은 평가셋 과적합이 되므로 하지 않았다. MiniLM 프로토타입을 채택하되 정확도가 낮다는 한계를 명시한다.
- 결과: validation 행 가중 all C 49.198 vs B 49.929(CI [-1.52,-0.01]), 사용자 가중 55.452 vs 55.876(CI가 0 포함). 개선은 첫 턴에 집중되고 후속 턴은 차이 없음.

## 후속: 범주 정보를 전역 B의 특징으로 추가

- 재분류 없이 같은 MiniLM·택소노미로 7개 범주 유사도를 다시 계산했다(classify_sims.py). 138,468행 모두 argmax가 저장 라벨과 일치(불일치 0).
- C_feat_label(B + 범주 categorical), C_feat_sim(B + 유사도 7개)은 전역 B의 하이퍼파라미터·분위수별 반복 수를 고정해 재학습했다.
- validation 전체 행 가중: B 49.929, C 49.198, C_feat_label 48.972, C_feat_sim 47.279(C_feat_sim−B CI [-3.56,-1.85]). 사용자 가중·첫 턴·후속 턴 모두 C_feat_sim이 가장 좋고 CI가 0을 넘지 않는다.
- 결론: 범주를 별도 모델로 나누기보다 연속 유사도를 B의 특징으로 넣는 쪽이 낫다. 이미 여러 번 본 validation 기준이므로 test 확인은 별도 결정이 필요하다.

## 후속 2: MiniLM 임베딩 PCA 성분을 전역 B 특징으로

- 이전 실행은 384차원 임베딩을 저장하지 않아 embed_save.py로 같은 모델·프롬프트를 한 번 더 임베딩해 저장소 밖 embeddings.npz(float16)로 보관했다.
- PCA는 train에서만 적합했다(k 선택은 내부 fit 행, validation 예측은 전체 purged train). k∈{8,16,32}는 #13 train 내부 홀드아웃 pinball로 선택했고 두 변형 모두 k=32(탐색 범위 상한)가 선택됐다. 전역 B 하이퍼파라미터·반복 수는 고정했다.
- validation 전체 행 가중: B 49.929, C_feat_sim 47.279, C_emb_PCA(k=32) 46.337, C_feat_sim_PCA(k=32) 46.204. 두 PCA 변형은 모든 턴·가중에서 B와 C_feat_sim보다 낫고 CI가 0을 넘지 않는다. PCA 변형 둘 사이 차이는 작다.
- 한계: 선택된 k가 범위 상한이라 더 큰 k가 나을 수 있다. validation은 이미 여러 번 본 기간이며 test 확인은 별도 결정이 필요하다.
