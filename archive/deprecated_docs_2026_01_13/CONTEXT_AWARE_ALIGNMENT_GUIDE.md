# Context-aware Alignment Model 구축 가이드

## 개요

**목표**: 의미 유사도는 높지만 원문 경계가 GT와 맞지 않는 문제 해결

**현재 상태**:
- F1: 0.41~0.43 (어절 분할 버그 수정 후 +9%p 개선)
- 문제: 의미 유사 후보를 선택하지만 경계가 GT와 다름
- 예시: "脩屢建言하니"가 문장1에 포함되지만 GT는 문장2로 분리

**해결책**: Boundary 정보를 모델 입력에 포함시켜 "의미 유사 + 경계 일치"를 함께 학습

---

## 학습 메커니즘 상세

### 의미 + 경계를 어떻게 학습하는가?

#### 1. 의미 학습 (Contrastive Learning)

**목표**: 의미적으로 대응되는 (원문, 번역문) 쌍을 가깝게, 대응 안 되는 쌍을 멀게

**방법**:
```python
# 배치 내 샘플 예시 (B=3)
원문_배치 = ["脩屢建言하니", "遂詔韓琦曾公亮", "其後中選者"]
번역_배치 = ["구양수가 건의하니", "한기에게 명하여", "그 후 선발된 자"]

# 1. 임베딩 생성
zs = encoder_src(원문_배치)  # [3, 256]
zt = encoder_tgt(번역_배치)  # [3, 256]

# 2. 유사도 행렬 (모든 조합 계산)
sim_mat = zs @ zt.T  # [3, 3]
# [
#   [원1↔번1,  원1↔번2,  원1↔번3]  ← 원1↔번1만 정답
#   [원2↔번1,  원2↔번2,  원2↔번3]  ← 원2↔번2만 정답
#   [원3↔번1,  원3↔번2,  원3↔번3]  ← 원3↔번3만 정답
# ]

# 3. Contrastive Loss: 대각선은 높게, 나머지는 낮게
target = [0, 1, 2]  # 대각선 인덱스
logits = sim_mat / temperature
contrast_loss = CrossEntropyLoss(logits, target)
```

**자동 Negative 생성**:
- Positive pairs: (원1, 번1), (원2, 번2), (원3, 번3)
- Negative pairs: 배치 내 다른 모든 조합 (자동 생성)
  - (원1, 번2), (원1, 번3), (원2, 번1), ... 총 6개

#### 2. 경계 학습 (Binary Classification)

**목표**: 원문과 번역문의 문장 경계가 일치하는지 판단

**입력에 경계 정보 포함**:
```python
# 예시 텍스트: "脩屢建言하니 遂詔韓琦"
text = "脩屢建言하니 遂詔韓琦"
boundaries = [0, 7]  # 0번 위치(시작)와 7번 위치(공백 후)

# Character embedding
char_ids = [脩, 屢, 建, 言, 하, 니, _, 遂, 詔, 韓, 琦]
char_emb = embedding(char_ids)  # [11, 64]

# Boundary embedding (학습 가능한 binary embedding)
boundary_flags = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  # 1=경계, 0=내부
boundary_emb = embedding(boundary_flags)  # [11, 32]

# 결합 임베딩
combined = concat(char_emb, boundary_emb)  # [11, 96]
# → LSTM → Mean pooling → Projection → [256]
```

**경계 일치 분류**:
```python
# 원문/번역문 임베딩 생성 (boundary 정보 포함)
zs = encoder_src(원문_tokens, 원문_boundaries)  # [B, 256]
zt = encoder_tgt(번역_tokens, 번역_boundaries)  # [B, 256]

# 두 임베딩을 합쳐서 분류기에 입력
combined = concat(zs, zt)  # [B, 512]
boundary_score = classifier(combined)  # [B, 1] → Sigmoid → 0~1

# Binary Cross-Entropy Loss
boundary_loss = BCE(boundary_score, ground_truth)
```

**학습 데이터 예시**:
```python
# Case 1: 경계 일치 (Positive)
{
  "src": "脩屢建言하니",
  "tgt": "구양수가 건의하니",
  "src_boundaries": [0],
  "tgt_boundaries": [0, 5],
  "label": 1,           # 의미 매칭
  "boundary_match": 1   # 경계 일치
}
→ 목표: boundary_score = 0.95 이상

# Case 2: 경계 불일치 (Hard Negative)
{
  "src": "脩屢建言하니 遂詔韓琦",  # 다음 어절 포함 (경계 틀림)
  "tgt": "구양수가 건의하니",
  "src_boundaries": [0, 7],
  "tgt_boundaries": [0, 5],
  "label": 1,           # 의미는 유사
  "boundary_match": 0   # 경계는 틀림
}
→ 목표: boundary_score = 0.1 이하
```

#### 3. Multitask Learning (결합)

```python
# 두 Loss를 가중 합산
total_loss = contrast_loss + boundary_weight * boundary_loss
#            ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#            의미 유사도 학습    경계 일치 학습
#            (0.7)              (0.3)

# 예시 (boundary_weight = 0.3):
contrast_loss = 0.25
boundary_loss = 0.18
total_loss = 0.25 + 0.3 * 0.18 = 0.304
```

**학습 효과**:
1. **의미 학습**: "脩屢建言하니" ↔ "구양수가 건의하니" 높은 유사도
2. **경계 학습**: 경계가 틀린 "脩屢建言하니 遂詔韓琦" ↔ "구양수가 건의하니"는 낮은 점수
3. **결합**: 의미 유사하면서 경계도 일치하는 후보 선호

---

## 추론 시 동작

### 기존 모델 (의미만)
```python
후보1: "脩屢建言하니"
  → similarity = 0.92 ✅ 선택

후보2: "脩屢建言하니 遂詔韓琦"
  → similarity = 0.91

GT: "脩屢建言하니"

문제: 후보2도 의미 유사도 높음 (0.91)
     → 경계 정보 없어서 구분 못함
     → 잘못된 후보 선택 가능
```

### 새 모델 (의미+경계)
```python
후보1: "脩屢建言하니"
  - similarity = 0.92
  - boundary_match = 0.95  ✅
  - combined = 0.7*0.92 + 0.3*0.95 = 0.929

후보2: "脩屢建言하니 遂詔韓琦"
  - similarity = 0.91
  - boundary_match = 0.12  ❌ (경계 틀림)
  - combined = 0.7*0.91 + 0.3*0.12 = 0.673

→ 후보1 선택 (combined score 더 높음)
→ 경계가 GT와 일치하는 후보 우선 선택!
```

---

## 1. 생성된 파일

### 1.1 데이터 생성 스크립트
📁 `scripts/prepare_boundary_aware_data.py`
- 입력: `datasets/alignment/pa/train.jsonl` (144,686개)
- 출력: `datasets/alignment/pa/train_boundary_aware.jsonl`
- 기능:
  - 원문/번역문에서 어절/구절 경계 추출
  - src_boundaries, tgt_boundaries 추가
  - Hard negative 생성 (경계 쉬프트, 부분 매칭)

### 1.2 모델 학습 스크립트
📁 `scripts/train_boundary_aware_alignment.py`
- 모델: `BoundaryAwareDualEncoder`
  - Character embedding + Boundary embedding
  - BiLSTM encoder
  - Contrastive learning + Boundary match classification
- 출력: `models/dual_encoder_boundary_aware_pa.pt`

### 1.3 추론 엔진
📁 `common/boundary_aware_alignment_loader.py`
- `BoundaryAwareAlignmentMatcher` 클래스
- 기존 `AlignmentMatcher`와 호환 인터페이스
- 추가 메서드:
  - `compute_similarity_with_boundary()`: (similarity, boundary_score) 반환
  - `compute_combined_score()`: 가중 결합 점수

### 1.4 통합 파이프라인
📁 `scripts/run_boundary_aware_pipeline.py`
- 데이터 생성 → 학습 자동 실행
- 설정 가능한 하이퍼파라미터

---

## 2. 사용 방법 (Docker)

### 2.1 데이터 생성 (1단계)

```powershell
# 기본 실행 (hard negative 없이)
docker compose run --rm csp python scripts/prepare_boundary_aware_data.py

# Hard negative 추가 (권장: 30%)
docker compose run --rm csp python scripts/prepare_boundary_aware_data.py --add-hard-neg --hard-neg-ratio 0.3
```

**출력 예시**:
```jsonl
{
  "book": "당송팔대가문초구양수1",
  "src": "歐陽文忠公文抄引",
  "tgt": "≪구양문충공문초(歐陽文忠公文抄)≫의 서문",
  "src_boundaries": [0, 7],
  "tgt_boundaries": [0, 2, 12, 29],
  "label": 1,
  "boundary_match": 1
}
```

### 2.2 모델 학습 (2단계)

```powershell
# 기본 학습 (5 epochs, boundary weight 0.3)
docker compose run --rm csp python scripts/train_boundary_aware_alignment.py --train-jsonl datasets/alignment/pa/train_boundary_aware.jsonl --epochs 5 --batch 64 --boundary-weight 0.3

# 빠른 테스트 (max-steps 제한)
docker compose run --rm csp python scripts/train_boundary_aware_alignment.py --epochs 1 --max-steps 1000 --batch 32
```

**학습 시간 예상**:
- GPU: 3~5일 (144k samples, 5 epochs)
- CPU: 2~3주 (권장하지 않음)

### 2.3 통합 실행 (1+2 자동)

```powershell
# 전체 파이프라인 실행
docker compose run --rm csp python scripts/run_boundary_aware_pipeline.py --add-hard-neg --epochs 5 --boundary-weight 0.3

# 데이터만 생성
docker compose run --rm csp python scripts/run_boundary_aware_pipeline.py --skip-train

# 학습만 실행 (데이터 이미 생성됨)
docker compose run --rm csp python scripts/run_boundary_aware_pipeline.py --skip-data
```

---

## 3. PA Processor 통합

### 3.1 기존 코드 (pa/processor.py)

```python
from common.alignment_model_loader import AlignmentMatcher

# 모델 로드
alignment_model = AlignmentMatcher(
    model_path=Path("models/dual_encoder_alignment_pa.pt"),
    device=device
)

# 유사도 계산
similarity = alignment_model.compute_similarity(src_text, tgt_text)
```

### 3.2 새 모델 적용 (수정 필요)

```python
from common.boundary_aware_alignment_loader import BoundaryAwareAlignmentMatcher

# 모델 로드
alignment_model = BoundaryAwareAlignmentMatcher(
    model_path=Path("models/dual_encoder_boundary_aware_pa.pt"),
    device=device
)

# 옵션 1: 기존 인터페이스 (similarity만)
similarity = alignment_model.compute_similarity(src_text, tgt_text)

# 옵션 2: 경계 점수 포함
similarity, boundary_score = alignment_model.compute_similarity_with_boundary(
    src_text, tgt_text
)

# 옵션 3: 결합 점수 (권장)
combined_score = alignment_model.compute_combined_score(
    src_text, tgt_text,
    boundary_weight=0.3  # 조정 가능
)
```

### 3.3 수정 대상 파일

1. **pa/processor.py** (라인 94~100 근처):
   - `_refine_alignments_with_models()` 함수
   - AlignmentMatcher → BoundaryAwareAlignmentMatcher 교체
   - `compute_combined_score()` 사용 권장

2. **common/config.py**:
   - `boundary_weight` 파라미터 추가 (기본값 0.3)

---

## 4. 평가 및 비교

### 4.1 평가 실행

```powershell
# PA strict 평가 (boundary-aware 모델)
docker compose run --rm csp python evaluate_pa_accuracy.py --ground-truth datasets/pa/test_100_from_pd.csv --prediction test_results/pa_strict_pd_test100_boundary_aware.csv --project pa
```

### 4.2 비교 지표

| 모델 | F1 | Precision | Recall | 특징 |
|------|-----|-----------|--------|------|
| **기존 (eojeol_fixed)** | 0.4125 | 0.451 | 0.380 | 의미 유사도만 학습 |
| **Boundary-aware (목표)** | 0.50+ | ? | ? | 의미 + 경계 학습 |

**기대 효과**:
- 경계 일치 케이스 증가 → Recall 개선
- False positive 감소 → Precision 유지/개선
- **목표 F1: 0.50 이상** (+8~10%p)

---

## 5. 하이퍼파라미터 튜닝

### 5.1 Boundary Loss Weight

`boundary_weight` (기본: 0.3)
- 높을수록: 경계 일치를 중요시 (precision 향상, recall 희생 가능)
- 낮을수록: 의미 유사도를 중요시 (recall 향상, precision 희생 가능)

**추천 범위**: 0.2 ~ 0.5

```powershell
# 실험 예시
docker compose run --rm csp python scripts/train_boundary_aware_alignment.py --boundary-weight 0.2  # 의미 중심
docker compose run --rm csp python scripts/train_boundary_aware_alignment.py --boundary-weight 0.5  # 경계 중심
```

### 5.2 Hard Negative Ratio

`hard_neg_ratio` (기본: 0.3)
- 경계 쉬프트 샘플 비율
- 너무 높으면: 학습 불안정
- 너무 낮으면: 경계 학습 부족

**추천 범위**: 0.2 ~ 0.4

### 5.3 Temperature

`temperature` (기본: 0.07)
- Contrastive learning 온도
- 작을수록: hard negative에 민감
- 클수록: smooth한 학습

**추천 범위**: 0.05 ~ 0.10

---

## 6. 트러블슈팅

### 6.1 학습 시간이 너무 오래 걸려요

**해결책**:
1. GPU 사용 확인: Docker GPU 지원 확인
2. Max steps 제한: `--max-steps 5000`
3. 배치 크기 증가: `--batch 128`
4. 데이터 샘플링: train.jsonl에서 일부만 사용

### 6.2 메모리 부족 (OOM)

**해결책**:
1. 배치 크기 감소: `--batch 32`
2. Max length 축소: 512 → 256
3. Gradient accumulation 사용

### 6.3 F1이 오히려 떨어져요

**원인**:
- Boundary weight가 너무 높음
- Hard negative가 너무 많음
- 학습 데이터 품질 문제

**해결책**:
1. Boundary weight 낮추기: 0.3 → 0.2
2. Hard negative 비율 낮추기: 0.3 → 0.1
3. Epochs 줄이기: 5 → 3 (overfitting 방지)

---

## 7. 다음 단계

### 7.1 즉시 실행

```powershell
# 1. 데이터 생성 + 학습 (GPU 필요)
docker compose run --rm csp python scripts/run_boundary_aware_pipeline.py --add-hard-neg --epochs 5 --boundary-weight 0.3
```

### 7.2 PA Processor 통합

1. `pa/processor.py` 수정 (AlignmentMatcher → BoundaryAwareAlignmentMatcher)
2. `common/config.py`에 boundary_weight 추가
3. 재평가: `evaluate_pa_accuracy.py`

### 7.3 추가 개선 (선택)

- [ ] Attention mechanism 추가
- [ ] Transformer 기반 인코더
- [ ] Multi-task learning (sentence boundary + word boundary)
- [ ] Active learning (low confidence 샘플 재학습)

---

## 8. 참고

**관련 파일**:
- 현재 모델: `models/dual_encoder_alignment_pa.pt`
- 학습 스크립트: `scripts/train_alignment_dual_encoder_trainonly.py`
- PA Processor: `pa/processor.py` (라인 94~100)

**GT 데이터**:
- `datasets/pa/test_100_from_pd.csv` (478개 문장)

**최신 결과**:
- `test_results/pa_strict_pd_test100_eojeol_fixed.csv` (F1=0.4125)
