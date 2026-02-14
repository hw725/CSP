# Models Directory Guide

> Trained PyTorch model weights (`.pt` files) for the CSP pipeline. Active models include `boundary_multitask.pt` (P2S boundary detection), `s2p_phrase_alignment.pt` (S2P v2.1 phrase alignment), and `s2p_crossattn_boundary.pt` (S2P cross-attention boundary). Legacy models with PA/SA naming convention are preserved for backward compatibility.

> **명칭 정책**: 새 모델 파일은 **P2S/S2P** 접두어를 사용합니다.
> 과거 실험 산출물은 **PA/SA** 접두어로 남아 있을 수 있습니다.

---

## ✅ 파이프라인에서 직접 사용되는 모델

| 모델 파일 | 용도 | 비고 |
|-----------|------|------|
| `boundary_multitask.pt` | P2S 경계/정렬 멀티태스크 | 기본 P2S 경계 모델 |
| `s2p_crossattn_boundary.pt` | S2P Cross-Attn 경계 | 기본 S2P 경계 모델 |
| `dual_encoder_alignment_s2p.pt` | S2P 정렬 모델 | **신규 명칭** (legacy: `dual_encoder_alignment_sa.pt`) |

---

## 🧪 레거시/연구용 모델 (PA/SA 명칭)

다음 파일은 연구 실험/이전 명칭 산출물로 유지됩니다.  
필요 시 수동으로 정리하거나 재학습 모델로 대체하세요.

- `dual_encoder_alignment_sa.pt` (legacy S2P 정렬 모델)
- `boundary_multitask_pa_hardneg.pt`
- `boundary_pa_single_hardneg.pt`
- `pa_parent_classifier*.pt`
- `pa_child_classifier.pt`
- `pa_marker_predictor_*.pt`

---

## 🔁 호환성 규칙

- S2P 정렬 로더는 **`dual_encoder_alignment_s2p.pt` → `dual_encoder_alignment_sa.pt`** 순서로 탐색합니다.
- S2P 경계 태거/의미 경계 로더는 **`s2p_*` → `sa_*`** 순서로 탐색합니다.
