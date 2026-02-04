# 클라우드 GPU 테스트 실행 가이드

> RunPod에서 P2S/S2P 추론 + F1 평가를 실행하는 절차

## 사전 준비

### 1. 테스트 데이터 및 모델이 Git에 포함되어 있어야 함

- `datasets/splits/*_test.xlsx` (paragraph, sentence, phrase)
- `models/*.pt` (boundary_multitask, dual_encoder_alignment_p2s/s2p, s2p_crossattn_boundary)
- `.gitignore`에서 예외 처리 필요 (feature/unignore-test-data 브랜치 참조)

### 2. RunPod Pod 생성

| 항목 | 설정값 |
|------|--------|
| Template | RunPod PyTorch 2.4.0 |
| GPU | A100 PCIe (80GB) |
| Instance pricing | On-Demand |
| Container Disk | 40GB |
| Volume Disk | 50GB |
| SSH terminal access | 체크 |

---

## 실행 절차

### 1단계: 환경 셋업

Web Terminal 또는 Jupyter Terminal에서 실행 (Ctrl+Shift+V로 붙여넣기).

```bash
cd /workspace
git clone -b feature/unignore-test-data https://github.com/hw725/CSP.git
cd CSP

# 충돌 패키지 제거 (필수 - torchvision/torchaudio가 torch 버전 충돌 유발)
pip uninstall torchvision torchaudio -y

# 패키지 설치
pip install -r requirements.txt

# transformers 호환 버전으로 교체
# - 4.57.3은 suparkanbun과 비호환 (BertForTokenClassification import 실패)
# - 4.48.0은 suparkanbun + FlagEmbedding 모두 호환
pip install transformers==4.48.0 FlagEmbedding>=1.2.0 --force-reinstall

# Stanza/SuPar 설치 (requirements.txt에서 누락될 수 있음)
pip install stanza suparkanbun esupar

# Stanza 리소스 다운로드
python -c "import stanza; stanza.download('ko'); stanza.download('zh')"

# 검증 (둘 다 OK 출력되어야 함)
python -c "from FlagEmbedding import BGEM3FlagModel; print('OK')"
python -c "from transformers import BertForTokenClassification; print('OK')"

mkdir -p test_results
```

### 2단계: 추론 + 평가

```bash
# === P2S 경계모델 ON ===
python p2s/main.py datasets/splits/paragraph_test.xlsx \
  test_results/p2s_on.xlsx \
  --checkpoint-path test_results/p2s_on_ckpt.csv

python accuracy/p2s_evaluator.py \
  test_results/p2s_on.xlsx datasets/splits/sentence_test.xlsx -v

# === P2S 경계모델 OFF ===
python p2s/main.py datasets/splits/paragraph_test.xlsx \
  test_results/p2s_off.xlsx \
  --no-boundary-model --checkpoint-path test_results/p2s_off_ckpt.csv

python accuracy/p2s_evaluator.py \
  test_results/p2s_off.xlsx datasets/splits/sentence_test.xlsx -v

# === S2P 경계모델 ON ===
python s2p/main.py datasets/splits/sentence_test.xlsx \
  test_results/s2p_on.xlsx

python accuracy/s2p_evaluator.py \
  datasets/splits/phrase_test.xlsx test_results/s2p_on.xlsx -v

# === S2P 경계모델 OFF ===
python s2p/main.py datasets/splits/sentence_test.xlsx \
  test_results/s2p_off.xlsx --no-boundary-model

python accuracy/s2p_evaluator.py \
  datasets/splits/phrase_test.xlsx test_results/s2p_off.xlsx -v
```

### 3단계: 결과 회수 및 종료

결과 파일 다운로드 후 **반드시 Stop 버튼으로 Pod 중지** (과금 방지).

---

## 경계 모델 플래그 (p2s, s2p 공통)

| 플래그 | 동작 |
|--------|------|
| `--use-boundary-model` | 경계 모델 ON (기본값) |
| `--no-boundary-model` | 경계 모델 OFF |

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| `No module named 'stanza'` | pip install 누락 | `pip install stanza suparkanbun esupar` |
| `BertForTokenClassification` not found | transformers 4.57.3 비호환 | `pip install transformers==4.48.0 --force-reinstall` |
| torchvision import 에러 | torch 버전 불일치 | `pip uninstall torchvision torchaudio -y` |
| BGE-M3 "구버전 API 시뮬레이션" 경고 + 극도로 느림 | FlagEmbedding이 최적 경로 사용 못함 | transformers==4.48.0 + FlagEmbedding 재설치 |
| venv에서 패키지 못 찾음 | RunPod에서 venv 격리 실패 | venv 사용하지 말고 시스템 Python 사용 |
| 웹 터미널 붙여넣기 안 됨 | 브라우저 보안 | Ctrl+Shift+V 또는 Jupyter Terminal 사용 |

---

## 핵심 주의사항

1. **venv 사용 금지**: RunPod 환경에서 `python -m venv`는 시스템 패키지와 격리되지 않음. 시스템 Python에 직접 설치할 것
2. **torchvision/torchaudio 제거 필수**: 이 프로젝트에서 미사용이며, torch 버전 충돌만 유발
3. **transformers 버전**: 4.48.0이 suparkanbun + FlagEmbedding 모두와 호환되는 검증된 버전
4. **Pod 종료 잊지 말 것**: On-Demand 과금 주의

---

작성: 2026-02-05
