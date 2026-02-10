# 클라우드 GPU 테스트 실행 가이드

> RunPod에서 P2S/S2P 추론 + F1 평가를 실행하는 절차

## 사전 준비

### 1. 테스트 데이터 및 모델이 Git에 포함되어 있어야 함

- `datasets/splits/*_test.xlsx` (paragraph, sentence, phrase)
- `models/*.pt` (boundary_multitask, dual_encoder_alignment_p2s/s2p, s2p_crossattn_boundary)
- main 브랜치에 머지 완료 (PR #5, #6)

### 2. RunPod Pod 생성

| 항목 | 설정값 |
|------|--------|
| Template | RunPod PyTorch 2.4.0 |
| GPU | **H200 SXM** ($3.59/hr) - H100 대비 더 빠르고 총 비용 유사 |
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
git clone https://github.com/hw725/CSP.git
cd CSP

# 충돌 패키지 제거 (필수 - torchvision/torchaudio가 torch 버전 충돌 유발)
pip uninstall torchvision torchaudio -y

# 패키지 설치
pip install -r requirements.txt

# transformers 호환 버전으로 교체
# - 4.57.3은 suparkanbun과 비호환 (BertForTokenClassification import 실패)
# - 4.48.0은 suparkanbun + FlagEmbedding 모두 호환
pip install transformers==4.48.0 FlagEmbedding>=1.2.0 --force-reinstall

# FlagEmbedding 최신 API 강제 적용 (구버전 시뮬레이션 모드 방지)
pip install FlagEmbedding>=1.2.0 --force-reinstall --no-deps

# Stanza/SuPar 설치 (requirements.txt에서 누락될 수 있음)
pip install stanza suparkanbun esupar

# Stanza 리소스 다운로드
python -c "import stanza; stanza.download('ko'); stanza.download('zh')"

# 검증 (3개 모두 OK 출력되어야 함)
python -c "from FlagEmbedding import BGEM3FlagModel; print('OK')"
python -c "from transformers import BertForTokenClassification; print('OK')"
python -c "from FlagEmbedding import BGEM3FlagModel; m = BGEM3FlagModel('BAAI/bge-m3'); r = m.encode(['test'], return_dense=True, return_sparse=True, return_colbert_vecs=True); print('OK - 최신 API')"

mkdir -p test_results
```

### 2단계: 스크립트 생성 (웹 터미널 호환)

RunPod 웹 터미널은 heredoc/multi-line 붙여넣기가 안 되므로, Python으로 스크립트를 생성한다. `>>>` 프롬프트에서 **한 줄씩** 붙여넣기:

```bash
python
```

```python
f = open('run_all.sh', 'w')
f.write('set -e\n')
f.write('cd /workspace/CSP\n')
f.write('mkdir -p test_results\n')
f.write('python p2s/main.py datasets/splits/paragraph_test.xlsx test_results/p2s_on.xlsx --checkpoint-path test_results/p2s_on_ckpt.csv\n')
f.write('python accuracy/p2s_evaluator.py test_results/p2s_on.xlsx datasets/splits/sentence_test.xlsx -v\n')
f.write('python p2s/main.py datasets/splits/paragraph_test.xlsx test_results/p2s_off.xlsx --no-boundary-model --checkpoint-path test_results/p2s_off_ckpt.csv\n')
f.write('python accuracy/p2s_evaluator.py test_results/p2s_off.xlsx datasets/splits/sentence_test.xlsx -v\n')
f.write('python s2p/main.py datasets/splits/sentence_test.xlsx test_results/s2p_on.xlsx\n')
f.write('python accuracy/s2p_evaluator.py datasets/splits/phrase_test.xlsx test_results/s2p_on.xlsx -v\n')
f.write('python s2p/main.py datasets/splits/sentence_test.xlsx test_results/s2p_off.xlsx --no-boundary-model\n')
f.write('python accuracy/s2p_evaluator.py datasets/splits/phrase_test.xlsx test_results/s2p_off.xlsx -v\n')
f.close()
exit()
```

확인 후 백그라운드 실행:
```bash
cat run_all.sh
nohup bash run_all.sh > test_results/all_runs.log 2>&1 &
```

진행 확인:
```bash
tail -20 test_results/all_runs.log
```

### 3단계: 결과 회수 및 종료

평가 결과 확인:
```bash
tail -50 test_results/all_runs.log
```

결과 파일 다운로드 후 **반드시 Stop 버튼으로 Pod 중지** (과금 방지).
테스트 완전히 끝나면 Pod를 **Terminate**(삭제)해야 Volume Disk 과금도 멈춘다.

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
| 웹 터미널 heredoc/multi-line 안 됨 | 웹 터미널 제한 | Python `f.write()` 방식으로 스크립트 생성 (2단계 참조) |
| `bash: !/bin/bash: event not found` | `!`가 bash 히스토리 확장으로 해석됨 | 쌍따옴표 대신 홑따옴표 사용 |

---

## 핵심 주의사항

1. **venv 사용 금지**: RunPod 환경에서 `python -m venv`는 시스템 패키지와 격리되지 않음. 시스템 Python에 직접 설치할 것
2. **torchvision/torchaudio 제거 필수**: 이 프로젝트에서 미사용이며, torch 버전 충돌만 유발
3. **transformers 버전**: 4.48.0이 suparkanbun + FlagEmbedding 모두와 호환되는 검증된 버전
4. **Pod 종료 잊지 말 것**: On-Demand 과금 주의

---

---

## 테스트 결과 (2026-02-09, RunPod H200 SXM)

### P2S (4,934문단 전체)

| 지표 | 결과 |
|------|------|
| **F1** | **0.9384** |
| Precision | 1.0 |
| Recall | 0.8840 |
| 원문유사도 | 0.9759 |
| 처리 시간 | ~5.2시간 |
| 무결성 FAIL | 228개 (골드 데이터 이슈, 프로세서 정상) |

> 228개 무결성 FAIL은 전부 '사정전훈의자치통감강목' 책에서 발생. 원인: 머지 스크립트의 컬럼 매핑 오류로 sentence_test.xlsx 원문 필드에 문단 ID 숫자가 들어감. 수정 완료.

### S2P (100행 샘플, Docker)

| 지표 | v2 Phrase Alignment | Baseline (DP only) |
|------|-------|---------|
| **F1** | **0.6900** | 0.1827 |
| Precision | 1.0 | - |
| Recall | 0.5267 | 0.1005 |
| 평균 유사도 | 0.8461 | 0.7559 |

---

작성: 2026-02-05
최종 수정: 2026-02-10 - P2S/S2P 전체 테스트 결과 추가
