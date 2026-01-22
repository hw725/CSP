---
description: 장시간 테스트/파이프라인 실행 전 필수 체크리스트
---

# Pre-Flight Checklist

장시간(10분+) 테스트 또는 파이프라인 실행 전 **반드시** 다음을 확인하세요.

## 1. 파라미터 확인

```
□ max-workers 값 확인 (권장: 4 이하)
□ batch-size 값 확인 (권장: 64 이하)
□ 경계 모델 활성화 여부 확인 (--use-boundary-model)
□ 기타 중요 플래그 확인
```

**확인 방법:**
```bash
# main.py에서 기본값 확인
grep -n "default=" <module>/main.py | head -20
```

## 2. 스모크 테스트

```
□ 알려진 문제 케이스로 먼저 테스트 (10개 샘플)
□ 입력 개수 = 출력 개수 확인
□ 에러/경고 메시지 확인
```

**확인 방법:**
```bash
# 샘플 테스트 실행
docker-compose exec -T csp python <module>/main.py <sample_input> <sample_output>
# 결과 검증
python -c "import pandas as pd; print(len(pd.read_excel('<output>')))"
```

## 3. 리소스 확인

```
□ GPU 메모리 여유 확인
□ 디스크 공간 확인
□ 다른 실행 중인 프로세스 확인
```

**확인 방법:**
```bash
docker-compose exec -T csp nvidia-smi
docker-compose exec -T csp df -h
docker-compose exec -T csp ps aux | grep python
```

## 4. 로깅/체크포인트

```
□ 로그 파일 경로 지정
□ 체크포인트 활성화 (가능한 경우)
□ unbuffered 출력 활성화 (-u 옵션)
```

## 5. 변경사항 확인

```
□ 이전 성공 실행과 무엇이 달라졌는가?
□ 코드 변경사항 영향 확인
□ 마이그레이션 후 파라미터 일관성 검토
```

## 6. 예상 시간

```
□ 예상 소요 시간 계산 (샘플 기준 추정)
□ 완료 예정 시간 사용자에게 보고
```

---

## 체크리스트 완료 보고 형식

모든 항목 확인 후 다음 형식으로 사용자에게 보고:

```
## Pre-Flight 체크리스트 완료

### 파라미터
- max-workers: 4 ✅
- batch-size: 64 ✅
- use-boundary-model: True ✅

### 스모크 테스트
- 입력 10개 → 출력 10개 ✅
- 에러 없음 ✅

### 리소스
- GPU 메모리: 8GB / 24GB 여유 ✅
- 디스크: 50GB 여유 ✅

### 예상 시간
- 예상 소요: 약 4시간
- 완료 예정: 오후 6시

테스트를 시작하시겠습니까?
```

---

## ⚠️ 체크리스트 미완료 시

**절대로** 장시간 테스트를 시작하지 마세요.
체크리스트 미완료 상태에서 테스트 시작 시 → `/post-mortem` 작성 의무.
