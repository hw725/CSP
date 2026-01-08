# CSP 하이퍼파라미터 최적화 시스템 사용 가이드

## 🎯 개요

CSP 시스템에 텍스트별 최적화 설정이 통합되어 더욱 정확한 PA/SA 성능을 제공합니다.

### 🏆 성능 등급 시스템

- **S등급** (F1 ≥ 0.95): 최고 성능 - 미세 조정만 필요
- **A등급** (F1 0.90-0.95): 우수 성능 - 안정성 중심 설정  
- **B등급** (F1 0.85-0.90): 양호 성능 - 적극적 최적화 필요

## 🚀 빠른 시작

### 1. 최적화 자동 실행 (권장)

```bash
# Linux/Mac
./run_optimized.sh input.xlsx output both

# Windows  
run_optimized.bat input.xlsx output both
```

### 2. 개별 모듈 최적화 실행

```bash
# SA 최적화 (파일명에서 JTI 자동 감지)
python sa/main_optimized.py jti_4c0201-한유문초.xlsx output_sa.xlsx --optimize

# PA 최적화 (JTI 코드 직접 지정)  
python pa/main_optimized.py input.xlsx output_pa.xlsx --optimize --jti-code 4c0201

# 기본 설정 (최적화 없음)
python sa/main_optimized.py input.xlsx output.xlsx
```

## 📊 텍스트별 최적화 설정

| JTI코드 | 텍스트명 | 성능등급 | 목표F1 | 주요 최적화 |
|---------|----------|----------|--------|-------------|
| 4c0201 | 당송팔대가문초한유1 | B+ | 0.91 | SA 강화, 토큰 확장 |
| 4j0202 | 대학연의1 | B+ | 0.91 | 경계 검출, 토큰 최적화 |
| 4j0201 | 논어집주2 | B+ | 0.92 | 구문 힌트, 정밀 분할 |
| 3a0101 | 관자2 | A- | 0.93 | 안정성 중심 |
| 2k0101 | 사기열전1 | A | 0.94 | 기본 설정 유지 |

*전체 15개 텍스트 설정은 `config/text_optimization_configs.py` 참조*

## ⚙️ 주요 최적화 파라미터

### SA (문장 분할) 최적화
- **토큰 길이 조정**: 텍스트별 최적 min/max 토큰 수
- **DP 윈도우**: 동적 프로그래밍 탐색 범위  
- **경계 보너스**: 문장 경계 검출 가중치
- **유사도 감마**: 유사도 점수 샤프닝 정도

### PA (문단 정렬) 최적화  
- **길이 임계값**: 최대 문장 길이 제한
- **유사도 임계값**: 정렬 품질 기준점
- **배치 크기**: 병렬 처리 최적화

## 🔧 고급 사용법

### 1. 커스텀 최적화 설정

```python
# config/text_optimization_configs.py 수정
OPTIMIZATION_CONFIGS = {
    'custom_text': {
        'text_name': '커스텀 텍스트',
        'performance_grade': 'B',
        'target_f1': 0.88,
        'sa_config': {
            'min_src_tokens': 2,
            'max_src_tokens': 25,
            # ... 기타 설정
        }
    }
}
```

### 2. 최적화 검증

```bash
# 최적화 전후 성능 비교
python optimized_runner.py --text-code 4c0201 --validate
```

### 3. 배치 최적화 실행

```bash  
# 여러 텍스트 동시 최적화
python optimized_runner.py --batch-mode --config-file batch_config.json
```

## 📈 성능 개선 현황

### 기대 성능 향상
- **한유문초** (4c0201): 0.8625 → 0.91 (+0.0475)
- **대학연의1** (4j0202): 0.8690 → 0.91 (+0.041)  
- **논어집주2** (4j0201): 0.8919 → 0.92 (+0.0281)
- **관자2** (3a0101): 0.9281 → 0.93 (+0.0019, 안정성 개선)

### 검증된 개선사항
- ✅ **대학연의1**: XML 구조 문제 해결로 PA 추출 성공 (0→141 문단)
- ✅ **전체 시스템**: S/A/B 등급별 맞춤 설정 완료
- ✅ **자동화**: JTI 코드 감지 및 설정 자동 적용

## 🛠️ 문제 해결

### 1. JTI 코드 감지 실패
```bash
# 수동으로 JTI 코드 지정
python sa/main_optimized.py input.xlsx output.xlsx --optimize --jti-code 4c0201
```

### 2. 메모리 부족 시
```bash
# 배치 크기 줄이기
python pa/main_optimized.py input.xlsx output.xlsx --optimize --batch-size 25
```

### 3. 성능이 목표에 못 미칠 때
1. `config/text_optimization_configs.py`에서 파라미터 조정
2. `--verbose` 옵션으로 상세 로그 확인
3. 개별 모듈 실행으로 문제 영역 특정

## 📝 로그 및 디버깅

### 상세 로그 출력
```bash
python sa/main_optimized.py input.xlsx output.xlsx --optimize --verbose
```

### 설정 확인
```python
from config.text_optimization_configs import get_sa_config_by_jti
config = get_sa_config_by_jti('4c0201')
print(config)
```

## 🔄 지속적 개선

1. **성능 모니터링**: 정기적으로 F1 점수 측정
2. **설정 업데이트**: 새로운 텍스트나 개선된 파라미터 추가  
3. **사용자 피드백**: 실제 사용 경험을 바탕으로 조정

---

더 자세한 정보는 `텍스트별_하이퍼파라미터_최적화_가이드.md`를 참조하세요.