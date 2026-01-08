# 현토(懸吐) 분석 프로젝트

> 사서삼경(四書三經) 등 유교 경전의 **현토(懸吐)** 패턴을 클러스터링 및 시각화하여 한문 구문 기능을 분석하는 연구 프로젝트

---

## 📚 프로젝트 개요

### 현토(懸吐)란?

**현토(懸吐)**는 한문 원문에 한글 토씨(조사, 어미 등)를 붙여 한국어로 읽을 수 있도록 하는 전통적인 독법입니다.

예시:
- 원문: `大學之道는 在明明德하며`
- 한자: `大學之道` / 현토: `는`
- 한자: `在明明德` / 현토: `하며`

### 연구 목표

1. **현토 패턴 클러스터링**: 한문 경계(boundary)에서 나타나는 현토의 기능적 역할 분류
2. **의미역 시각화**: 현토 마커(marker)들의 분포를 2D 공간에 투영하여 유사성 분석
3. **문단/문장 정렬 지원**: CSP 프로젝트의 PA(문단 정렬) 모델 학습 데이터 구축

---

## 📂 디렉토리 구조

```
hyeonto/
├── _tmp_selected_sources/     # 선택된 사서 XML 원본 (논어·맹자·대학·중용)
├── datasets/                  # 학습용 데이터셋
│   ├── pa/                    # 문단 정렬(PA) 학습 데이터
│   │   ├── train.csv          # 학습셋
│   │   ├── val.csv            # 검증셋
│   │   └── test.csv           # 테스트셋
│   ├── pd/                    # 문단 분할(PD) 데이터
│   └── sa/                    # 문장 정렬(SA) 데이터
├── reports/                   # 분석 결과 (gitignore)
│   ├── k16_analysis_minper50/ # K=16 클러스터 분석 결과
│   └── recluster_k16_child_minper50/ # 2단계 재클러스터링 결과
├── xlsx/                      # Excel 형식 산출물
└── jti_*.xml                  # 원본 현토 XML 파일들
```

---

## 📖 분석 대상 텍스트

### 사서(四書)
| 코드 | 서명 | 파일 |
|------|------|------|
| 1h0301 | 논어집주(論語集註) | `jti_1h0301-[현토]논어집주_*.xml` |
| 1h0601 | 맹자집주(孟子集註) | `jti_1h0601-[현토]맹자집주_*.xml` |
| 1h0801 | 대학장구(大學章句) | `jti_1h0801-[현토]대학장구_*.xml` |
| 1h1001 | 중용장구(中庸章句) | `jti_1h1001-[현토]중용장구_*.xml` |

### 삼경(三經)
| 코드 | 서명 | 파일 |
|------|------|------|
| 1a0201-02 | 주역전의(周易傳義) 상/하 | `jti_1a0201-[현토]주역전의_*.xml` |
| 1b0201-02 | 서경집전(書經集傳) 상/하 | `jti_1b0201-[현토]서경집전_*.xml` |
| 1c0201-02 | 시경집전(詩經集傳) 상/하 | `jti_1c0201-[현토]시경집전_*.xml` |

---

## 🔬 XML 데이터 구조

각 XML 파일은 다음과 같은 계층 구조를 가집니다:

```xml
<동양고전>
  <ViewdocList>
    <장 편명="...">
      <document level="N">
        <제목 type="장">...</제목>
        <단락 type="P" id="N">
          <s id="N">              <!-- 문장(sentence) -->
            <c id="N">            <!-- 구(clause) -->
              <w id="N">漢字現吐</w>  <!-- 어절(word) -->
            </c>
          </s>
        </단락>
      </document>
    </장>
  </ViewdocList>
</동양고전>
```

### 어절(word) 형식
- **일반 현토**: `漢字토씨` (예: `大學之道는`, `在明明德하며`)
- **생략 표시**: `[-토씨]` (예: `[-則]`, `[-而]`)
- **한자 단독**: `漢字` (현토 없이 한자만 있는 경우)

---

## 🛠 분석 파이프라인

### 1단계: 데이터셋 구축
```bash
python scripts/hyeonto_build_datasets.py
```
- XML 파일에서 문단/문장/구 경계 정보 추출
- PA(문단 정렬) 학습용 train/val/test 분할

### 3단계: Joint Embedding (Parent + Marker) 시각화

```bash
# 도커 환경에서 실행 (의존성 포함)
powershell -ExecutionPolicy Bypass -File .\docker.ps1 python scripts/visualize_parent_marker_joint_embedding_ext.py \
    --csv hyeonto/reports/recluster_k16_child_minper50/reclustered.csv \
    --out-dir hyeonto/reports/joint_embedding_viz \
    --method umap \
    --saseo-weight 5.0 \
    --label-top-markers 40
```

**주요 기능:**
- **Shared Coordinate System**: Parent 클러스터(다이아몬드)와 Marker(원형)를 동일 공간에 배치.
- **사서(四書) 가중치 부여**: 기준 용례인 사서 텍스트의 패턴을 더 강하게 반영 (`--saseo-weight 5.0`).
- **클러스터 영역(Convex Hull)**: Parent 별로 영역 테두리를 표시하여 영역성 시각화.
- **다양한 버전 지원**: Parent-only 버전과 Parent+Child 버전 HTML을 각각 생성.

---

### 주요 옵션
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--k` | 16 | Parent 클러스터 수 |
| `--saseo-weight` | 2.0 | 사서(Four Books) 텍스트에 부여할 가중치 |
| `--dim` | 2 | 시각화 차원 (2 또는 3) |
| `--method` | umap | 차원 축소 방법 (umap, pca, tsne) |
| `--clean` | - | reports 폴더 초기화 후 재생성 |

---

### 2단계: 경계 기능 클러스터링
```bash
python scripts/hyeonto_train_and_visualize.py --device-id 0
```

**파이프라인 구성:**
1. **Parent 클러스터링 (K=16)**: 경계 문맥의 임베딩을 KMeans로 분류
2. **Child 재클러스터링**: 각 parent 내부를 다시 세분화
3. **시각화**: 현토 마커의 분포를 2D 공간에 투영

### 주요 옵션 (경계 기능 클러스터링)
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--k` | 16 | Parent 클러스터 수 |
| `--child-k` | 16 | Child 클러스터 수 |
| `--parent-min-size` | 50 | 재클러스터링 대상 parent 최소 크기 |
| `--min-per-child` | 50 | Child K 자동 축소 기준 |
| `--max-boundaries` | 20000 | 분석 대상 최대 경계 수 |
| `--clean` | - | reports 폴더 초기화 후 재생성 |

---

## 📊 산출물 설명

### 시각화 결과 (`reports/k16_analysis_minper50/`)

| 파일 | 설명 |
|------|------|
| `marker_semantic_embedding.html` | 현토 마커의 2D 분포 (interactive) |
| `marker_semantic_embedding.png` | 현토 마커의 2D 분포 (static) |
| `marker_semantic_embedding.csv` | 마커별 좌표, 빈도, dominant 그룹 정보 |
| `group_embedding_parent.html` | Parent 클러스터의 2D 분포 |
| `group_embedding_parent_child.html` | Parent_Child 클러스터의 2D 분포 |
| `biplot_group_marker_*.html` | CA(대응분석) biplot: 그룹과 마커 동시 표시 |

### 프로필 리포트 (`reports/k16_analysis_minper50/`)

| 파일 | 설명 |
|------|------|
| `group_profiles_parent.md` | Parent 클러스터별 대표 마커 및 예문 |
| `group_profiles_parent_child.md` | Parent_Child 클러스터별 대표 마커 및 예문 |
| `final_cluster_table_*.csv` | 클러스터별 상세 통계 |

---

## 🔍 현토 분석 예시

### 클러스터별 대표 현토

시각화 결과에서 각 클러스터는 특정 구문 기능을 나타내는 현토들이 모입니다:

- **종결형**: `라`, `니라`, `하니라` (문장 종결)
- **접속형**: `하고`, `하며`, `하여` (병렬 연결)
- **조건형**: `면`, `이면`, `하면` (조건문)
- **주격형**: `이`, `는`, `가` (주어 표지)
- **목적격**: `를`, `을` (목적어 표지)

### Lift 기반 특징 마커

각 그룹의 **lift(과대표)** 지표로 해당 그룹에서 특히 많이 나타나는 현토를 식별합니다:

```
그룹 p3: lift 상위 = 요, 라, 니라  → 종결 기능
그룹 p7: lift 상위 = 하고, 하며    → 병렬 접속 기능
그룹 p12: lift 상위 = 면, 거든    → 조건절 기능
```

---

## 💡 활용 방안

### 1. PA(문단 정렬) 모델 개선
- 현토 패턴을 경계 감지 feature로 활용
- 클러스터 정보를 boundary score 개선에 반영

### 2. 한문 구문 연구
- 현토의 구문론적 기능 분류 체계 수립
- 경전별/시대별 현토 사용 패턴 비교

### 3. 한문 교육 자료 개발
- 현토 암기 순서 최적화 (빈도/기능별 분류)
- 유사 현토 그룹화로 학습 효율 증대

---

## ⚠️ 주의사항

1. **gitignore**: `reports/`, `datasets/` 폴더는 `.gitignore`에 포함되어 있습니다.
2. **GPU 권장**: 임베딩 계산에 GPU 사용 시 성능이 크게 향상됩니다.
3. **한글 폰트**: PNG 생성 시 한글 폰트(NanumGothic 등)가 필요합니다.

---

## 📚 관련 문서

- [CSP 프로젝트 README](../README.md) - 전체 프로젝트 개요
- [PA 워크플로우](../documents/WORKFLOW.md) - 문단 정렬 상세 설명
- [무결성 검증](../documents/TROUBLESHOOTING.md) - 데이터 검증 가이드

---

**마지막 업데이트**: 2026년 1월 7일
