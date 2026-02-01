# ? hyeonto/reports/ 디렉토리 구조

본 디렉토리에는 **현토 분석 결과물**이 저장됩니다.

---

## ? 핵심 파일

| 파일 | 설명 |
|:---|:---|
| **dashboard.html** | ? **통합 분석 대시보드** - 모든 결과 인터랙티브 탐색 |
| **md_viewer.html** | 마크다운 렌더링 뷰어 |
| **FINAL_ANALYSIS_REPORT.md** | ? **통합 마스터 리포트** |
| **K4_CLUSTER_ANALYSIS.md** | K=4 집중 분석 보고서 |

### 대시보드 실행
```bash
cd hyeonto/reports
python -m http.server 8080
# 브라우저에서 http://localhost:8080/dashboard.html 접속
```
또는:
```bash
# Windows
serve.bat  # 자동 브라우저 열기
```

---

## ? K=4 핵심 분석 결과

### 정규화 클러스터링 (Zero-Gap 171-항목 스키마)
| 폴더 | 설명 |
|:---|:---|
| `sentence_k4_normalized/` | Sentence K=4 클러스터 프로파일 (150,545건) |
| `phrase_k4_normalized/` | Phrase K=4 클러스터 프로파일 (366,222건) |

### UMAP 임베딩 시각화
| 파일 | 설명 |
|:---|:---|
| `k4_embedding_overlay_3d.html` | ? Sentence/Phrase 3D UMAP 오버레이 |
| `k4_embedding_overlay_2d.html` | Sentence/Phrase 2D UMAP 오버레이 |
| `k4_sentence_phrase_overlay_normalized.html` | Sentence/Phrase 서종 분포 비교 |

### Sankey 다이어그램
| 파일 | 설명 |
|:---|:---|
| `sankey_diagrams/sankey_sentence4_phrase4.html` | Sentence K=4 ↔ Phrase K=4 클러스터 흐름 |

---

## ? 주제별 분석

### `genre_examples/`
- `GENRE_SPECIFIC_EXAMPLES.md`: 서종별 현토 패턴 비교 분석
- 사서(四書), 삼경(三經), 사서(史書), 집부(集部) 대표 예시

### `first_person_analysis/`
- `FIRST_PERSON_MARKER_ANALYSIS.md`: 1인칭 화자 표지 분석

### `normalization_review/`
- 정규화 갭 분석 결과 (Zero-Gap 달성)

---

## ? Exploratory 분석

### `exploratory/`
탐색적 분석 및 다양한 시각화
- `cooccurrence_normalized/`: 한자-현토 공기 네트워크 (정규화, 라이트/다크 테마)
- `outliers_sentence/`: Sentence 이상치 분석
- `ngram_phrase/`: Phrase N-gram 분석
- `phonetic/`: 음운 패턴 분석

---

## ? 코퍼스 정전성(Canonicity) 노트

### 사서(四書)와 삼경(三經)의 정전적 가치

동양고전종합DB에 수록된 사서와 삼경은 **조선시대 언해본을 거의 그대로 계승**하여 정전적(canonical) 자료로서의 가치를 지닙니다.

| 서종 | 정전성 | 현토 특성 |
|:---|:---|:---|
| **사서(四書)** | ? 높음 | 논증적 어미 (`면`, `니라`) 중심 |
| **삼경(三經)** | ? 높음 | 정의적 어미 (`는`, `라`) 중심, 고어투 다소 많음 |
| 사서(史書) | 중간 | 서사적 연결 (`하고`, `하니`) |
| 집부(集部) | 중간 | 혼합 패턴 |

**삼경의 고어투 경향**: 삼경은 사서보다 `할새`, `어늘`, `러니` 등 중세국어 계통 어미가 상대적으로 많이 관찰됩니다. 이는 언해 시기 차이 또는 원전 문체 특성에 기인합니다.

---

## ? 분석 버전 이력

| 버전 | 날짜 | 주요 변경 |
|:---:|:---:|:---|
| **Production** | 2026-01-27 | K=4 핵심 분석 집중, Zero-Gap 171-스키마, 150,545/366,222건 코퍼스 |
| v6+Multi | 2026-01-13 | 다중 해상도(K=4/14/24) 분석 |
| v6 | 2026-01-11 | 번역문(tgt) 포함 임베딩으로 전면 재분석 |

---

**최종 업데이트**: 2026-01-27
