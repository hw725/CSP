# CSP — 에이전트 컨텍스트

> 상세 감사 결과는 루트의 cognitive-debt-audit.html, 운영 문서는 head-repo\docs\유지보수문서\01_CSP_병렬정렬파이프라인.md 참조.

## 해소 로그

### 2026-07-06 — git 엉킴 3종 정리
- **[함정 1 해소]** `xml_extractor/`를 추적 편입(커밋). `git clean` 붕괴 위험 제거. 감사 문서(AGENTS.md·cognitive-debt-audit.html)도 함께 추적.
- **[함정 2 해소]** 4개월 정지 리팩터를 논리적 1커밋으로 마무리 — 추출 로직을 `xml_extractor`로 분리(−394줄), `xml_pipeline`은 re-export만, `xlsx_scripts` 9종 추적.
- **[함정 3 해소]** 로컬 `main`↔`csp/main` 11:11 분기는 최종 트리 동일한 이중커밋(허깨비)로 확인 → `reset --soft`로 원격과 역사 일치. `origin/main`(6bb01dc)은 비표준 refspec(`+refs/heads/*:refs/remotes/csp/*`)이 만든 화석 ref이고, **진짜 원격은 `csp/main`**. force push 없이 정리됨.
- **[함정 4 해소]** 누락된 특수 서종 스크립트 4종(`xlsx_scripts/extract_yeogi.py`·`create_yeogi_paragraph.py`·`add_paragraph_id_to_gubyeollyeol.py`·`scripts/merge_parallel_xlsx.py`)을 git 이력(2026-01~02 시점)에서 복구. 복구본은 과거 버전이라 현행 파이프라인 대비 검증 후 사용.
- **[함정 5 해소]** 3중 복제본(`jti_code_mappings`·`xml_unit_parser`·`xml_file_browser`)의 `xml_pipeline` 사본을 `xml_extractor`를 re-export하는 얇은 껍데기로 교체. **정본=xml_extractor 단일화**, 한쪽만 고쳐 갈라질 위험 제거.
- **[잔여]** `codex/pre-rebase-5ec5931`·`dansa-research` 워크트리(별개 라인) 정리 여부만 미결.

## 인지 부채 지도 (2026-07-04 감사 시점 스냅샷 — 해소 항목은 위 로그 참조)

(2026-07-04 감사 기준. 코드 대부분이 AI 작성 — 아래는 "코드가 실제로 하는 일"과 "사용자가 안다고 착각하기 쉬운 지점"의 지도)

### 이 프로젝트가 실제로 하는 일
한문 원문↔한국어 번역문을 3단계 계층으로 정렬하는 파이프라인. 실제 데이터 흐름은 문서보다 한 단계 김:
1. **추출**: `sources/`(고전번역원 XML 134개, 92MB) → `xlsx_scripts/`·`xml_extractor/` → 문단·문장·구 3종 병렬 XLSX (`xlsx/` 44권 폴더)
2. **정렬(ML)**: `batch_44books.py` → P2S(`p2s/`, 문단→문장, F1 0.938) → S2P(`s2p/`, 문장→구, Viterbi+Punctuation Guard) → `xlsx_pipeline_results/`
3. **연구 분석**: `hyeonto/`(추적 파일 127개로 repo 최대 모듈, 현토 마커 클러스터링·대시보드) — 문서화 거의 없음

### 손대기 전 반드시 알아야 할 함정 (위험 순)
1. **`xml_extractor/`는 git 미추적(untracked)이다.** 그런데 미커밋 상태의 `xml_pipeline/__init__.py`·`xml_pipeline_cli.py`가 여기서 `XMLProcessor, XMLPair, XMLUnitParser`를 import한다. `git clean -fd` 또는 폴더 삭제 시 xml_pipeline 전체가 import 에러로 죽고 **복구 불가**(백업 없음).
2. **반쯤 끝난 리팩터가 4개월째 미커밋** (마지막 커밋 2026-02-26). staged: `xlsx_scripts/` 9개 파일(+1,372줄). unstaged: `xml_pipeline_processor.py`에서 −394줄(추출 로직을 xml_extractor로 이관). staged만 커밋하면 되는 게 아니라 **xml_pipeline 수정 + xml_extractor 신규 추적을 한 세트로 커밋해야** 클론이 동작한다.
3. **push 금지 상태**: 로컬 `main`(d64772f) ↔ `csp/main`(a8085a9)이 11 ahead/11 behind 분기. 두 HEAD의 커밋 메시지가 동일("xml_pipeline 복구") — 같은 작업이 양쪽에 따로 커밋된 흔적. rebase/merge 정리 전 push·force push 금지. fetch는 `refs/remotes/csp/*` 네임스페이스 사용(dansa-research 워크트리와 .git 공유 때문).
4. **유령 스크립트**: `xlsx_scripts/README.md`가 특수 서종 4권(예기집설대전1, 당시삼백수1~3)의 처리기로 `extract_yeogi.py`·`create_yeogi_paragraph.py`를 지시하지만 **repo 어디에도 없다**. 특수 4권은 `<단락 id>` 구조라 일반 스크립트(`xml_to_sentence_parallel.py`)가 처리 못함 → 현재 특수 4권 재생성 불가능. 실행 순서 관례: 일반 39권 먼저, 특수 4권 나중(덮어쓰기 방지).
5. **3중 복제 코드**: `jti_code_mappings.py`·`xml_unit_parser.py`·`xml_file_browser.py`가 `xml_pipeline/`과 `xml_extractor/`에 md5 동일 사본으로 존재. 한쪽만 고치면 조용히 갈라진다. canonical은 xml_extractor 쪽(리팩터 방향 기준)이나 미확정.
6. **설정 우선순위**: `common/config.py`는 `csp_config.json` 값이 **환경변수보다 우선**. json에 `device: "cuda:0"` 하드코딩 — GPU 없는 로컬에선 CSP_DEVICE 환경변수로 못 덮는다(json을 직접 고쳐야 함). API 키는 json이 null이라 `.env`의 OPENAI_API_KEY로 폴백.
7. **로컬 전용 대용량**: `models/*.pt`(78M)·`xlsx/`(139M)·`datasets/`(25G)·`cache/`(6.5G)·`sources/`(92M) 전부 git 밖. 이 PC가 유일본.

### 암묵 관례
- `batch_44books.py`는 `xlsx/{책}/{책}_문단병렬.xlsx` 파일명 규약에 하드 의존. 컬럼명(문단식별자/원문/번역문)도 규약.
- `main.py`는 "범용 진입점"이라지만 실제 동작은 XML 계열만: txt/xlsx 서브커맨드는 "미래 지원" 스텁. XLSX 정렬의 실제 진입은 `batch_44books.py` 또는 `p2s/main.py`·`s2p/main.py`.
- `renumber_excel_indices.py`가 원본 XML 식별자(ID:W1 등)를 1,2,3… 연번으로 덮어씀 — 원본 provenance가 이 단계에서 끊긴다.
- 무결성(1:1 정렬)은 S2P의 Punctuation Guard가 강제. 정렬 결과 검증은 `accuracy/*_evaluator.py`.
