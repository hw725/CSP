#!/usr/bin/env python3
"""
Hyeonto 전체 리서치 파이프라인 자동 실행 스크립트

실행 순서:
1. xlsx → CSV 변환 (hyeonto/datasets/*.xlsx → datasets/*.csv)
2. 데이터 정규화 (phrase_normalized.csv 생성)
3. 마커 분류 (phase4_premodern_classify.py)
4. (선택) LLM 분석 (dansa_full_survey.py - API 키 필요)
5. 시각화 생성 (generate_*.py)
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "datasets"
CSP_DATASETS_DIR = BASE_DIR.parent / "datasets"
SCRIPTS_DIR = BASE_DIR / "scripts"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"

# 로그 파일
LOG_FILE = RESULTS_DIR / "pipeline_log.txt"
RESULTS_DIR.mkdir(exist_ok=True)

def log(message: str):
    """콘솔과 로그 파일에 출력"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {message}"
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def run_script(
    script_name: str,
    script_path: Path = None,
    args: list = None,
    cwd: Path = None,
    skip_if_failed: bool = False,
) -> bool:
    """파이썬 스크립트 실행"""
    if script_path is None:
        script_path = BASE_DIR / script_name
    if cwd is None:
        cwd = BASE_DIR

    if not script_path.exists():
        log(f"❌ 스크립트 없음: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    log(f"\n{'='*70}")
    log(f"▶ {script_name} 실행 중...")
    log(f"{'='*70}")

    try:
        result = subprocess.run(cmd, cwd=str(cwd), capture_output=False)
        if result.returncode == 0:
            log(f"✅ {script_name} 완료")
            return True
        else:
            log(f"❌ {script_name} 실패 (exit code: {result.returncode})")
            if skip_if_failed:
                log(f"   (계속 진행...)")
            return False
    except Exception as e:
        log(f"❌ {script_name} 오류: {e}")
        return False

def convert_xlsx_to_csv():
    """Step 1: hyeonto/datasets/*.xlsx → CSV 변환"""
    log("\n" + "=" * 70)
    log("STEP 1: XLSX → CSV 변환")
    log("=" * 70)

    xlsx_files = {
        "sentence.xlsx": "sentence_full.csv",
        "phrase.xlsx": "phrase_full.csv",
        "paragraph.xlsx": "paragraph_full.csv",
    }

    for xlsx_name, csv_name in xlsx_files.items():
        xlsx_path = DATASETS_DIR / xlsx_name
        csv_path = DATASETS_DIR / csv_name

        if not xlsx_path.exists():
            log(f"⚠️  {xlsx_name} 없음")
            continue

        if csv_path.exists():
            log(f"✅ {csv_name} 이미 존재 (스킵)")
            continue

        try:
            log(f"변환 중: {xlsx_name} → {csv_name}")
            df = pd.read_excel(xlsx_path)
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            log(f"✅ {csv_path} 저장 ({len(df):,}행)")
        except Exception as e:
            log(f"❌ {xlsx_name} 변환 실패: {e}")

def check_required_data():
    """Step 2: 필요한 데이터 확인"""
    log("\n" + "=" * 70)
    log("STEP 2: 필수 데이터 확인")
    log("=" * 70)

    # phrase_normalized.csv 확인
    phrase_norm_path = DATASETS_DIR / "phrase_normalized.csv"
    if phrase_norm_path.exists():
        df = pd.read_csv(phrase_norm_path)
        log(f"✅ phrase_normalized.csv 존재 ({len(df):,}행)")
        return True
    else:
        log(f"⚠️  phrase_normalized.csv 없음")
        # 대안: phrase_full.csv에서 생성 시도
        phrase_full_path = DATASETS_DIR / "phrase_full.csv"
        if phrase_full_path.exists():
            log(f"📋 phrase_full.csv로부터 정규화 데이터 생성 예정")
            return True
        return False

def run_hyeonto_pipeline():
    """Step 3-6: Hyeonto 분석 파이프라인 실행"""

    steps = [
        # Step 3: 마커 분류
        (
            "phase4_premodern_classify.py",
            BASE_DIR / "phase4_premodern_classify.py",
            ["--input", str(DATASETS_DIR / "phrase_normalized.csv")],
            False,
        ),  # 필수
        # Step 4: LLM 분석 (선택, API 키 필요)
        (
            "dansa_full_survey.py",
            SCRIPTS_DIR / "dansa_full_survey.py",
            ["--input", str(DATASETS_DIR / "phrase_normalized.csv")],
            True,
        ),  # 실패해도 계속
    ]

    for i, (script_name, script_path, args, skip_on_fail) in enumerate(steps, 3):
        log("\n" + "=" * 70)
        log(f"STEP {i}: {script_name}")
        log("=" * 70)

        run_script(script_name, script_path, args, skip_if_failed=skip_on_fail)

def run_visualizations():
    """Step 7: 시각화 생성 (선택)"""
    log("\n" + "=" * 70)
    log("STEP 7: 시각화 생성 (선택)")
    log("=" * 70)

    # 간단한 시각화부터 시작
    viz_scripts = [
        ("analyze_k3_clusters.py", None, False),
        ("generate_all_visualizations.py", ["--skip-umap"], True),
    ]

    for script_name, args, skip_on_fail in viz_scripts:
        script_path = BASE_DIR / script_name
        run_script(script_name, script_path, args, skip_if_failed=skip_on_fail)

def main():
    """메인 파이프라인"""
    start_time = datetime.now()

    log("=" * 70)
    log("🔬 Hyeonto 전체 리서치 파이프라인 시작")
    log("=" * 70)
    log(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"작업 디렉토리: {BASE_DIR}")

    try:
        # Step 1: XLSX → CSV 변환
        convert_xlsx_to_csv()

        # Step 2: 필수 데이터 확인
        if not check_required_data():
            log("⚠️  필수 데이터 부족. 일부 단계 스킵 가능")

        # Step 3-6: 핵심 분석 파이프라인
        run_hyeonto_pipeline()

        # Step 7: 시각화 (선택)
        log("\n" + "=" * 70)
        log("시각화 생성 스킵 (선택 사항)")
        log("필요하면: python generate_all_visualizations.py --skip-umap")
        log("=" * 70)

        # 완료
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds() / 60

        log("\n" + "=" * 70)
        log("✅ 파이프라인 완료!")
        log("=" * 70)
        log(f"종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"소요 시간: {elapsed:.1f}분")
        log(f"\n📊 결과 위치:")
        log(f"   - 분류 결과: {REPORTS_DIR / 'phase4'}")
        log(f"   - 통계 분석: {RESULTS_DIR}")
        log(f"   - 로그: {LOG_FILE}")

    except Exception as e:
        log(f"\n❌ 파이프라인 오류: {e}")
        import traceback

        log(traceback.format_exc())

if __name__ == "__main__":
    main()
