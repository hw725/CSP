#!/usr/bin/env python3
"""43권 전체 PA+SA 배치 처리"""

import subprocess
import os
from pathlib import Path
import time

# 43권 폴더 목록 (xlsx/* 하위 폴더명 기준)
books = [
    "예기집설대전1", "예기집설대전2",
    "춘추좌씨전1", "춘추좌씨전2", "춘추좌씨전3", "춘추좌씨전4", 
    "춘추좌씨전5", "춘추좌씨전6", "춘추좌씨전7", "춘추좌씨전8",
    "자치통감강목1", "자치통감강목2", "자치통감강목3", "자치통감강목4",
    "자치통감강목5", "자치통감강목6", "자치통감강목7",
    "당시삼백수1", "당시삼백수2", "당시삼백수3",
    "당송팔대가문초한유1", "당송팔대가문초한유2", "당송팔대가문초한유3",
    "당송팔대가문초유종원1", "당송팔대가문초유종원2",
    "당송팔대가문초구양수1", "당송팔대가문초구양수2", "당송팔대가문초구양수3",
    "당송팔대가문초구양수4", "당송팔대가문초구양수5", "당송팔대가문초구양수6",
    "당송팔대가문초소순1",
    "당송팔대가문초소식1", "당송팔대가문초소식2", "당송팔대가문초소식3",
    "당송팔대가문초소식4", "당송팔대가문초소식5",
    "당송팔대가문초소철1", "당송팔대가문초소철2", "당송팔대가문초소철3",
    "당송팔대가문초왕안석1", "당송팔대가문초왕안석2",
    "당송팔대가문초증공1",
]

print("="*70)
print(f"🚀 43권 전체 PA+SA 배치 처리 시작")
print("="*70)
print(f"총 {len(books)}권 처리 예정")
print()

success_count = 0
error_count = 0

output_dir = Path("xlsx_pipeline_results")
output_dir.mkdir(exist_ok=True)

# 정답 데이터 위치(환경변수로 재지정 가능)
PA_GT_DIR = Path(os.environ.get("PA_GT_DIR", "accuracy/pa_gt"))
SA_GT_DIR = Path(os.environ.get("SA_GT_DIR", "accuracy/sa_gt"))

def resolve_gt_path(book: str, task: str) -> Path | None:
    """책별 정답 파일 경로를 탐색한다.

    우선순위:
    1) 환경변수 기반 디렉터리(PA_GT_DIR/SA_GT_DIR)/<book>.xlsx
    2) PA: xlsx/<book>/<book>_문장병렬.xlsx (PA 정답)
       SA: xlsx/<book>/<book>_구병렬.xlsx (SA 정답)
    """
    candidates = []
    if task == "pa":
        candidates.append(PA_GT_DIR / f"{book}.xlsx")
        candidates.append(Path("xlsx") / book / f"{book}_문장병렬.xlsx")
    else:  # sa
        candidates.append(SA_GT_DIR / f"{book}.xlsx")
        candidates.append(Path("xlsx") / book / f"{book}_구병렬.xlsx")
    for c in candidates:
        if c.exists():
            return c
    return None

for idx, book in enumerate(books, 1):
    folder_path = f"xlsx/{book}"
    
    # 문단병렬 파일 찾기
    para_parallel = None
    if Path(f"{folder_path}/{book}_문단병렬.xlsx").exists():
        para_parallel = f"{folder_path}/{book}_문단병렬.xlsx"
    
    if not para_parallel:
        print(f"[{idx:2d}/43] ⚠️  {book}: 문단병렬 파일 없음")
        error_count += 1
        continue
    
    print(f"[{idx:2d}/43] {book}")
    print(f"  입력: {para_parallel}")
    
    # 책별 결과 폴더 생성
    book_output_dir = output_dir / book
    book_output_dir.mkdir(exist_ok=True, parents=True)
    
    # PA 실행
    pa_output = book_output_dir / f"{book}_PA_문장병렬.xlsx"
    sa_output = book_output_dir / f"{book}_SA.xlsx"
    print(f"  PA 실행 중...")
    
    try:
        # 🚀 PA 성능 최적화 설정 (하드웨어 최대 활용)
        # - 배치 크기 256 (GPU VRAM 활용도 극대화)
        # - 워커 16 (8코어 CPU 최대 활용)
        # - Multi-Vector 임베딩 유지 (정확도 우선)
        # - 캐시 자동 활성화
        result = subprocess.run(
            ["python", "pa/main.py", para_parallel, str(pa_output), 
             "--embedder", "bge",
             "--max-workers", "16",
             "--batch-size", "256"],
            capture_output=True
        )
        
        if result.returncode == 0:
            print(f"  ✅ PA 완료")

            # PA 정확도 평가 (정답 파일이 있을 때만)
            pa_gt_file = resolve_gt_path(book, "pa")
            if pa_gt_file:
                pa_eval_output = book_output_dir / f"{book}_PA_eval_row.xlsx"
                print(f"  📊 PA 정확도 평가 중... (GT: {pa_gt_file})")
                try:
                    eval_cmd = [
                        "python", "accuracy/accuracy_evaluator.py",
                        str(pa_gt_file), str(pa_output),
                        "--project", "pa",
                        "--unit", "row",
                        "--ignore-space-punct",
                        "-o", str(pa_eval_output)
                    ]
                    eval_result = subprocess.run(
                        eval_cmd,
                        capture_output=True
                    )
                    if eval_result.returncode == 0:
                        print("  ✅ PA 정확도 평가 완료")
                    else:
                        print("  ⚠️  PA 정확도 평가 실패 (계속 진행)")
                        if eval_result.stderr:
                            print("    ", eval_result.stderr.decode(errors="ignore").strip())
                except Exception as e:
                    print(f"  ⚠️  PA 정확도 평가 오류 (계속 진행): {e}")
            else:
                print("  ℹ️  PA 정답 파일 없음 → 평가 건너뜀")
            
            # SA 실행 - **입력은 GT 문장병렬 파일을 사용!**
            print(f"  SA 실행 중...")
            
            # SA는 GT 문장병렬을 입력으로 받아야 함
            sa_input_file = Path(f"xlsx/{book}/{book}_문장병렬.xlsx")
            if not sa_input_file.exists():
                print(f"  ❌ SA 입력 파일 없음: {sa_input_file} (건너뜀)")
                error_count += 1
                continue
            
            # 🚀 SA 성능 최적화 설정 (하드웨어 최대 활용)
            # - 배치 크기 512 (GPU VRAM 활용도 극대화)
            # - 워커 16 (8코어 CPU 최대 활용)
            # - Multi-Vector 임베딩 유지 (정확도 우선)
            # - 캐시 자동 활성화 (5-10배 반복 텍스트 빠름)
            result = subprocess.run(
                ["python", "sa/main.py", str(sa_input_file), str(sa_output), 
                 "--embedder", "bge",
                 "--max-workers", "16",
                 "--chunk-size", "512"],
                capture_output=True
            )
            
            if result.returncode == 0:
                print(f"  ✅ SA 완료")

                # SA 정확도 평가 (정답 파일이 있을 때만)
                sa_gt_file = resolve_gt_path(book, "sa")
                if sa_gt_file:
                    sa_eval_output = book_output_dir / f"{book}_SA_eval_row.xlsx"
                    print(f"  📊 SA 정확도 평가 중... (GT: {sa_gt_file})")
                    try:
                        eval_cmd = [
                            "python", "accuracy/accuracy_evaluator.py",
                            str(sa_gt_file), str(sa_output),
                            "--project", "sa",
                            "--unit", "row",  # 예측 식별자(row_*) 호환 위해 row 평가
                            "--ignore-space-punct",
                            "--no-translation-match",
                            "-o", str(sa_eval_output)
                        ]
                        eval_result = subprocess.run(
                            eval_cmd,
                            capture_output=True
                        )
                        if eval_result.returncode == 0:
                            print("  ✅ SA 정확도 평가 완료")
                        else:
                            print("  ⚠️  SA 정확도 평가 실패 (계속 진행)")
                            if eval_result.stderr:
                                print("    ", eval_result.stderr.decode(errors="ignore").strip())
                    except Exception as e:
                        print(f"  ⚠️  SA 정확도 평가 오류 (계속 진행): {e}")
                else:
                    print("  ℹ️  SA 정답 파일 없음 → 평가 건너뜀")
                success_count += 1
            else:
                print(f"  ❌ SA 실패")
                error_count += 1
        else:
            print(f"  ❌ PA 실패")
            error_count += 1
    except Exception as e:
        print(f"  ❌ 오류: {e}")
        error_count += 1
    
    print()

print("="*70)
print(f"📊 배치 처리 완료")
print(f"  성공: {success_count}권")
print(f"  실패: {error_count}권")
print("="*70)

# 전체 평가 결과 집계
print()
print("="*70)
print("📈 평가 결과 집계 시작")
print("="*70)
try:
    result = subprocess.run(
        ["python", "analytics/aggregate_batch_results.py",
         "--results-dir", str(output_dir),
         "--output-dir", "analytics"],
        capture_output=True,
        timeout=300
    )
    if result.returncode == 0:
        print("✅ 평가 결과 집계 완료")
        print(result.stdout.decode(errors="ignore"))
    else:
        print("⚠️  평가 결과 집계 실패 (수동으로 실행 가능)")
        if result.stderr:
            print(result.stderr.decode(errors="ignore"))
except Exception as e:
    print(f"⚠️  평가 결과 집계 오류: {e}")
