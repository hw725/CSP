#!/usr/bin/env python3
"""44권 전체 P2S+S2P 배치 처리"""

import subprocess
import os
from pathlib import Path
import time

# 44권 폴더 목록 (xlsx/* 하위 폴더명 기준)
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
    "당송팔대가문초구양수6", "당송팔대가문초소순1",
    "당송팔대가문초소식1", "당송팔대가문초소식2", "당송팔대가문초소식3",
    "당송팔대가문초소식4", "당송팔대가문초소식5",
    "당송팔대가문초소철1", "당송팔대가문초소철2", "당송팔대가문초소철3",
    "당송팔대가문초왕안석1", "당송팔대가문초왕안석2",
    "당송팔대가문초증공1",
]

print("="*70)
print(f"🚀 44권 전체 P2S+S2P 배치 처리 시작")
print("="*70)
print(f"총 {len(books)}권 처리 예정")
print()

success_count = 0
error_count = 0

output_dir = Path("xlsx_pipeline_results")
output_dir.mkdir(exist_ok=True)

# 정답 데이터 위치(환경변수로 재지정 가능)
SENTENCE_GT = Path(os.environ.get("SENTENCE_GT", "xlsx/f{book}/f{book}_문장병렬.xlsx"))
PHRASE_GT = Path(os.environ.get("PHRASE_GT", "xlsx/f{book}/f{book}_구병렬.xlsx"))

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
    
    print(f"[{idx:2d}/44] {book}")
    print(f"  입력: {para_parallel}")
    
    # 책별 결과 폴더 생성
    book_output_dir = output_dir / book
    book_output_dir.mkdir(exist_ok=True, parents=True)
    
    # PA 실행
    p2s_output = book_output_dir / f"{book}_P2S_문장병렬.xlsx"
    s2p_output = book_output_dir / f"{book}_S2P.xlsx"
    print(f"  P2S 실행 중...")
    
    try:
        # 🚀 P2S 성능 최적화 설정 (하드웨어 최대 활용)
        # - 배치 크기 256 (GPU VRAM 활용도 극대화)
        # - 워커 16 (8코어 CPU 최대 활용)
        # - Multi-Vector 임베딩 유지 (정확도 우선)
        # - 캐시 자동 활성화
        result = subprocess.run(
            ["python", "p2s/main.py", para_parallel, str(p2s_output), 
             "--embedder", "bge",
             "--max-workers", "16",
             "--batch-size", "256"],
            capture_output=True
        )
        
        if result.returncode == 0:
            print(f"  ✅ P2S 완료")

            # P2S 정확도 평가 (정답 파일이 있을 때만)
            p2s_gt_file = resolve_gt_path(book, "p2s")
            if p2s_gt_file:
                p2s_eval_output = book_output_dir / f"{book}_P2S_eval_row.xlsx"
                print(f"  📊 P2S 정확도 평가 중... (GT: {p2s_gt_file})")
                try:
                    eval_cmd = [
                        "python", "accuracy/accuracy_evaluator.py",
                        str(p2s_gt_file), str(p2s_output),
                        "--project", "p2s",
                        "--unit", "row",
                        "--ignore-space-punct",
                        "-o", str(p2s_eval_output)
                    ]
                    eval_result = subprocess.run(
                        eval_cmd,
                        capture_output=True
                    )
                    if eval_result.returncode == 0:
                        print("  ✅ P2S 정확도 평가 완료")
                    else:
                        print("  ⚠️  P2S 정확도 평가 실패 (계속 진행)")
                        if eval_result.stderr:
                            print("    ", eval_result.stderr.decode(errors="ignore").strip())
                except Exception as e:
                    print(f"  ⚠️  P2S 정확도 평가 오류 (계속 진행): {e}")
            else:
                print("  ℹ️  P2S 정답 파일 없음 → 평가 건너뜀")
            
            # S2P 실행 - **입력은 GT 문장병렬 파일을 사용!**
            print(f"  S2P 실행 중...")
            
            # S2P는 GT 문장병렬을 입력으로 받아야 함
            s2p_input_file = Path(f"xlsx/{book}/{book}_문장병렬.xlsx")
            if not s2p_input_file.exists():
                print(f"  ❌ S2P 입력 파일 없음: {s2p_input_file} (건너뜀)")
                error_count += 1
                continue
            
            # 🚀 S2P 성능 최적화 설정 (하드웨어 최대 활용)
            # - 배치 크기 512 (GPU VRAM 활용도 극대화)
            # - 워커 16 (8코어 CPU 최대 활용)
            # - Multi-Vector 임베딩 유지 (정확도 우선)
            # - 캐시 자동 활성화 (5-10배 반복 텍스트 빠름)
            result = subprocess.run(
                ["python", "s2p/main.py", str(s2p_input_file), str(s2p_output), 
                 "--embedder", "bge",
                 "--max-workers", "16",
                 "--chunk-size", "512"],
                capture_output=True
            )
            
            if result.returncode == 0:
                print(f"  ✅ S2P 완료")

                # S2P 정확도 평가 (정답 파일이 있을 때만)
                s2p_gt_file = resolve_gt_path(book, "s2p")
                if s2p_gt_file:
                    s2p_eval_output = book_output_dir / f"{book}_S2P_eval_row.xlsx"
                    print(f"  📊 S2P 정확도 평가 중... (GT: {s2p_gt_file})")
                    try:
                        eval_cmd = [
                            "python", "accuracy/accuracy_evaluator.py",
                            str(s2p_gt_file), str(s2p_output),
                            "--project", "s2p",
                            "--unit", "row",  # 예측 식별자(row_*) 호환 위해 row 평가
                            "--ignore-space-punct",
                            "--no-translation-match",
                            "-o", str(s2p_eval_output)
                        ]
                        eval_result = subprocess.run(
                            eval_cmd,
                            capture_output=True
                        )
                        if eval_result.returncode == 0:
                            print("  ✅ S2P 정확도 평가 완료")
                        else:
                            print("  ⚠️  S2P 정확도 평가 실패 (계속 진행)")
                            if eval_result.stderr:
                                print("    ", eval_result.stderr.decode(errors="ignore").strip())
                    except Exception as e:
                        print(f"  ⚠️  S2P 정확도 평가 오류 (계속 진행): {e}")
                else:
                    print("  ℹ️  S2P 정답 파일 없음 → 평가 건너뜀")
                success_count += 1
            else:
                print(f"  ❌ S2P 실패")
                error_count += 1
        else:
            print(f"  ❌ P2S 실패")
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
