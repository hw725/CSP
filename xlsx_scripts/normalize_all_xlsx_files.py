#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xlsx 폴더 전체 정규화 스크립트
모든 XLSX 파일의 원문/번역문을 정규화하고 원본 덮어쓰기
기존 파일은 .bak으로 백업
"""

import sys
import os
import shutil
from pathlib import Path
import pandas as pd

# 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from common.text_normalizer import normalize_source_and_target

def normalize_xlsx_file(file_path: str, backup: bool = True) -> dict:
    """
    단일 XLSX 파일 정규화 + 백업

    Args:
        file_path: XLSX 파일 경로
        backup: True이면 기존 파일을 .bak으로 백업

    Returns:
        {'success': bool, 'message': str, 'normalized_rows': int, 'backup_path': str}
    """
    try:
        file_path_obj = Path(file_path)
        backup_path = None

        # 백업 생성
        if backup and file_path_obj.exists():
            backup_path = f"{file_path}.bak"
            shutil.copy2(file_path, backup_path)

        df = pd.read_excel(file_path)

        # 원문/번역문 컬럼 찾기
        src_col = None
        tgt_col = None

        for col in df.columns:
            if "원문" in col:
                src_col = col
            elif "번역문" in col:
                tgt_col = col

        if not src_col or not tgt_col:
            return {
                "success": False,
                "message": f"❌ 원문/번역문 컬럼 없음 (컬럼: {list(df.columns)})",
                "normalized_rows": 0,
                "backup_path": backup_path,
            }

        # 정규화 적용
        normalized_count = 0
        for idx in df.index:
            src = str(df.at[idx, src_col]) if pd.notna(df.at[idx, src_col]) else ""
            tgt = str(df.at[idx, tgt_col]) if pd.notna(df.at[idx, tgt_col]) else ""

            # 내부 개행 제거, 공백만 정규화 ([-텍스트] 보존!)
            norm_src, norm_tgt = normalize_source_and_target(
                src, tgt, normalize_brackets_in_tgt=False
            )

            # 변화가 있으면 카운트
            if norm_src != src or norm_tgt != tgt:
                normalized_count += 1

            df.at[idx, src_col] = norm_src
            df.at[idx, tgt_col] = norm_tgt

        # 파일 저장 (원본 덮어쓰기)
        df.to_excel(file_path, index=False, engine="openpyxl")

        return {
            "success": True,
            "message": f"✅ 정규화 완료 ({normalized_count}/{len(df)}행 변경)",
            "normalized_rows": normalized_count,
            "backup_path": backup_path,
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"❌ 처리 실패: {e}",
            "normalized_rows": 0,
            "backup_path": None,
        }

def normalize_all_xlsx_files(root_dir: str = "xlsx", backup: bool = True):
    """
    xlsx 폴더 전체 정규화

    Args:
        root_dir: xlsx 폴더 경로
        backup: True이면 기존 파일을 .bak으로 백업
    """
    root_path = Path(root_dir).resolve()  # 절대 경로로 변환

    if not root_path.exists():
        print(f"❌ 폴더 없음: {root_dir}")
        return

    print(f"📂 xlsx 폴더 정규화 시작: {root_path.absolute()}")
    print(f"💾 백업 생성: {backup}")
    print()

    # 모든 XLSX 파일 스캔 (.bak 제외)
    xlsx_files = sorted(
        [f for f in root_path.rglob("*.xlsx") if not str(f).endswith(".bak")]
    )

    if not xlsx_files:
        print(f"❌ XLSX 파일 없음")
        return

    print(f"📊 발견된 XLSX 파일: {len(xlsx_files)}개\n")

    total_normalized = 0
    success_count = 0
    error_count = 0
    backup_files = []

    for file_path in xlsx_files:
        # 상대 경로로 표시
        file_path_abs = file_path.resolve()  # 절대 경로로 확보
        try:
            rel_path = file_path_abs.relative_to(project_root)
        except ValueError:
            rel_path = file_path_abs  # 상대 경로 변환 실패 시 절대 경로 사용

        print(f"🔄 처리 중: {rel_path}")

        result = normalize_xlsx_file(str(file_path_abs), backup=backup)

        if result["success"]:
            print(f"   {result['message']}")
            if result["backup_path"]:
                print(f"   💾 백업: {Path(result['backup_path']).name}")
                backup_files.append(result["backup_path"])
            success_count += 1
            total_normalized += result["normalized_rows"]
        else:
            print(f"   {result['message']}")
            error_count += 1

        print()

    # 요약
    print("=" * 70)
    print(f"📊 정규화 완료 요약")
    print(f"  • 총 파일: {len(xlsx_files)}개")
    print(f"  • 성공: {success_count}개")
    print(f"  • 실패: {error_count}개")
    print(f"  • 총 정규화 행: {total_normalized}개")
    print(f"  • 백업 파일: {len(backup_files)}개")
    if backup_files:
        print(f"  • 백업 위치: 각 파일명.xlsx.bak")
    print("=" * 70)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="xlsx 폴더 전체 정규화 (기존 파일 .bak 백업)"
    )
    parser.add_argument(
        "--root-dir", type=str, default="xlsx", help="xlsx 폴더 경로 (기본: xlsx)"
    )
    parser.add_argument("--no-backup", action="store_true", help="백업 생성 안 함")

    args = parser.parse_args()

    normalize_all_xlsx_files(args.root_dir, backup=not args.no_backup)
