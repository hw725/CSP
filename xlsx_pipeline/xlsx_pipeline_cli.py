#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XLSX 파이프라인 CLI 도구
Excel 파일(구병렬, 문장병렬, 문단병렬) 기반 데이터 처리 및 분석
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging

# Python 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# XLSX 파이프라인 모듈 import
try:
    from xlsx_pipeline.xlsx_pipeline_processor import XLSXPipelineProcessor, XLSXBook
except ImportError:
    from xlsx_pipeline_processor import XLSXPipelineProcessor, XLSXBook

class XLSXPipelineManager:
    """XLSX 파이프라인 관리자"""

    def __init__(
        self, xlsx_root_dir: str = "xlsx", output_dir: str = "xlsx_pipeline_results"
    ):
        self.processor = XLSXPipelineProcessor(xlsx_root_dir, output_dir)
        self.output_dir = Path(output_dir)
        # 초기화 시 자동으로 책 발견
        self.processor.discover_books()

    def list_books(self):
        """모든 책 목록 출력"""
        print("\n📚 책 목록")
        print("=" * 80)

        books = self.processor.list_books()

        if not books:
            print("❌ 발견된 책이 없습니다.")
            return

        for i, book_info in enumerate(books, 1):
            book_id = book_info["book_id"]
            exists = book_info["exists"]

            status = []
            if exists.get("word"):
                status.append("구병렬")
            if exists.get("sentence"):
                status.append("문장병렬")
            if exists.get("paragraph"):
                status.append("문단병렬")

            print(f"{i:3d}. {book_id}")
            print(f"     파일: {', '.join(status)}")

        print(f"\n총 {len(books)}개 책")

    def show_statistics(self, book_id: Optional[str] = None):
        """통계 정보 출력"""
        if book_id:
            # 특정 책 통계
            book = self.processor.get_book(book_id)
            if not book:
                print(f"❌ 책을 찾을 수 없습니다: {book_id}")
                return

            print(f"\n📊 책 통계: {book_id}")
            print("=" * 80)

            stats = book.get_statistics()
            print(json.dumps(stats, ensure_ascii=False, indent=2))
        else:
            # 전체 통계
            print("\n📊 전체 통계")
            print("=" * 80)

            all_stats = self.processor.get_all_statistics()

            # 요약 정보
            total_books = all_stats["total_books"]
            print(f"총 책 수: {total_books}")

            # 파일별 존재 여부 카운트
            word_count = sum(
                1
                for b in all_stats["books"].values()
                if b.get("files_exist", {}).get("word", False)
            )
            sent_count = sum(
                1
                for b in all_stats["books"].values()
                if b.get("files_exist", {}).get("sentence", False)
            )
            para_count = sum(
                1
                for b in all_stats["books"].values()
                if b.get("files_exist", {}).get("paragraph", False)
            )

            print(f"\n파일 유형별:")
            print(f"  - 구병렬: {word_count}개")
            print(f"  - 문장병렬: {sent_count}개")
            print(f"  - 문단병렬: {para_count}개")

            # 총 행 수
            total_words = sum(
                b.get("word_count", 0) for b in all_stats["books"].values()
            )
            total_sents = sum(
                b.get("sentence_count", 0) for b in all_stats["books"].values()
            )
            total_paras = sum(
                b.get("paragraph_count", 0) for b in all_stats["books"].values()
            )

            print(f"\n총 데이터량:")
            print(f"  - 구병렬: {total_words:,}행")
            print(f"  - 문장병렬: {total_sents:,}행")
            print(f"  - 문단병렬: {total_paras:,}행")

            # JSON 파일 저장
            output_file = self.processor.save_statistics()
            print(f"\n💾 상세 통계 저장: {output_file}")

    def process_single_book(
        self,
        book_id: str,
        levels: List[str] = None,
        analysis: List[str] = None,
        project: str = None,
    ):
        """단일 책 처리"""
        print(f"\n🚀 책 처리 시작: {book_id}")
        print("=" * 80)

        config = {}
        if levels:
            config["levels"] = levels
        if analysis:
            config["analysis"] = analysis
        if project:
            config["project"] = project

        try:
            result = self.processor.process_book_pipeline(book_id, config)

            print("\n✅ 처리 완료")
            print(json.dumps(result, ensure_ascii=False, indent=2))

            # 결과 저장
            output_file = (
                self.output_dir
                / f"{book_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"\n💾 결과 저장: {output_file}")

        except Exception as e:
            print(f"\n❌ 처리 실패: {e}")
            import traceback

            traceback.print_exc()

    def batch_process(
        self,
        book_ids: List[str] = None,
        levels: List[str] = None,
        analysis: List[str] = None,
        project: str = None,
    ):
        """배치 처리"""
        if book_ids:
            print(f"\n🚀 배치 처리 시작: {len(book_ids)}개 책")
        else:
            print("\n🚀 전체 배치 처리 시작")
        print("=" * 80)

        config = {}
        if levels:
            config["levels"] = levels
        if analysis:
            config["analysis"] = analysis
        if project:
            config["project"] = project

        try:
            results = self.processor.batch_process(book_ids, config)

            success_count = sum(
                1 for r in results["results"].values() if "error" not in r
            )
            fail_count = len(results["results"]) - success_count

            print(f"\n✅ 배치 처리 완료")
            print(f"   - 성공: {success_count}개")
            print(f"   - 실패: {fail_count}개")

            # 결과 저장
            output_file = (
                self.output_dir
                / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(f"\n💾 결과 저장: {output_file}")

        except Exception as e:
            print(f"\n❌ 배치 처리 실패: {e}")
            import traceback

            traceback.print_exc()

    def discover(self):
        """책 자동 발견"""
        print("\n🔍 책 자동 발견 중...")
        print("=" * 80)

        discovered = self.processor.discover_books()

        if discovered:
            print(f"\n✅ {len(discovered)}개 책 발견됨:")
            for book_id in discovered:
                print(f"  - {book_id}")
        else:
            print("\n❌ 발견된 책이 없습니다.")

def main():
    """CLI 메인 함수"""
    parser = argparse.ArgumentParser(
        description="XLSX 파이프라인 CLI - Excel 파일 기반 데이터 처리",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 책 목록 조회
  python xlsx_pipeline_cli.py list

  # 책 자동 발견
  python xlsx_pipeline_cli.py discover

  # 전체 통계
  python xlsx_pipeline_cli.py stats

  # 특정 책 통계
  python xlsx_pipeline_cli.py stats --book 당송팔대가문초구양수1

  # 단일 책 처리
  python xlsx_pipeline_cli.py process --book 당송팔대가문초구양수1

  # 특정 레벨만 처리
  python xlsx_pipeline_cli.py process --book 당송팔대가문초구양수1 --levels word sentence

  # 배치 처리 (전체)
  python xlsx_pipeline_cli.py batch

  # 배치 처리 (특정 책들)
  python xlsx_pipeline_cli.py batch --books 당송팔대가문초구양수1 당송팔대가문초구양수2
        """,
    )

    parser.add_argument(
        "command",
        choices=["list", "discover", "stats", "process", "batch"],
        help="실행할 명령",
    )

    parser.add_argument(
        "--xlsx-root", default="xlsx", help="Excel 파일 루트 디렉토리 (기본값: xlsx)"
    )

    parser.add_argument(
        "--output",
        default="xlsx_pipeline_results",
        help="결과 출력 디렉토리 (기본값: xlsx_pipeline_results)",
    )

    parser.add_argument("--book", help="처리할 책 ID (process 명령용)")

    parser.add_argument("--books", nargs="+", help="처리할 책 ID 리스트 (batch 명령용)")

    parser.add_argument(
        "--levels",
        nargs="+",
        choices=["word", "sentence", "paragraph"],
        help="처리할 레벨 (기본값: 전체)",
    )

    parser.add_argument(
        "--analysis",
        nargs="+",
        default=["statistics"],
        help="수행할 분석 (기본값: statistics) - full: PA+SA+Accuracy 전체 실행 (쉼표/공백 구분 모두 지원)",
    )

    parser.add_argument("--project", help="프로젝트 이름 (정확도 평가용 임계값 설정)")

    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로그 출력")

    args = parser.parse_args()

    # 분석 옵션 쉼표/공백 혼합 입력 허용 및 검증
    valid_analysis = {
        "statistics",
        "similarity",
        "quality",
        "pa",
        "sa",
        "accuracy",
        "full",
    }
    normalized_analysis = []
    for item in args.analysis or []:
        for part in str(item).split(","):
            p = part.strip()
            if not p:
                continue
            if p not in valid_analysis:
                parser.error(
                    f"--analysis 잘못된 값: {p} (허용: {sorted(valid_analysis)})"
                )
            normalized_analysis.append(p)
    if not normalized_analysis:
        normalized_analysis = ["statistics"]
    args.analysis = normalized_analysis

    # 로깅 설정
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 매니저 초기화
    manager = XLSXPipelineManager(args.xlsx_root, args.output)

    # 명령 실행
    if args.command == "list":
        manager.list_books()

    elif args.command == "discover":
        manager.discover()

    elif args.command == "stats":
        manager.show_statistics(args.book)

    elif args.command == "process":
        if not args.book:
            print("❌ --book 옵션이 필요합니다.")
            sys.exit(1)
        manager.process_single_book(args.book, args.levels, args.analysis, args.project)

    elif args.command == "batch":
        manager.batch_process(args.books, args.levels, args.analysis, args.project)

if __name__ == "__main__":
    main()
