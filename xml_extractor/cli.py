#!/usr/bin/env python3
"""
XML Extractor CLI — XML 원문/번역문에서 문단·문장·구 단위 XLSX 추출

사용법:
  # 단일 쌍 추출 (문단+문장+구 전체)
  python -m xml_extractor.cli extract \\
      --original sources/원문.xml --translation sources/번역문.xml

  # 특정 단위만 추출
  python -m xml_extractor.cli extract \\
      --original sources/원문.xml --translation sources/번역문.xml \\
      --levels sentence phrase

  # 디렉토리 일괄 추출
  python -m xml_extractor.cli batch --xml-dir sources

  # 책 목록 조회
  python -m xml_extractor.cli list --xml-dir sources

  # 서지정보 추출 (XML 출력)
  python -m xml_extractor.cli biblio --xml-dir sources --format xml

  # 서지정보 추출 (XLSX 출력)
  python -m xml_extractor.cli biblio --xml-dir sources --format xlsx
"""

import argparse
import sys
from pathlib import Path

# 패키지 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from xml_extractor.xml_processor import (
    XMLProcessor,
    XMLPair,
    create_xml_pair_from_directory,
)
from xml_extractor.xml_biblio_extractor import (
    extract_biblio_from_directory,
    export_biblio_xml,
    export_biblio_xlsx,
)


def extract_single(args):
    """단일 XML 쌍에서 추출"""
    orig = args.original
    trans = args.translation
    output_dir = Path(args.output)
    levels = args.levels or ["paragraph", "sentence", "phrase"]

    if not Path(orig).exists():
        print(f"[ERROR] 원문 파일 없음: {orig}")
        sys.exit(1)
    if not Path(trans).exists():
        print(f"[ERROR] 번역문 파일 없음: {trans}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # pair_id 생성
    pair_id = args.pair_id
    if not pair_id:
        name = Path(orig).stem
        pair_id = name.replace('_원문', '').replace('-원문', '').replace('원문', '')
        if not pair_id:
            pair_id = name

    print(f"=== XML Extractor: {pair_id} ===")
    print(f"  원문: {orig}")
    print(f"  번역문: {trans}")
    print(f"  출력: {output_dir}")
    print(f"  단위: {', '.join(levels)}")
    print()

    results = {}

    if "paragraph" in levels:
        df = XMLProcessor.extract_paragraph_data(orig, trans)
        out_file = output_dir / "paragraph_parallel.xlsx"
        df.to_excel(out_file, index=False)
        results["paragraph"] = len(df)
        print(f"  -> {out_file} ({len(df)}행)")

    if "sentence" in levels:
        df = XMLProcessor.extract_sentence_data(orig, trans)
        out_file = output_dir / "sentence_parallel.xlsx"
        df.to_excel(out_file, index=False)
        results["sentence"] = len(df)
        print(f"  -> {out_file} ({len(df)}행)")

    if "phrase" in levels:
        df = XMLProcessor.extract_phrase_data(orig, trans)
        out_file = output_dir / "phrase_parallel.xlsx"
        df.to_excel(out_file, index=False)
        results["phrase"] = len(df)
        print(f"  -> {out_file} ({len(df)}행)")

    print()
    print(f"=== 완료: {', '.join(f'{k} {v}행' for k, v in results.items())} ===")


def batch_extract(args):
    """디렉토리 일괄 추출"""
    xml_dir = args.xml_dir
    output_root = Path(args.output)
    levels = args.levels or ["paragraph", "sentence", "phrase"]

    pairs = create_xml_pair_from_directory(xml_dir)

    if not pairs:
        print(f"[ERROR] XML 쌍을 찾을 수 없습니다: {xml_dir}")
        sys.exit(1)

    print(f"=== 일괄 추출: {len(pairs)}개 쌍 ===")
    for pair in pairs:
        print(f"  - {pair.pair_id}")
    print()

    for i, pair in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {pair.pair_id}")
        out_dir = output_root / pair.pair_id
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            if "paragraph" in levels:
                df = XMLProcessor.extract_paragraph_data(pair.original_path, pair.translation_path)
                df.to_excel(out_dir / "paragraph_parallel.xlsx", index=False)

            if "sentence" in levels:
                df = XMLProcessor.extract_sentence_data(pair.original_path, pair.translation_path)
                df.to_excel(out_dir / "sentence_parallel.xlsx", index=False)

            if "phrase" in levels:
                df = XMLProcessor.extract_phrase_data(pair.original_path, pair.translation_path)
                df.to_excel(out_dir / "phrase_parallel.xlsx", index=False)

            print(f"   -> {out_dir}")
        except Exception as e:
            print(f"   [ERROR] {e}")

    print(f"\n=== 일괄 추출 완료: {len(pairs)}개 ===")


def list_pairs(args):
    """XML 쌍 목록 조회"""
    xml_dir = args.xml_dir
    pairs = create_xml_pair_from_directory(xml_dir)

    if not pairs:
        print(f"XML 쌍 없음: {xml_dir}")
        return

    print(f"=== XML 쌍 목록 ({xml_dir}) ===")
    for i, pair in enumerate(pairs, 1):
        print(f"  {i:3d}. {pair.pair_id}")
        print(f"       원문: {Path(pair.original_path).name}")
        print(f"       번역: {Path(pair.translation_path).name}")
    print(f"\n총 {len(pairs)}개")


def extract_biblio(args):
    """서지정보 추출"""
    xml_dir = args.xml_dir
    output_dir = Path(args.output)
    fmt = args.format

    biblios = extract_biblio_from_directory(xml_dir)

    if not biblios:
        print(f"[ERROR] XML 파일을 찾을 수 없습니다: {xml_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== 서지정보 추출: {len(biblios)}개 ===")

    if fmt in ("xml", "both"):
        xml_path = output_dir / "biblio_metadata.xml"
        export_biblio_xml(biblios, str(xml_path))
        print(f"  -> {xml_path}")

    if fmt in ("xlsx", "both"):
        xlsx_path = output_dir / "biblio_metadata.xlsx"
        export_biblio_xlsx(biblios, str(xlsx_path))
        print(f"  -> {xlsx_path}")

    # 개별 항목 요약
    print()
    for i, b in enumerate(biblios, 1):
        print(f"  {i:3d}. [{b.jti_code}] {b.대표서명한글} ({b.대표서명})")

    print(f"\n=== 완료: {len(biblios)}개 서지정보 추출 ===")


def main():
    parser = argparse.ArgumentParser(
        description="XML Extractor — XML 원문/번역문에서 병렬 XLSX 추출",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="명령")

    # extract: 단일 쌍 추출
    p_extract = subparsers.add_parser("extract", help="단일 XML 쌍 추출")
    p_extract.add_argument("--original", "-o", required=True, help="원문 XML 파일")
    p_extract.add_argument("--translation", "-t", required=True, help="번역문 XML 파일")
    p_extract.add_argument("--output", "-O", default="xml_extractor_results", help="출력 디렉토리")
    p_extract.add_argument("--pair-id", help="쌍 ID (미지정 시 파일명에서 자동 생성)")
    p_extract.add_argument(
        "--levels", nargs="+",
        choices=["paragraph", "sentence", "phrase"],
        help="추출 단위 (기본: 전체)",
    )

    # batch: 디렉토리 일괄 추출
    p_batch = subparsers.add_parser("batch", help="디렉토리 일괄 추출")
    p_batch.add_argument("--xml-dir", default="sources", help="XML 디렉토리")
    p_batch.add_argument("--output", "-O", default="xml_extractor_results", help="출력 디렉토리")
    p_batch.add_argument(
        "--levels", nargs="+",
        choices=["paragraph", "sentence", "phrase"],
        help="추출 단위 (기본: 전체)",
    )

    # list: XML 쌍 목록
    p_list = subparsers.add_parser("list", help="XML 쌍 목록 조회")
    p_list.add_argument("--xml-dir", default="sources", help="XML 디렉토리")

    # biblio: 서지정보 추출
    p_biblio = subparsers.add_parser("biblio", help="서지정보 추출")
    p_biblio.add_argument("--xml-dir", default="sources", help="XML 디렉토리")
    p_biblio.add_argument("--output", "-O", default="xml_extractor_results", help="출력 디렉토리")
    p_biblio.add_argument(
        "--format", "-f", default="xml",
        choices=["xml", "xlsx", "both"],
        help="출력 형식 (기본: xml)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "extract":
        extract_single(args)
    elif args.command == "batch":
        batch_extract(args)
    elif args.command == "list":
        list_pairs(args)
    elif args.command == "biblio":
        extract_biblio(args)


if __name__ == "__main__":
    main()
