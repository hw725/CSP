#!/usr/bin/env python3
"""
CSP (Classical Korean Parallel Processing) 범용 메인 실행 파일

지원 파일 형식:
- XML: 고전문헌 XML 파일 (원문/번역문 쌍)
- TXT: 일반 텍스트 파일
- XLSX: 엑셀 데이터 파일
- CSV: CSV 데이터 파일

사용법:
    python main.py --help
    python main.py xml process 원문.xml 번역.xml  # XML 처리
    python main.py xml smart /workspace/2025      # XML 스마트 처리
    python main.py txt process file.txt           # TXT 처리 (미래 지원)
    python main.py xlsx process file.xlsx         # XLSX 처리 (미래 지원)
    python main.py auto file1 file2               # 자동 형식 감지
"""

import sys
import argparse
from pathlib import Path

# 패키지 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

def detect_file_type(filepath):
    """파일 확장자를 기반으로 파일 타입 감지"""
    if not filepath:
        return None

    path = Path(filepath)
    if not path.exists():
        return None

    ext = path.suffix.lower()

    if ext in [".xml"]:
        return "xml"
    elif ext in [".txt"]:
        return "txt"
    elif ext in [".xlsx", ".xls"]:
        return "xlsx"
    elif ext in [".csv"]:
        return "csv"
    else:
        return "unknown"

def analyze_xml_structure(filepath):
    """XML 파일 구조 분석: 통합형 vs 분리형"""
    try:
        import xml.etree.ElementTree as ET

        path = Path(filepath)
        if not path.exists():
            return "unknown"

        tree = ET.parse(filepath)
        root = tree.getroot()

        # 통합형: 원문과 번역문이 같은 파일에 있는 경우
        has_original = any(elem.tag in ["원문", "original"] for elem in root.iter())
        has_translation = any(
            elem.tag in ["번역문", "translation"] for elem in root.iter()
        )

        if has_original and has_translation:
            return "merged"  # 통합 XML (원문+번역문)
        else:
            return "separate"  # 분리 XML (원문만 or 번역문만)

    except Exception as e:
        print(f"⚠️ XML 구조 분석 실패: {e}")
        return "unknown"

def find_xml_pair(filepath):
    """XML 파일의 쌍(원문/번역문) 찾기"""
    path = Path(filepath)
    parent_dir = path.parent
    filename = path.stem

    # 원문/번역문 패턴 감지 및 쌍 찾기
    if "원문" in filename:
        # 원문 파일이면 번역문 찾기
        translation_name = filename.replace("원문", "번역문")
        for ext in [".xml", ".txt"]:
            pair_path = parent_dir / (translation_name + ext)
            if pair_path.exists():
                return str(pair_path)
    elif "번역문" in filename:
        # 번역문 파일이면 원문 찾기
        original_name = filename.replace("번역문", "원문")
        for ext in [".xml", ".txt"]:
            pair_path = parent_dir / (original_name + ext)
            if pair_path.exists():
                return str(pair_path)
    elif "original" in filename.lower():
        # 영문 패턴
        translation_name = filename.lower().replace("original", "translation")
        for ext in [".xml", ".txt"]:
            pair_path = parent_dir / (translation_name + ext)
            if pair_path.exists():
                return str(pair_path)
    elif "translation" in filename.lower():
        # 영문 패턴
        original_name = filename.lower().replace("translation", "original")
        for ext in [".xml", ".txt"]:
            pair_path = parent_dir / (original_name + ext)
            if pair_path.exists():
                return str(pair_path)

    return None

def process_merged_xml(files):
    """통합 XML 파일 처리"""
    print("🔄 통합 XML 파일 처리를 시작합니다...")

    for filepath in files:
        print(f"📄 처리 중: {Path(filepath).name}")

        try:
            # 향후 구현: 통합 XML을 분리하여 기존 파이프라인에 연결
            print("   ✅ 구조 분석 완료 (통합 XML)")
            print("   🚧 상세 처리는 향후 구현 예정")

            # 미리보기: XML 내부 구조 출력
            import xml.etree.ElementTree as ET

            tree = ET.parse(filepath)
            root = tree.getroot()

            original_count = len(
                [elem for elem in root.iter() if elem.tag in ["원문", "original"]]
            )
            translation_count = len(
                [elem for elem in root.iter() if elem.tag in ["번역문", "translation"]]
            )

            print(f"   📊 원문 섹션: {original_count}개")
            print(f"   📊 번역문 섹션: {translation_count}개")

        except Exception as e:
            print(f"   ❌ 처리 실패: {e}")

    print("\n💡 통합 XML 처리 기능은 현재 개발 중입니다.")
    print("   기존 XML 파이프라인과 연동하여 완전한 처리를 제공할 예정입니다.")

def main():
    parser = argparse.ArgumentParser(
        description="CSP (Classical Korean Parallel Processing) 범용 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 분리 XML 파일 처리 (원문.xml + 번역문.xml)
  python main.py xml smart /workspace/2025
  python main.py xml process 원문.xml 번역.xml

  # 통합 XML 파일 처리 (원문+번역문이 한 파일에)
  python main.py xml-merged process 01병렬변환.xml

  # 자동 형식 감지
  python main.py auto file1.xml file2.xml
  python main.py auto 01병렬변환.xml

  # 특정 형식 지정
  python main.py txt process file.txt
  python main.py xlsx process file.xlsx
        """,
    )

    subparsers = parser.add_subparsers(dest="format", help="파일 형식 또는 모드")

    # XML 서브커맨드
    xml_parser = subparsers.add_parser(
        "xml", help="XML 파일 처리 (분리된 원문/번역문 쌍)"
    )
    xml_subparsers = xml_parser.add_subparsers(dest="xml_command", help="XML 명령어")

    # XML process 서브커맨드
    xml_process_parser = xml_subparsers.add_parser("process", help="XML 파일 쌍 처리")
    xml_process_parser.add_argument(
        "--original", required=True, help="원문 XML 파일 경로"
    )
    xml_process_parser.add_argument(
        "--translation", required=True, help="번역문 XML 파일 경로"
    )
    xml_process_parser.add_argument("--pair-id", help="파일 쌍 ID")
    xml_process_parser.add_argument("--output-dir", help="출력 디렉토리")

    # XML smart 서브커맨드
    xml_smart_parser = xml_subparsers.add_parser("smart", help="스마트 XML 처리")
    xml_smart_parser.add_argument("--xml-dir", required=True, help="XML 파일 디렉토리")
    xml_smart_parser.add_argument("--output-dir", help="출력 디렉토리")

    # XML 통합 서브커맨드 (원문+번역문이 한 파일에 있는 경우)
    xml_merged_parser = subparsers.add_parser(
        "xml-merged", help="통합 XML 파일 처리 (원문+번역문 통합)"
    )
    xml_merged_parser.add_argument(
        "merged_command", help="통합 XML 명령어 (process, split, analyze)"
    )
    xml_merged_parser.add_argument(
        "merged_args", nargs="*", help="통합 XML 명령어 인수들"
    )

    # TXT 서브커맨드 (미래 확장)
    txt_parser = subparsers.add_parser("txt", help="TXT 파일 처리 (미래 지원)")
    txt_parser.add_argument("txt_args", nargs="*", help="TXT 처리 인수들")

    # XLSX 서브커맨드 (미래 확장)
    xlsx_parser = subparsers.add_parser("xlsx", help="XLSX 파일 처리 (미래 지원)")
    xlsx_parser.add_argument("xlsx_args", nargs="*", help="XLSX 처리 인수들")

    # 자동 감지 서브커맨드
    auto_parser = subparsers.add_parser("auto", help="파일 형식 자동 감지")
    auto_parser.add_argument("files", nargs="+", help="처리할 파일들")

    args = parser.parse_args()

    if args.format == "xml":
        # XML 파이프라인으로 전달
        from xml_pipeline.xml_pipeline_cli import main as xml_main

        # sys.argv를 XML CLI 형식으로 재구성
        original_argv = sys.argv[:]  # 백업

        if args.xml_command == "process":
            sys.argv = [
                "xml_pipeline_cli.py",
                "process",
                "--original",
                args.original,
                "--translation",
                args.translation,
            ]
            if hasattr(args, "pair_id") and args.pair_id:
                sys.argv.extend(["--pair-id", args.pair_id])
            if hasattr(args, "output_dir") and args.output_dir:
                sys.argv.extend(["--output-dir", args.output_dir])
        elif args.xml_command == "smart":
            sys.argv = ["xml_pipeline_cli.py", "smart", "--xml-dir", args.xml_dir]
            if hasattr(args, "output_dir") and args.output_dir:
                sys.argv.extend(["--output-dir", args.output_dir])
        else:
            print(f"❌ 지원하지 않는 XML 명령: {args.xml_command}")
            return

        try:
            xml_main()
        finally:
            sys.argv = original_argv  # 복원

    elif args.format == "xml-merged":
        # 통합 XML 처리
        print("🔄 통합 XML 파이프라인으로 처리합니다.")
        if args.merged_command == "process" and args.merged_args:
            process_merged_xml(args.merged_args)
        else:
            print("사용법: python main.py xml-merged process <XML파일>")

    elif args.format == "auto":
        # 자동 형식 감지
        if not args.files:
            print("❌ 처리할 파일을 지정해주세요.")
            return

        file_types = [detect_file_type(f) for f in args.files]
        unique_types = set(file_types)

        print(f"🔍 감지된 파일 형식: {dict(zip(args.files, file_types))}")

        if "xml" in unique_types and len(unique_types) == 1:
            # XML 파일들 분석
            xml_files = [f for f in args.files if detect_file_type(f) == "xml"]

            if len(xml_files) == 1:
                # 단일 XML 파일인 경우
                first_xml = xml_files[0]
                xml_type = analyze_xml_structure(first_xml)

                if xml_type == "merged":
                    print(
                        "🎯 통합 XML 파이프라인으로 처리합니다 (원문+번역문 통합 파일)."
                    )
                    process_merged_xml([first_xml])
                else:
                    # 분리형이면 쌍을 찾아보기
                    pair_file = find_xml_pair(first_xml)
                    if pair_file:
                        print(f"🎯 XML 쌍을 찾았습니다:")
                        print(f"   원본: {Path(first_xml).name}")
                        print(f"   쌍: {Path(pair_file).name}")
                        print("🎯 분리 XML 파이프라인으로 처리합니다.")

                        from xml_pipeline.xml_pipeline_cli import main as xml_main

                        sys.argv = [
                            "xml_pipeline_cli.py",
                            "process",
                            "--original",
                            first_xml,
                            "--translation",
                            pair_file,
                        ]
                        xml_main()
                    else:
                        print(
                            "🎯 분리 XML 스마트 모드로 처리합니다 (디렉토리 전체 스캔)."
                        )
                        from xml_pipeline.xml_pipeline_cli import main as xml_main

                        sys.argv = [
                            "xml_pipeline_cli.py",
                            "smart",
                            "--xml-dir",
                            str(Path(first_xml).parent),
                        ]
                        xml_main()
            else:
                # 여러 XML 파일인 경우
                print("🎯 분리 XML 파이프라인으로 처리합니다 (여러 파일).")
                from xml_pipeline.xml_pipeline_cli import main as xml_main

                sys.argv = [
                    "xml_pipeline_cli.py",
                    "smart",
                    "--xml-dir",
                    str(Path(xml_files[0]).parent),
                ]
                xml_main()
        else:
            print("❌ 혼합된 파일 형식이거나 지원하지 않는 형식입니다.")
            print("   현재는 XML 파일만 지원합니다.")

    elif args.format == "txt":
        print("📝 TXT 파일 처리는 아직 구현되지 않았습니다.")
        print("   향후 업데이트에서 지원 예정입니다.")

    elif args.format == "xlsx":
        print("📊 XLSX 파일 처리는 아직 구현되지 않았습니다.")
        print("   향후 업데이트에서 지원 예정입니다.")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
