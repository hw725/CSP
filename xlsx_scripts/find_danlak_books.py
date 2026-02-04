"""
43권 전체에서 원문 식별자가 동일한 (단락 ID 기반 추출 필요한) 책 찾기
"""

import xml.etree.ElementTree as ET
from pathlib import Path

def check_wonmun_identifiers(xml_path):
    """원문 태그의 식별자가 모두 동일한지 확인"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        wonmun_list = root.findall(".//원문")
        if not wonmun_list:
            return None, 0, None

        identifiers = set()
        for wonmun in wonmun_list:
            ident = wonmun.get("식별자", "")
            if ident:
                identifiers.add(ident)

        return len(identifiers), len(wonmun_list), identifiers
    except Exception as e:
        return None, 0, None

source_dir = Path("sources")
xml_files = sorted(source_dir.glob("*_원문_*.xml"))

print("=" * 80)
print("원문 식별자 동일 여부 확인 (단락 ID 기반 추출 필요 여부)")
print("=" * 80)

danlak_based_books = []

for xml_file in xml_files:
    book_name = xml_file.stem.split("_")[1]
    unique_count, total_count, identifiers = check_wonmun_identifiers(xml_file)

    if unique_count is not None:
        if unique_count == 1:
            # 원문 식별자가 모두 동일 -> 단락 ID 기반 추출 필요
            print(f"\n⚠️  {book_name}")
            print(f"    원문 태그: {total_count}개, 유니크 식별자: {unique_count}개")
            print(f"    식별자: {list(identifiers)}")
            danlak_based_books.append(book_name)
        elif unique_count < total_count * 0.5:
            # 유니크 식별자가 너무 적음 (의심스러운 케이스)
            print(f"\n🔍 {book_name} (확인 필요)")
            print(f"    원문 태그: {total_count}개, 유니크 식별자: {unique_count}개")

print("\n" + "=" * 80)
print(f"단락 ID 기반 추출 필요: {len(danlak_based_books)}권")
print("=" * 80)
for book in danlak_based_books:
    print(f"  - {book}")
