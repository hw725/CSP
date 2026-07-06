"""
단락 ID 기반 추출 스크립트
원문 식별자가 모두 동일한 책들 (예기집설대전1, 당시삼백수1~3) 전용
단락 태그의 id 속성을 문단식별자로 사용
"""

import xml.etree.ElementTree as ET
import os
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd


def extract_book_name(filename):
    """파일명에서 책 이름 추출"""
    match = re.search(r'\[역주\](.+?)_(?:원문|번역문)', filename)
    if match:
        return match.group(1)
    return None


def extract_text_from_s(s_elem):
    """s 태그에서 텍스트 추출 (w 태그 통해), 줄바꿈 제거"""
    text_parts = []
    for c_elem in s_elem.findall('.//c'):
        for w_elem in c_elem.findall('.//w'):
            text = ''.join(w_elem.itertext()).strip()
            # 줄바꿈, 탭, 다중 공백 제거
            text = ' '.join(text.split())
            if text:
                text_parts.append(text)
    
    # c 태그가 없으면 직접 텍스트 추출
    if not text_parts:
        text = ''.join(s_elem.itertext()).strip()
        # 줄바꿈, 탭, 다중 공백 제거
        text = ' '.join(text.split())
        if text:
            text_parts.append(text)
    
    return text_parts


def parse_yeogi_xml(xml_path, content_type):
    """
    예기집설대전 XML 파일 파싱
    
    단락 태그를 기준으로 문단 단위 추출
    s id를 반환하여 원문 기준 문단 번호 매칭에 사용
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    data = []  # [(문단식별자, 문장id, text), ...]
    
    # 단락 태그를 기준으로 문단 추출
    danlak_list = root.findall('.//단락')
    
    for danlak_elem in danlak_list:
        # 단락 ID 속성을 문단식별자로 사용 (정수형으로 변환)
        para_id_str = danlak_elem.get('id', '0')
        para_identifier = int(para_id_str)
        
        # 단락 내의 모든 s 태그 찾기
        s_list = danlak_elem.findall('.//s')
        
        for s_elem in s_list:
            s_id = s_elem.get('id', '')
            text_parts = extract_text_from_s(s_elem)
            
            if text_parts:
                full_text = ' '.join(text_parts)
                # s_id를 문장식별자로 사용 (원문/번역문 매칭용)
                data.append((para_identifier, s_id, full_text))
    
    return data


def merge_and_save_yeogi(source_data, translation_data, output_excel_path, book_name):
    """원문과 번역문 데이터를 병합하여 문장 병렬 Excel로 저장
    
    s_id를 기준으로 매칭하고, 원문의 문단 번호를 사용
    예기집설대전1: 원문/번역문 단락 개수가 다르므로 s_id로 매칭
    당시삼백수: 원문/번역문 단락 개수가 같으므로 직접 매칭
    """
    
    # s_id → 원문 문단번호 매핑 생성
    s_id_to_src_para = {}
    for para_id, s_id, text in source_data:
        s_id_to_src_para[s_id] = para_id
    
    # s_id 기준으로 데이터 병합
    merged = defaultdict(dict)
    
    for para_id, s_id, text in source_data:
        # 텍스트 정규화: 줄바꿈, 탭 제거
        text = ' '.join(text.split())
        key = s_id  # s_id만 키로 사용
        merged[key]['원문'] = text
        merged[key]['문단식별자'] = para_id  # 원문 문단번호 저장
    
    for para_id, s_id, text in translation_data:
        # 텍스트 정규화: 줄바꿈, 탭 제거
        text = ' '.join(text.split())
        key = s_id
        merged[key]['번역문'] = text
        # 번역문의 문단번호는 무시하고, 원문 문단번호 사용
        if key not in merged:
            # 원문에 없는 s_id (예외 상황)
            merged[key]['번역문'] = text
            merged[key]['문단식별자'] = para_id  # 임시로 번역문 문단번호 사용
    
    # s_id 정렬하여 행 생성
    rows = []
    for s_id in sorted(merged.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        para_id = merged[s_id].get('문단식별자', '?')
        rows.append({
            '문단식별자': para_id,
            '문장식별자': s_id,
            '원문': merged[s_id].get('원문', ''),
            '번역문': merged[s_id].get('번역문', '')
        })
    
    # DataFrame 생성
    df = pd.DataFrame(rows)
    
    # Excel 파일로 저장
    df.to_excel(output_excel_path, index=False, engine='openpyxl')
    print(f"✓ 문장병렬 Excel 생성: {output_excel_path.name} ({len(rows)}개 행, {df['문단식별자'].nunique()} 문단)")
    
    return len(rows)


def process_yeogi_files(source_dir, output_base_dir):
    """단락 ID 기반 책들 처리 (예기집설대전1, 당시삼백수1~3)"""
    
    source_dir = Path(source_dir)
    output_base_dir = Path(output_base_dir)
    
    # 단락 ID 기반 추출이 필요한 책들
    target_books = [
        '예기집설대전1',
        '당시삼백수1',
        '당시삼백수2',
        '당시삼백수3'
    ]
    
    # 해당 책들의 원문 파일 찾기
    yeogi_files = []
    for book in target_books:
        files = list(source_dir.glob(f'*{book}*_원문_*.xml'))
        yeogi_files.extend(files)
    yeogi_files = sorted(yeogi_files)
    
    print(f"\n단락 ID 기반 책 처리 중...")
    print(f"대상: 예기집설대전1, 당시삼백수1~3")
    print(f"발견된 파일: {len(yeogi_files)}개\n")
    
    processed_count = 0
    total_rows = 0
    
    for source_xml in yeogi_files:
        # 대응하는 번역문 파일 찾기
        translation_xml = source_xml.parent / source_xml.name.replace('원문', '번역문')
        
        if not translation_xml.exists():
            print(f"⚠ 번역문 파일 없음: {source_xml.name}")
            continue
        
        # 책 이름 추출
        book_name = extract_book_name(source_xml.name)
        if not book_name:
            print(f"⚠ 책 이름 추출 실패: {source_xml.name}")
            continue
        
        # 출력 디렉토리
        book_output_dir = output_base_dir / book_name
        book_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Excel 파일명
        output_excel = book_output_dir / f"{book_name}_문장병렬.xlsx"
        
        try:
            print(f"처리 중: {source_xml.name}")
            
            # XML 파싱
            source_data = parse_yeogi_xml(source_xml, '원문')
            translation_data = parse_yeogi_xml(translation_xml, '번역문')
            
            print(f"  원문: {len(source_data)}개 문장 추출")
            print(f"  번역문: {len(translation_data)}개 문장 추출")
            
            # Excel로 저장
            row_count = merge_and_save_yeogi(source_data, translation_data, output_excel, book_name)
            
            processed_count += 1
            total_rows += row_count
            print()
            
        except Exception as e:
            print(f"✗ 오류 발생: {source_xml.name}")
            print(f"  에러: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"단락 ID 기반 책 처리 완료!")
    print(f"처리된 파일: {processed_count}개")
    print(f"총 데이터 행: {total_rows:,}개")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    source_directory = '/workspace/sources'
    output_directory = '/workspace/tsv_output'
    
    process_yeogi_files(source_directory, output_directory)
