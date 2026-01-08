"""
XML 파일 쌍(원문, 번역문)을 TSV 및 Excel로 변환하는 스크립트
컬럼: 문장식별자, 구식별자, 원문, 번역문
"""

import xml.etree.ElementTree as ET
import os
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd


def extract_book_name(filename):
    """파일명에서 책 이름 추출"""
    # 예: jti_1e0201-[역주]춘추좌씨전1_원문_x-C2017.xml -> 춘추좌씨전1
    match = re.search(r'\[역주\](.+?)_(?:원문|번역문)', filename)
    if match:
        return match.group(1)
    return None


def parse_xml_content(xml_path, content_type):
    """
    XML 파일을 파싱하여 s_id, w_id, 텍스트를 추출
    content_type: '원문' 또는 '번역문'
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    data = defaultdict(lambda: defaultdict(str))  # {s_id: {w_id: text}}
    
    # 원문의 경우 여러 태그 형식을 지원
    if content_type == '원문':
        tag_names = ['원문', '경문', '전', '목원문', '강원문', '훈의원문']
    else:
        tag_names = [content_type, '경문', '전', '목번역문', '강번역문', '훈의번역문']
    
    # content_type에 따라 태그 찾기
    for tag_name in tag_names:
        for content_elem in root.findall(f'.//{tag_name}'):
            s_id_attr = content_elem.get('식별자', '')
            lang_attr = content_elem.get('lang', '')
            
            # 원문은 lang="chi", 번역문은 lang="kor"인 태그만 처리
            if content_type == '원문' and lang_attr and lang_attr != 'chi':
                continue
            if content_type == '번역문' and lang_attr and lang_attr != 'kor':
                continue
            
            # 단락 내의 s 태그들 찾기
            for para in content_elem.findall('.//단락'):
                for s_elem in para.findall('.//s'):
                    s_id = s_elem.get('id', '')
                    
                    # s 태그 내의 모든 w 태그 찾기
                    for c_elem in s_elem.findall('.//c'):
                        for w_elem in c_elem.findall('.//w'):
                            w_id = w_elem.get('id', '')
                            text = ''.join(w_elem.itertext()).strip()
                            
                            if s_id and w_id and text:
                                # w_id별로 텍스트 저장
                                if w_id in data[s_id]:
                                    data[s_id][w_id] += ' ' + text
                                else:
                                    data[s_id][w_id] = text
    
    return data


def merge_and_save(source_data, translation_data, output_excel_path, book_name):
    """원문과 번역문 데이터를 병합하여 Excel로 저장"""
    
    # 모든 s_id와 w_id 수집
    all_s_ids = sorted(set(list(source_data.keys()) + list(translation_data.keys())), 
                       key=lambda x: int(x) if x.isdigit() else 0)
    
    rows = []
    
    for s_id in all_s_ids:
        # 해당 s_id의 모든 w_id 수집
        source_w_ids = set(source_data.get(s_id, {}).keys())
        trans_w_ids = set(translation_data.get(s_id, {}).keys())
        all_w_ids = sorted(source_w_ids | trans_w_ids, 
                          key=lambda x: int(x) if x.isdigit() else 0)
        
        for w_id in all_w_ids:
            source_text = source_data.get(s_id, {}).get(w_id, '')
            trans_text = translation_data.get(s_id, {}).get(w_id, '')
            
            rows.append({
                '문장식별자': s_id,
                '구식별자': w_id,
                '원문': source_text,
                '번역문': trans_text
            })
    
    # DataFrame 생성
    df = pd.DataFrame(rows)
    
    # Excel 파일로 저장
    df.to_excel(output_excel_path, index=False, engine='openpyxl')
    print(f"✓ Excel 생성: {output_excel_path.name} ({len(rows)}개 행)")
    
    return len(rows)


def process_xml_pairs(source_dir, output_base_dir):
    """sources 디렉토리의 모든 XML 쌍을 처리"""
    
    source_dir = Path(source_dir)
    output_base_dir = Path(output_base_dir)
    
    # 원문 파일 목록 가져오기
    source_files = sorted(source_dir.glob('*_원문_*.xml'))
    
    print(f"\n총 {len(source_files)}개의 원문 파일 발견\n")
    
    processed_count = 0
    total_rows = 0
    
    for source_xml in source_files:
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
        
        # 출력 디렉토리 생성 (책별)
        book_output_dir = output_base_dir / book_name
        book_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Excel 파일명 생성 (형식: 책이름_구병렬.xlsx)
        output_excel = book_output_dir / f"{book_name}_구병렬.xlsx"
        
        try:
            print(f"처리 중: {source_xml.name}")
            
            # XML 파싱
            source_data = parse_xml_content(source_xml, '원문')
            translation_data = parse_xml_content(translation_xml, '번역문')
            
            # Excel로 저장
            row_count = merge_and_save(source_data, translation_data, output_excel, book_name)
            
            processed_count += 1
            total_rows += row_count
            print()
            
        except Exception as e:
            print(f"✗ 오류 발생: {source_xml.name}")
            print(f"  에러: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print(f"변환 완료!")
    print(f"처리된 파일: {processed_count}개")
    print(f"총 데이터 행: {total_rows:,}개")
    print(f"출력 디렉토리: {output_base_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # 경로 설정 (도커 환경)
    source_directory = '/workspace/sources'
    output_directory = '/workspace/tsv_output'
    
    # 실행
    process_xml_pairs(source_directory, output_directory)
