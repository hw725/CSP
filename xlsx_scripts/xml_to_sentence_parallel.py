"""
XML 파일 쌍(원문, 번역문)을 문장 병렬 Excel로 변환하는 스크립트
컬럼: 문단식별자, 문장식별자, 원문, 번역문
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


def parse_xml_sentence_level(xml_path, content_type):
    """
    XML 파일에서 모든 <s>를 순회하며 문단식별자, 문장id, 텍스트를 추출한다.
    - content_type: '원문' 또는 '번역문'
    - 문단식별자: 가장 가까운 상위 노드의 '식별자' 속성 (예: ID:W1, ID:M1 등)
    - 문장id: s 태그의 id 속성
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    data = []  # [(문단식별자, 문장id, text), ...]

    def should_use_lang(lang_value):
        """lang 필터를 통과해야만 수집한다."""
        if not lang_value:
            return True
        if content_type == '원문':
            return lang_value == 'chi'
        return lang_value == 'kor'

    def collect_sentence_text(s_elem):
        text_parts = []
        for w_elem in s_elem.findall('.//w'):
            text_parts.append(''.join(w_elem.itertext()).strip())
        return ' '.join(text_parts).strip()

    def walk(elem, parent_para_ids=None, current_lang=None):
        """
        parent_para_ids: 상위 노드들의 식별자 리스트 (가장 먼 조상부터)
        s 태그 발견 시, 가장 가까운 상위 노드의 식별자를 사용
        """
        if parent_para_ids is None:
            parent_para_ids = []
        if current_lang is None:
            current_lang = root.get('lang', '')

        # 현재 노드의 식별자가 있으면 리스트에 추가
        next_para_ids = parent_para_ids.copy()
        if '식별자' in elem.attrib:
            next_para_ids.append(elem.get('식별자'))

        # 현재 노드의 언어 설정
        if 'lang' in elem.attrib:
            current_lang = elem.get('lang')

        if elem.tag == 's':
            if should_use_lang(current_lang):
                s_id = elem.get('id', '')
                full_text = collect_sentence_text(elem)
                # 가장 가까운 상위 식별자 사용 (리스트의 마지막 요소)
                para_id = next_para_ids[-1] if next_para_ids else ''
                if s_id and para_id:
                    data.append((para_id, s_id, full_text))

        for child in elem:
            walk(child, next_para_ids, current_lang)

    walk(root)
    return data


def merge_and_save_sentence(source_data, translation_data, output_excel_path, book_name):
    """원문과 번역문 데이터를 병합하여 문장 병렬 Excel로 저장
    - 병합 키는 문장식별자(s_id)만 사용해 중복 행을 방지한다.
    - 문단식별자 선택 우선순위: 원문 -> 번역문 (ID:W1_T는 ID:W1으로 정규화)
    """

    def normalize_para_id(para_id):
        if para_id and para_id.endswith('_T'):
            return para_id[:-2]
        return para_id

    merged = defaultdict(dict)  # {s_id: {"para_id": str, "원문": text, "번역문": text}}

    for para_id, s_id, text in source_data:
        normalized_para_id = normalize_para_id(para_id)
        entry = merged[s_id]
        # 원문이 우선하므로 먼저 para_id 설정
        if 'para_id' not in entry or not entry['para_id']:
            entry['para_id'] = normalized_para_id
        entry['원문'] = text

    for para_id, s_id, text in translation_data:
        normalized_para_id = normalize_para_id(para_id)
        entry = merged[s_id]
        # 원문에 para_id가 없는 경우 번역문 것으로 보충
        if 'para_id' not in entry or not entry['para_id']:
            entry['para_id'] = normalized_para_id
        entry['번역문'] = text

    rows = []
    for s_id in sorted(merged.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        entry = merged[s_id]
        rows.append({
            '문단식별자': entry.get('para_id', ''),
            '문장식별자': s_id,
            '원문': entry.get('원문', ''),
            '번역문': entry.get('번역문', '')
        })

    df = pd.DataFrame(rows)
    df.to_excel(output_excel_path, index=False, engine='openpyxl')
    
    # 원본 식별자 정보 출력
    unique_para_ids = df['문단식별자'].unique()
    print(f"✓ 문장병렬 Excel 생성: {output_excel_path.name} ({len(rows)}개 행, {len(unique_para_ids)}개 고유 문단식별자)")
    if len(unique_para_ids) <= 50:
        print(f"   고유 식별자: {sorted(unique_para_ids)}")

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
        
        # 출력 디렉토리 (기존 디렉토리 사용)
        book_output_dir = output_base_dir / book_name
        book_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Excel 파일명 생성 (형식: 책이름_문장병렬.xlsx)
        output_excel = book_output_dir / f"{book_name}_문장병렬.xlsx"
        
        try:
            print(f"처리 중: {source_xml.name}")
            
            # XML 파싱
            source_data = parse_xml_sentence_level(source_xml, '원문')
            translation_data = parse_xml_sentence_level(translation_xml, '번역문')
            
            # Excel로 저장
            row_count = merge_and_save_sentence(source_data, translation_data, output_excel, book_name)
            
            processed_count += 1
            total_rows += row_count
            print()
            
        except Exception as e:
            print(f"✗ 오류 발생: {source_xml.name}")
            print(f"  에러: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print(f"문장 병렬 변환 완료!")
    print(f"처리된 파일: {processed_count}개")
    print(f"총 데이터 행: {total_rows:,}개")
    print(f"출력 디렉토리: {output_base_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # 경로 설정 (도커 환경)
    source_directory = '/workspace/sources'
    output_directory = '/workspace/xlsx'
    
    # 실행
    process_xml_pairs(source_directory, output_directory)
