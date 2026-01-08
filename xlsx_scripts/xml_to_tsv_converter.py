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
from datetime import datetime


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
    # 모든 단락을 찾아서 처리 (중첩된 원문/번역문 태그 무시)
    all_danrak = root.findall('.//단락')
    
    for para in all_danrak:
        for s_elem in para.findall('.//s'):
            s_id = s_elem.get('id', '')
            
            # s 태그 내의 모든 w 태그 찾기
            for c_elem in s_elem.findall('.//c'):
                for w_elem in c_elem.findall('.//w'):
                    w_id = w_elem.get('id', '')
                    text = ''.join(w_elem.itertext()).strip()
                    
                    if s_id and w_id:
                        # w_id별로 텍스트 저장 (빈 텍스트도 포함)
                        if w_id in data[s_id]:
                            data[s_id][w_id] += ' ' + text if text else ''
                        else:
                            data[s_id][w_id] = text
    
    # 단락 밖에 있는 s 태그들도 처리
    all_s = root.findall('.//s')
    s_in_danrak = set()
    for para in all_danrak:
        for s in para.findall('.//s'):
            s_in_danrak.add(id(s))
    
    for s_elem in all_s:
        if id(s_elem) in s_in_danrak:
            continue  # 이미 처리됨
        
        s_id = s_elem.get('id', '')
        
        # s 태그 내의 모든 w 태그 찾기
        for c_elem in s_elem.findall('.//c'):
            for w_elem in c_elem.findall('.//w'):
                w_id = w_elem.get('id', '')
                text = ''.join(w_elem.itertext()).strip()
                
                if s_id and w_id:
                    # w_id별로 텍스트 저장 (빈 텍스트도 포함)
                    if w_id in data[s_id]:
                        data[s_id][w_id] += ' ' + text if text else ''
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


def count_w_tags_in_xml(xml_path):
    """XML 파일의 전체 <w> 태그 개수 카운트"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return len(root.findall('.//w'))
    except Exception as e:
        return 0


def process_xml_pairs(source_dir, output_base_dir):
    """sources 디렉토리의 모든 XML 쌍을 처리"""
    
    source_dir = Path(source_dir)
    output_base_dir = Path(output_base_dir)
    
    # 로그 파일 초기화
    log_file = output_base_dir / 'conversion_issues.log'
    log_entries = []
    log_entries.append(f"{'='*80}")
    log_entries.append(f"XML to XLSX 변환 검증 로그")
    log_entries.append(f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_entries.append(f"{'='*80}\n")
    
    # 원문 파일 목록 가져오기
    source_files = sorted(source_dir.glob('*_원문_*.xml'))
    
    print(f"\n총 {len(source_files)}개의 원문 파일 발견\n")
    
    processed_count = 0
    total_rows = 0
    mismatch_count = 0
    
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
            
            # 검증: XML의 w 태그 개수와 XLSX 행 수 비교
            src_w_count = count_w_tags_in_xml(source_xml)
            tgt_w_count = count_w_tags_in_xml(translation_xml)
            
            if src_w_count != row_count or tgt_w_count != row_count:
                mismatch_count += 1
                diff = src_w_count - row_count
                status = "초과" if diff < 0 else "누락"
                
                log_entries.append(f"{'='*80}")
                log_entries.append(f"❌ {book_name}")
                log_entries.append(f"   파일: {source_xml.name}")
                log_entries.append(f"   원문 XML: {src_w_count}개 <w> 태그")
                log_entries.append(f"   번역문 XML: {tgt_w_count}개 <w> 태그")
                log_entries.append(f"   생성된 XLSX: {row_count}행")
                log_entries.append(f"   차이: {diff:+d}개 ({abs(diff)}개 {status})")
                log_entries.append("")
                
                # 원문/번역문 데이터 비교 분석
                src_pairs = set()
                for s_id in source_data:
                    for w_id in source_data[s_id]:
                        src_pairs.add((s_id, w_id))
                
                tgt_pairs = set()
                for s_id in translation_data:
                    for w_id in translation_data[s_id]:
                        tgt_pairs.add((s_id, w_id))
                
                # 원문에만 있는 (s_id, w_id)
                only_in_src = src_pairs - tgt_pairs
                if only_in_src:
                    log_entries.append(f"   ⚠️ 원문에만 존재 ({len(only_in_src)}개):")
                    for s_id, w_id in sorted(list(only_in_src)[:10]):  # 최대 10개
                        src_text = source_data[s_id][w_id][:50]
                        log_entries.append(f"      - s_id={s_id}, w_id={w_id}: '{src_text}'")
                    if len(only_in_src) > 10:
                        log_entries.append(f"      ... 외 {len(only_in_src) - 10}개")
                    log_entries.append("")
                
                # 번역문에만 있는 (s_id, w_id)
                only_in_tgt = tgt_pairs - src_pairs
                if only_in_tgt:
                    log_entries.append(f"   ⚠️ 번역문에만 존재 ({len(only_in_tgt)}개):")
                    for s_id, w_id in sorted(list(only_in_tgt)[:10]):  # 최대 10개
                        tgt_text = translation_data[s_id][w_id][:50]
                        log_entries.append(f"      - s_id={s_id}, w_id={w_id}: '{tgt_text}'")
                    if len(only_in_tgt) > 10:
                        log_entries.append(f"      ... 외 {len(only_in_tgt) - 10}개")
                    log_entries.append("")
                
                # 중복 w_id 검사 (원문)
                from collections import Counter
                src_w_ids = []
                for s_id in source_data:
                    src_w_ids.extend(source_data[s_id].keys())
                
                w_id_counts = Counter(src_w_ids)
                duplicates = {wid: cnt for wid, cnt in w_id_counts.items() if cnt > 1}
                
                if duplicates:
                    log_entries.append(f"   ⚠️ 원문 중복 w_id ({len(duplicates)}개):")
                    for wid, cnt in sorted(list(duplicates.items())[:10], key=lambda x: -x[1]):
                        # 어느 s_id들에서 중복되는지 찾기
                        s_ids_with_wid = [s_id for s_id in source_data if wid in source_data[s_id]]
                        log_entries.append(f"      - w_id={wid}: {cnt}번 등장, s_id={s_ids_with_wid[:5]}")
                    if len(duplicates) > 10:
                        log_entries.append(f"      ... 외 {len(duplicates) - 10}개")
                    log_entries.append("")
                
                # 중복 w_id 검사 (번역문)
                tgt_w_ids = []
                for s_id in translation_data:
                    tgt_w_ids.extend(translation_data[s_id].keys())
                
                tgt_w_id_counts = Counter(tgt_w_ids)
                tgt_duplicates = {wid: cnt for wid, cnt in tgt_w_id_counts.items() if cnt > 1}
                
                if tgt_duplicates:
                    log_entries.append(f"   ⚠️ 번역문 중복 w_id ({len(tgt_duplicates)}개):")
                    for wid, cnt in sorted(list(tgt_duplicates.items())[:10], key=lambda x: -x[1]):
                        s_ids_with_wid = [s_id for s_id in translation_data if wid in translation_data[s_id]]
                        log_entries.append(f"      - w_id={wid}: {cnt}번 등장, s_id={s_ids_with_wid[:5]}")
                    if len(tgt_duplicates) > 10:
                        log_entries.append(f"      ... 외 {len(tgt_duplicates) - 10}개")
                    log_entries.append("")
                
                log_entries.append("")
                
                print(f"  ⚠️ 불일치: 원문 {src_w_count}개, 번역문 {tgt_w_count}개, XLSX {row_count}행")
            
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
    print(f"불일치: {mismatch_count}개")
    print(f"출력 디렉토리: {output_base_dir}")
    print(f"{'='*60}\n")
    
    # 로그 파일 저장
    if mismatch_count > 0:
        log_entries.append(f"{'='*80}")
        log_entries.append(f"요약: 총 {processed_count}개 중 {mismatch_count}개 불일치")
        log_entries.append(f"{'='*80}")
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(log_entries))
        
        print(f"⚠️ 불일치 상세 로그: {log_file}")
    else:
        print(f"✅ 모든 파일이 정확히 일치합니다!")


if __name__ == '__main__':
    # 경로 설정
    source_directory = 'sources'
    output_directory = 'xlsx'
    
    # 실행
    process_xml_pairs(source_directory, output_directory)
