#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XML 파일 브라우저 - Docker 환경용 스마트 파일 선택 도구
복잡한 한국어 파일명을 번호로 쉽게 선택할 수 있는 GUI 대안
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse


class XMLFileBrowser:
    """Docker 환경용 스마트 XML 파일 브라우저"""
    
    def __init__(self, base_dir: str = "sources"):
        self.base_dir = Path(base_dir)
        self.xml_pairs = []
        self.current_page = 0
        self.items_per_page = 10
        
    def scan_xml_pairs(self, pattern: str = "*") -> List[Dict]:
        """XML 쌍들을 자동으로 스캔하여 매칭"""
        print(f"📂 XML 파일 스캔 중: {self.base_dir}")
        
        if not self.base_dir.exists():
            print(f"❌ 디렉토리가 존재하지 않습니다: {self.base_dir}")
            return []
        
        # 원문 파일들 찾기
        original_files = []
        for ext in ['*.xml']:
            for keyword in ['원문', 'original', 'Original']:
                pattern_search = f"*{keyword}*{ext}"
                found_files = list(self.base_dir.glob(pattern_search))
                original_files.extend(found_files)
        
        # 중복 제거
        original_files = list(set(original_files))
        
        xml_pairs = []
        
        for orig_file in original_files:
            # 대응하는 번역문 파일 찾기
            orig_name = orig_file.stem
            
            # 다양한 매칭 패턴 시도
            translation_candidates = []
            
            # 패턴 1: 원문 → 번역문
            if "원문" in orig_name:
                trans_name = orig_name.replace("원문", "번역문")
                translation_candidates.append(trans_name)
            
            # 패턴 2: original → translation
            if "original" in orig_name.lower():
                trans_name = orig_name.replace("original", "translation").replace("Original", "Translation")
                translation_candidates.append(trans_name)
            
            # 패턴 3: 기타 키워드
            for old, new in [("src", "tgt"), ("source", "target"), ("한문", "번역")]:
                if old in orig_name:
                    trans_name = orig_name.replace(old, new)
                    translation_candidates.append(trans_name)
            
            # 번역문 파일 찾기
            trans_file = None
            for candidate in translation_candidates:
                candidate_path = self.base_dir / f"{candidate}.xml"
                if candidate_path.exists():
                    trans_file = candidate_path
                    break
            
            if trans_file:
                # 책 이름 추출 (공통 부분)
                book_name = self._extract_book_name(orig_name)
                
                xml_pairs.append({
                    'book_name': book_name,
                    'original_file': str(orig_file),
                    'translation_file': str(trans_file),
                    'original_name': orig_file.name,
                    'translation_name': trans_file.name,
                    'size_mb': (orig_file.stat().st_size + trans_file.stat().st_size) / 1024 / 1024
                })
        
        # 책 이름으로 정렬
        xml_pairs.sort(key=lambda x: x['book_name'])
        
        self.xml_pairs = xml_pairs
        print(f"✅ {len(xml_pairs)}개 XML 쌍 발견")
        
        return xml_pairs
    
    def _extract_book_name(self, filename: str) -> str:
        """파일명에서 책 이름 추출"""
        # 공통 패턴들 제거
        clean_name = filename
        
        # 접두사 제거
        prefixes = [r'jti_\w+-', r'\[역주\]']
        for prefix in prefixes:
            clean_name = re.sub(prefix, '', clean_name)
        
        # 접미사 제거  
        suffixes = [r'_원문_.*', r'_번역문_.*', r'_original.*', r'_translation.*', r'-C\d+.*']
        for suffix in suffixes:
            clean_name = re.sub(suffix, '', clean_name)
        
        # 대괄호와 하이픈 제거
        clean_name = re.sub(r'[\[\-\]]', '', clean_name)
        
        # 연속된 공백을 하나로 통일
        clean_name = re.sub(r'\s+', ' ', clean_name).strip()
        
        # 숫자로 끝나는 경우 (권수)
        if re.search(r'\d+$', clean_name):
            return clean_name
        
        return clean_name or filename[:20]  # 최대 20자
    
    def display_pairs(self, pattern_filter: str = None) -> None:
        """XML 쌍들을 페이지별로 표시"""
        pairs = self.xml_pairs
        
        if pattern_filter:
            pairs = [pair for pair in pairs if pattern_filter.lower() in pair['book_name'].lower()]
        
        if not pairs:
            print("❌ 매칭되는 XML 쌍이 없습니다.")
            return
        
        total_pages = (len(pairs) - 1) // self.items_per_page + 1
        start_idx = self.current_page * self.items_per_page
        end_idx = min(start_idx + self.items_per_page, len(pairs))
        
        print(f"\n📚 XML 쌍 목록 (페이지 {self.current_page + 1}/{total_pages})")
        print("=" * 80)
        
        for i in range(start_idx, end_idx):
            pair = pairs[i]
            idx = i + 1
            
            print(f"{idx:2d}. 📖 {pair['book_name']}")
            print(f"     원문: {pair['original_name']}")
            print(f"     번역: {pair['translation_name']}")
            print(f"     크기: {pair['size_mb']:.1f}MB")
            print()
        
        print("=" * 80)
        if total_pages > 1:
            print(f"📄 페이지 {self.current_page + 1}/{total_pages}")
            if self.current_page > 0:
                print("   이전 페이지: 'p' 입력")
            if self.current_page < total_pages - 1:
                print("   다음 페이지: 'n' 입력")
        
        print(f"📝 선택: 번호 입력 (1-{len(pairs)})")
        print(f"🔍 필터: 'f 키워드' 입력")
        print(f"🔄 새로고침: 'r' 입력")
        print(f"❌ 종료: 'q' 입력")
    
    def interactive_select(self) -> Optional[Tuple[str, str]]:
        """대화형 XML 쌍 선택"""
        if not self.xml_pairs:
            print("❌ 스캔된 XML 쌍이 없습니다. 먼저 scan_xml_pairs()를 실행하세요.")
            return None
        
        current_filter = None
        
        while True:
            self.display_pairs(current_filter)
            
            try:
                user_input = input("\n선택> ").strip()
                
                if not user_input:
                    continue
                
                # 종료
                if user_input.lower() == 'q':
                    print("👋 종료합니다.")
                    return None
                
                # 새로고침
                elif user_input.lower() == 'r':
                    print("🔄 파일 목록 새로고침 중...")
                    self.scan_xml_pairs()
                    current_filter = None
                    self.current_page = 0
                    continue
                
                # 다음 페이지
                elif user_input.lower() == 'n':
                    pairs = self.xml_pairs
                    if current_filter:
                        pairs = [pair for pair in pairs if current_filter.lower() in pair['book_name'].lower()]
                    
                    total_pages = (len(pairs) - 1) // self.items_per_page + 1
                    if self.current_page < total_pages - 1:
                        self.current_page += 1
                    continue
                
                # 이전 페이지
                elif user_input.lower() == 'p':
                    if self.current_page > 0:
                        self.current_page -= 1
                    continue
                
                # 필터
                elif user_input.lower().startswith('f '):
                    current_filter = user_input[2:].strip()
                    self.current_page = 0
                    print(f"🔍 필터 적용: '{current_filter}'")
                    continue
                
                # 번호 선택
                else:
                    try:
                        choice = int(user_input)
                        pairs = self.xml_pairs
                        if current_filter:
                            pairs = [pair for pair in pairs if current_filter.lower() in pair['book_name'].lower()]
                        
                        if 1 <= choice <= len(pairs):
                            selected_pair = pairs[choice - 1]
                            
                            print(f"\n✅ 선택됨: {selected_pair['book_name']}")
                            print(f"   원문: {selected_pair['original_file']}")
                            print(f"   번역: {selected_pair['translation_file']}")
                            
                            return (selected_pair['original_file'], selected_pair['translation_file'])
                        else:
                            print(f"❌ 잘못된 번호입니다. 1-{len(pairs)} 범위 내에서 선택하세요.")
                    
                    except ValueError:
                        print("❌ 숫자를 입력하세요.")
            
            except KeyboardInterrupt:
                print("\n👋 종료합니다.")
                return None
            except EOFError:
                print("\n👋 종료합니다.")
                return None
    
    def quick_select_by_pattern(self, pattern: str) -> Optional[List[Tuple[str, str]]]:
        """패턴으로 빠른 선택"""
        matching_pairs = []
        
        for pair in self.xml_pairs:
            if pattern.lower() in pair['book_name'].lower():
                matching_pairs.append((pair['original_file'], pair['translation_file']))
        
        if not matching_pairs:
            print(f"❌ '{pattern}' 패턴과 매칭되는 XML 쌍이 없습니다.")
            return None
        
        print(f"✅ '{pattern}' 패턴으로 {len(matching_pairs)}개 쌍 선택됨")
        for orig, trans in matching_pairs:
            print(f"   - {Path(orig).name} + {Path(trans).name}")
        
        return matching_pairs


def main():
    """CLI 인터페이스"""
    parser = argparse.ArgumentParser(description="XML 파일 브라우저 - Docker 환경용")
    parser.add_argument('--dir', '-d', default='sources', help='XML 파일 디렉토리 (기본: sources)')
    parser.add_argument('--pattern', '-p', help='패턴으로 빠른 선택')
    parser.add_argument('--interactive', '-i', action='store_true', help='대화형 모드')
    
    args = parser.parse_args()
    
    browser = XMLFileBrowser(args.dir)
    
    print("🔍 XML 파일 브라우저 시작")
    print(f"📂 스캔 디렉토리: {browser.base_dir}")
    
    # 파일 스캔
    pairs = browser.scan_xml_pairs()
    
    if not pairs:
        print("❌ XML 쌍을 찾을 수 없습니다.")
        return
    
    # 패턴 모드
    if args.pattern:
        result = browser.quick_select_by_pattern(args.pattern)
        if result:
            print(f"\n📋 선택된 {len(result)}개 쌍:")
            for i, (orig, trans) in enumerate(result, 1):
                print(f"{i}. {Path(orig).name} → {Path(trans).name}")
    
    # 대화형 모드
    elif args.interactive or not args.pattern:
        result = browser.interactive_select()
        if result:
            orig_file, trans_file = result
            print(f"\n🎯 최종 선택:")
            print(f"원문: {orig_file}")
            print(f"번역: {trans_file}")
            print(f"\n💡 다음 단계: 이 파일들로 XML 파이프라인을 실행하세요!")


if __name__ == "__main__":
    main()