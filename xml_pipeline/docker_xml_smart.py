#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Docker XML 스마트 파이프라인
Docker 환경에서 XML 파일을 쉽게 처리할 수 있는 통합 인터페이스
GUI 없이도 복잡한 파일명을 번호로 간편하게 선택
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
import argparse
import json
from datetime import datetime

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from .xml_file_browser import XMLFileBrowser
    from .xml_pipeline_cli import XMLPipelineManager
except ImportError as e:
    print(f"⚠️ 모듈 import 오류: {e}")
    print("xml_file_browser.py와 xml_pipeline_cli.py가 같은 디렉토리에 있는지 확인하세요.")
    sys.exit(1)


class DockerXMLSmart:
    """Docker 환경용 스마트 XML 파이프라인"""
    
    def __init__(self):
        self.browser = XMLFileBrowser()
        self.pipeline_manager = XMLPipelineManager()
        
    def smart_menu(self):
        """스마트 메뉴 인터페이스"""
        
        while True:
            self._display_main_menu()
            
            try:
                choice = input("\n선택> ").strip()
                
                if choice == '1':
                    self._single_pair_processing()
                elif choice == '2':
                    self._batch_processing()
                elif choice == '3':
                    self._quick_pattern_processing()
                elif choice == '4':
                    self._view_recent_results()
                elif choice == '5':
                    self._file_browser_only()
                elif choice == '6':
                    self._show_help()
                elif choice.lower() in ['q', 'quit', 'exit']:
                    print("👋 Docker XML Smart 종료")
                    break
                else:
                    print("❌ 잘못된 선택입니다. 1-6 또는 q를 입력하세요.")
            
            except KeyboardInterrupt:
                print("\n👋 종료합니다.")
                break
            except EOFError:
                print("\n👋 종료합니다.")
                break
    
    def _display_main_menu(self):
        """메인 메뉴 표시"""
        print("\n" + "="*60)
        print("🐳 Docker XML Smart Pipeline")
        print("="*60)
        print("1. 📖 단일 XML 쌍 처리 (대화형 선택)")
        print("2. 📚 배치 처리 (여러 쌍 연속 처리)")  
        print("3. 🔍 패턴 검색 처리 (특정 키워드)")
        print("4. 📊 최근 결과 보기")
        print("5. 📂 파일 브라우저만 사용")
        print("6. ❓ 도움말")
        print("q. 🚪 종료")
        print("="*60)
    
    def _single_pair_processing(self):
        """단일 XML 쌍 처리"""
        print("\n📖 단일 XML 쌍 처리")
        print("-" * 40)
        
        # 파일 스캔
        print("🔍 XML 파일 스캔 중...")
        pairs = self.browser.scan_xml_pairs()
        
        if not pairs:
            print("❌ XML 쌍을 찾을 수 없습니다.")
            input("\nPress Enter to continue...")
            return
        
        # 대화형 선택
        result = self.browser.interactive_select()
        
        if not result:
            print("❌ 선택이 취소되었습니다.")
            input("\nPress Enter to continue...")
            return
        
        original_file, translation_file = result
        
        # 처리 확인
        print(f"\n🎯 선택된 XML 쌍:")
        print(f"원문: {Path(original_file).name}")
        print(f"번역: {Path(translation_file).name}")
        
        confirm = input("\n이 XML 쌍을 처리하시겠습니까? (y/N): ").strip().lower()
        
        if confirm not in ['y', 'yes']:
            print("❌ 처리가 취소되었습니다.")
            input("\nPress Enter to continue...")
            return
        
        # 파이프라인 실행
        try:
            print(f"\n🚀 XML 파이프라인 처리 시작...")
            
            # pair_id 생성
            book_name = Path(original_file).stem
            if '원문' in book_name:
                pair_id = book_name.replace('원문', '').replace('_x-C', '').replace('-', '').strip('_')
            else:
                pair_id = book_name[:20]
            
            result = self.pipeline_manager.process_single_pair(
                original_xml=original_file,
                translation_xml=translation_file,
                pair_id=pair_id
            )
            
            print(f"\n🎉 처리 완료!")
            print(f"📁 결과 위치: {result.get('result_folder', 'N/A')}")
            
        except Exception as e:
            print(f"\n❌ 처리 중 오류 발생: {e}")
        
        input("\nPress Enter to continue...")
    
    def _batch_processing(self):
        """배치 처리"""
        print("\n📚 배치 처리 모드")
        print("-" * 40)
        
        # 파일 스캔  
        print("🔍 XML 파일 스캔 중...")
        pairs = self.browser.scan_xml_pairs()
        
        if not pairs:
            print("❌ XML 쌍을 찾을 수 없습니다.")
            input("\nPress Enter to continue...")
            return
        
        print(f"✅ {len(pairs)}개 XML 쌍 발견")
        
        # 처리할 범위 선택
        print(f"\n📝 처리 옵션:")
        print(f"1. 전체 처리 ({len(pairs)}개 쌍)")
        print(f"2. 범위 선택 (예: 1-5)")
        print(f"3. 개별 선택 (예: 1,3,5)")
        
        choice = input("\n선택> ").strip()
        
        selected_pairs = []
        
        if choice == '1':
            # 전체 처리
            selected_pairs = [(pair['original_file'], pair['translation_file']) for pair in pairs]
        
        elif choice == '2':
            # 범위 선택
            range_input = input("범위 입력 (예: 1-5): ").strip()
            try:
                if '-' in range_input:
                    start, end = map(int, range_input.split('-'))
                    start = max(1, start)
                    end = min(len(pairs), end)
                    
                    for i in range(start-1, end):
                        pair = pairs[i]
                        selected_pairs.append((pair['original_file'], pair['translation_file']))
                else:
                    print("❌ 잘못된 형식입니다. 예: 1-5")
                    input("\nPress Enter to continue...")
                    return
            except ValueError:
                print("❌ 잘못된 형식입니다. 숫자로 입력하세요.")
                input("\nPress Enter to continue...")
                return
        
        elif choice == '3':
            # 개별 선택
            indices_input = input("번호 입력 (예: 1,3,5): ").strip()
            try:
                indices = [int(x.strip()) for x in indices_input.split(',')]
                
                for idx in indices:
                    if 1 <= idx <= len(pairs):
                        pair = pairs[idx-1]
                        selected_pairs.append((pair['original_file'], pair['translation_file']))
                    else:
                        print(f"⚠️ 번호 {idx}는 범위를 벗어납니다.")
                
            except ValueError:
                print("❌ 잘못된 형식입니다. 숫자와 쉼표로 입력하세요.")
                input("\nPress Enter to continue...")
                return
        
        else:
            print("❌ 잘못된 선택입니다.")
            input("\nPress Enter to continue...")
            return
        
        if not selected_pairs:
            print("❌ 선택된 쌍이 없습니다.")
            input("\nPress Enter to continue...")
            return
        
        # 최종 확인
        print(f"\n🎯 {len(selected_pairs)}개 쌍이 선택되었습니다:")
        for i, (orig, trans) in enumerate(selected_pairs[:5], 1):  # 처음 5개만 표시
            print(f"  {i}. {Path(orig).name} + {Path(trans).name}")
        
        if len(selected_pairs) > 5:
            print(f"  ... 및 {len(selected_pairs) - 5}개 더")
        
        confirm = input(f"\n이 {len(selected_pairs)}개 쌍을 모두 처리하시겠습니까? (y/N): ").strip().lower()
        
        if confirm not in ['y', 'yes']:
            print("❌ 배치 처리가 취소되었습니다.")
            input("\nPress Enter to continue...")
            return
        
        # 배치 처리 실행
        print(f"\n🚀 배치 처리 시작... ({len(selected_pairs)}개 쌍)")
        
        success_count = 0
        error_count = 0
        
        for i, (original_file, translation_file) in enumerate(selected_pairs, 1):
            try:
                print(f"\n📖 처리 중 ({i}/{len(selected_pairs)}): {Path(original_file).name}")
                
                # pair_id 생성
                book_name = Path(original_file).stem
                if '원문' in book_name:
                    pair_id = book_name.replace('원문', '').replace('_x-C', '').replace('-', '').strip('_')
                else:
                    pair_id = f"{book_name[:20]}_{i}"
                
                result = self.pipeline_manager.process_single_pair(
                    original_xml=original_file,
                    translation_xml=translation_file,
                    pair_id=pair_id
                )
                
                success_count += 1
                print(f"✅ 완료: {Path(original_file).name}")
                
            except Exception as e:
                error_count += 1
                print(f"❌ 오류: {Path(original_file).name} - {e}")
        
        print(f"\n🎉 배치 처리 완료!")
        print(f"✅ 성공: {success_count}개")
        print(f"❌ 실패: {error_count}개")
        
        input("\nPress Enter to continue...")
    
    def _quick_pattern_processing(self):
        """패턴 검색 처리"""
        print("\n🔍 패턴 검색 처리")
        print("-" * 40)
        
        # 파일 스캔
        print("🔍 XML 파일 스캔 중...")
        pairs = self.browser.scan_xml_pairs()
        
        if not pairs:
            print("❌ XML 쌍을 찾을 수 없습니다.")
            input("\nPress Enter to continue...")
            return
        
        # 패턴 입력
        pattern = input("검색할 패턴을 입력하세요 (예: 한유, 구양수): ").strip()
        
        if not pattern:
            print("❌ 패턴이 입력되지 않았습니다.")
            input("\nPress Enter to continue...")
            return
        
        # 패턴 매칭
        matching_pairs = self.browser.quick_select_by_pattern(pattern)
        
        if not matching_pairs:
            input("\nPress Enter to continue...")
            return
        
        # 처리 확인
        confirm = input(f"\n매칭된 {len(matching_pairs)}개 쌍을 모두 처리하시겠습니까? (y/N): ").strip().lower()
        
        if confirm not in ['y', 'yes']:
            print("❌ 처리가 취소되었습니다.")
            input("\nPress Enter to continue...")
            return
        
        # 처리 실행
        print(f"\n🚀 패턴 처리 시작... ({len(matching_pairs)}개 쌍)")
        
        success_count = 0
        error_count = 0
        
        for i, (original_file, translation_file) in enumerate(matching_pairs, 1):
            try:
                print(f"\n📖 처리 중 ({i}/{len(matching_pairs)}): {Path(original_file).name}")
                
                # pair_id 생성
                book_name = Path(original_file).stem
                if '원문' in book_name:
                    pair_id = book_name.replace('원문', '').replace('_x-C', '').replace('-', '').strip('_')
                else:
                    pair_id = f"{book_name[:20]}_{i}"
                
                result = self.pipeline_manager.process_single_pair(
                    original_xml=original_file,
                    translation_xml=translation_file,
                    pair_id=pair_id
                )
                
                success_count += 1
                print(f"✅ 완료: {Path(original_file).name}")
                
            except Exception as e:
                error_count += 1
                print(f"❌ 오류: {Path(original_file).name} - {e}")
        
        print(f"\n🎉 패턴 처리 완료!")
        print(f"✅ 성공: {success_count}개")
        print(f"❌ 실패: {error_count}개")
        
        input("\nPress Enter to continue...")
    
    def _view_recent_results(self):
        """최근 결과 보기"""
        print("\n📊 최근 결과 조회")
        print("-" * 40)
        
        try:
            self.pipeline_manager.list_recent_results(limit=20)
        except Exception as e:
            print(f"❌ 결과 조회 중 오류: {e}")
        
        input("\nPress Enter to continue...")
    
    def _file_browser_only(self):
        """파일 브라우저만 사용"""
        print("\n📂 XML 파일 브라우저")
        print("-" * 40)
        
        # 파일 스캔
        print("🔍 XML 파일 스캔 중...")
        pairs = self.browser.scan_xml_pairs()
        
        if not pairs:
            print("❌ XML 쌍을 찾을 수 없습니다.")
            input("\nPress Enter to continue...")
            return
        
        # 브라우저만 실행
        result = self.browser.interactive_select()
        
        if result:
            original_file, translation_file = result
            print(f"\n🎯 선택된 XML 쌍:")
            print(f"원문: {original_file}")
            print(f"번역: {translation_file}")
            print(f"\n💡 이 파일들을 메뉴 1번에서 처리하실 수 있습니다.")
        
        input("\nPress Enter to continue...")
    
    def _show_help(self):
        """도움말 표시"""
        help_text = """
📖 Docker XML Smart Pipeline 도움말

🎯 주요 기능:
  • 복잡한 한국어 XML 파일명을 번호로 쉽게 선택
  • 단일 쌍 처리부터 대량 배치 처리까지 지원
  • 패턴 검색으로 특정 도서만 골라서 처리
  • 실시간 파일 스캔 및 자동 매칭

🔍 파일 스캔:
  • sources/ 디렉토리에서 자동으로 XML 쌍 검색
  • '원문', '번역문' 키워드로 자동 매칭
  • 'original', 'translation' 키워드도 지원

📝 선택 방법:
  • 번호 입력: 1, 2, 3...
  • 페이지 이동: 'n' (다음), 'p' (이전)
  • 필터링: 'f 키워드' (예: f 한유)
  • 새로고침: 'r'
  • 종료: 'q'

🚀 처리 방식:
  1. 단일 처리: 하나씩 선택해서 처리
  2. 배치 처리: 여러 개를 범위나 개별 선택으로 처리
  3. 패턴 처리: 키워드로 검색해서 일괄 처리

📁 결과 저장:
  • xml_pipeline_results/ 디렉토리에 책별로 저장
  • 각 책마다 독립적인 폴더에 모든 분석 결과 포함
  • JSON, CSV, Excel 등 다양한 형식 지원

💡 사용 팁:
  • 처음에는 단일 처리로 테스트해보세요
  • 패턴 검색은 책 제목의 일부만 입력해도 됩니다
  • 배치 처리 전에 범위를 잘 확인하세요
  • 오류 발생 시 로그를 확인하여 문제를 파악하세요
"""
        print(help_text)
        input("\nPress Enter to continue...")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Docker XML Smart Pipeline")
    parser.add_argument('--mode', '-m', choices=['menu', 'single', 'batch', 'pattern'], 
                       default='menu', help='실행 모드 선택')
    parser.add_argument('--pattern', '-p', help='패턴 모드에서 사용할 검색 패턴')
    
    args = parser.parse_args()
    
    smart = DockerXMLSmart()
    
    if args.mode == 'menu':
        smart.smart_menu()
    elif args.mode == 'single':
        smart._single_pair_processing()
    elif args.mode == 'batch':
        smart._batch_processing()  
    elif args.mode == 'pattern':
        if args.pattern:
            # 패턴이 주어진 경우 자동 실행
            print(f"🔍 패턴 '{args.pattern}'로 자동 처리 중...")
            # TODO: 자동 패턴 처리 구현
        else:
            smart._quick_pattern_processing()


if __name__ == "__main__":
    main()