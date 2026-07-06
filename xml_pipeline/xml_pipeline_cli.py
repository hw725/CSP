#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XML 파이프라인 관리 CLI 도구
XML 쌍 처리, 결과 조회, 분석을 위한 통합 인터페이스
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Python 경로 설정 (CLI 실행용)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# XML 파이프라인 모듈 import
try:
    from xml_pipeline.xml_pipeline_processor import XMLPipelineProcessor
    from xml_extractor import XMLPair, create_xml_pair_from_directory
except ImportError:
    # 직접 실행 시 대체 import
    from xml_pipeline_processor import XMLPipelineProcessor
    from xml_extractor import XMLPair, create_xml_pair_from_directory

class XMLPipelineManager:
    """XML 파이프라인 관리자"""
    
    def __init__(self, output_dir: str = "xml_pipeline_results"):
        self.processor = XMLPipelineProcessor(output_dir)
        self.output_dir = Path(output_dir)
    
    def process_single_pair(self, original_xml: str, translation_xml: str, pair_id: str = None):
        """단일 XML 쌍 처리"""
        
        if not pair_id:
            # 파일명에서 자동 생성
            orig_name = Path(original_xml).stem
            pair_id = orig_name.replace('원문', '').replace('번역문', '').strip('-_')
            if not pair_id:
                pair_id = f"pair_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        xml_pair = XMLPair(
            pair_id=pair_id,
            name=f"{Path(original_xml).stem} + {Path(translation_xml).stem}",
            original_path=original_xml,
            translation_path=translation_xml,
            description="Single pair processing"
        )
        
        print(f"🚀 XML 쌍 처리 시작: {pair_id}")
        print(f"   원문: {original_xml}")
        print(f"   번역문: {translation_xml}")
        print()
        
        # XML 쌍 등록
        self.processor.add_xml_pair(xml_pair)
        
        # 파이프라인 실행
        results = self.processor.process_xml_pair_pipeline(xml_pair)
        
        # 결과 출력
        self._print_results(pair_id, results)
        
        return results
    
    def process_directory(self, xml_dir: str, pattern: str = "*원문*.xml"):
        """디렉토리의 모든 XML 쌍 처리"""
        
        print(f"📂 디렉토리 검색: {xml_dir}")
        print(f"   패턴: {pattern}")
        print()
        
        # XML 쌍들 자동 생성
        xml_pairs = create_xml_pair_from_directory(xml_dir, pattern)
        
        if not xml_pairs:
            print("❌ 매칭되는 XML 쌍을 찾을 수 없습니다.")
            print("   - 원문 XML 파일이 있는지 확인하세요")
            print("   - 대응하는 번역문 XML 파일이 있는지 확인하세요")
            return
        
        print(f"✅ 발견된 XML 쌍: {len(xml_pairs)}개")
        for pair in xml_pairs:
            print(f"   - {pair.pair_id}: {pair.name}")
        print()
        
        # 확인 메시지
        response = input(f"{len(xml_pairs)}개 XML 쌍을 모두 처리하시겠습니까? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("취소됨")
            return
        
        # 각 쌍 처리
        all_results = {}
        for i, xml_pair in enumerate(xml_pairs, 1):
            print(f"\n{'='*60}")
            print(f"처리 중 ({i}/{len(xml_pairs)}): {xml_pair.pair_id}")
            print(f"{'='*60}")
            
            try:
                # XML 쌍 등록
                self.processor.add_xml_pair(xml_pair)
                
                # 파이프라인 실행
                results = self.processor.process_xml_pair_pipeline(xml_pair)
                all_results[xml_pair.pair_id] = results
                
                # 간단한 결과 출력
                self._print_brief_results(xml_pair.pair_id, results)
                
            except Exception as e:
                print(f"❌ {xml_pair.pair_id} 처리 실패: {e}")
                all_results[xml_pair.pair_id] = {
                    'pair_id': xml_pair.pair_id,
                    'error': str(e),
                    'status': 'failed'
                }
                continue  # 다음 쌍 계속 처리
        
        # 전체 요약
        print(f"\n{'='*60}")
        print("전체 처리 완료 요약")
        print(f"{'='*60}")
        
        for pair_id, results in all_results.items():
            # 안전한 처리시간 및 성공 카운트 계산
            success_count = 0
            total_count = 0
            total_time = 0
            
            if 'stages' in results:
                for stage_data in results['stages'].values():
                    if isinstance(stage_data, dict):
                        total_count += 1
                        if 'time' in stage_data:
                            total_time += stage_data['time']
                        if stage_data.get('status') == 'success':
                            success_count += 1
            
            print(f"✅ {pair_id}: {success_count}/{total_count} 단계 성공 ({total_time:.1f}초)")
        
        return all_results
    
    def list_recent_results(self, limit: int = 10):
        """최근 처리 결과 목록"""
        
        import sqlite3
        
        conn = sqlite3.connect(self.processor.db_path)
        cursor = conn.cursor()
        
        # 최근 XML 쌍들 조회  
        cursor.execute("""
            SELECT DISTINCT pair_id, name, status
            FROM xml_pairs
            ORDER BY pair_id DESC
            LIMIT ?
        """, (limit,))
        
        pairs = cursor.fetchall()
        
        if not pairs:
            print("처리된 XML 쌍이 없습니다.")
            return
        
        print(f"최근 처리된 XML 쌍 ({len(pairs)}개):")
        print("-" * 80)
        
        for pair_id, name, status in pairs:
            # 각 쌍의 단계별 결과 조회
            cursor.execute("""
                SELECT stage, status, processing_time, accuracy_score
                FROM pipeline_results
                WHERE pair_id = ?
                ORDER BY timestamp
            """, (pair_id,))
            
            stage_results = cursor.fetchall()
            
            success_count = sum(1 for _, status, _, _ in stage_results if status == 'success')
            total_time = sum(time for _, _, time, _ in stage_results if time)
            
            # 정확도 점수들 수집
            accuracy_scores = [score for _, _, _, score in stage_results if score is not None]
            avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else None
            
            print(f"📄 {pair_id}")
            print(f"   이름: {name}")
            print(f"   상태: {status}")
            print(f"   성공 단계: {success_count}/{len(stage_results)}")
            print(f"   처리 시간: {total_time:.1f}초")
            if avg_accuracy:
                print(f"   평균 정확도: {avg_accuracy:.3f}")
            print()
        
        conn.close()
    
    def show_pair_details(self, pair_id: str):
        """특정 XML 쌍의 상세 결과 조회"""
        
        import sqlite3
        
        conn = sqlite3.connect(self.processor.db_path)
        cursor = conn.cursor()
        
        # XML 쌍 기본 정보
        cursor.execute("""
            SELECT name, original_file, translation_file, created_at, status
            FROM xml_pairs
            WHERE pair_id = ?
        """, (pair_id,))
        
        pair_info = cursor.fetchone()
        
        if not pair_info:
            print(f"❌ XML 쌍을 찾을 수 없습니다: {pair_id}")
            return
        
        name, original_file, translation_file, created_at, status = pair_info
        
        print(f"📄 XML 쌍 상세 정보: {pair_id}")
        print(f"{'='*60}")
        print(f"이름: {name}")
        print(f"상태: {status}")
        print(f"처리 일시: {created_at}")
        print(f"원문 파일: {original_file}")
        print(f"번역문 파일: {translation_file}")
        print()
        
        # 단계별 결과
        cursor.execute("""
            SELECT stage, status, processing_time, accuracy_score, notes, timestamp
            FROM pipeline_results
            WHERE pair_id = ?
            ORDER BY timestamp
        """, (pair_id,))
        
        results = cursor.fetchall()
        
        if not results:
            print("처리 결과가 없습니다.")
            return
        
        print("단계별 처리 결과:")
        print("-" * 60)
        
        for (stage, status, processing_time, accuracy_score, notes, timestamp) in results:
            
            status_emoji = "✅" if status == 'success' else "❌"
            
            print(f"{status_emoji} {stage.upper()}")
            print(f"   상태: {status}")
            print(f"   시간: {timestamp}")
            if processing_time:
                print(f"   처리시간: {processing_time:.2f}초")
            if accuracy_score:
                print(f"   정확도: {accuracy_score:.3f}")
            if notes:
                print(f"   메모: {notes}")
            
            print()
        
        conn.close()
        
        # 결과 디렉토리 경로 표시
        result_dirs = list(self.output_dir.glob(f"{pair_id}_*"))
        if result_dirs:
            latest_dir = max(result_dirs, key=os.path.getmtime)
            print(f"📁 결과 디렉토리: {latest_dir}")
    
    def cleanup_old_results(self, days: int = 7):
        """오래된 결과 정리"""
        
        import sqlite3
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()
        
        conn = sqlite3.connect(self.processor.db_path)
        cursor = conn.cursor()
        
        # 오래된 XML 쌍들 조회
        cursor.execute("""
            SELECT pair_id, name, created_at
            FROM xml_pairs
            WHERE created_at < ?
        """, (cutoff_str,))
        
        old_pairs = cursor.fetchall()
        
        if not old_pairs:
            print(f"✅ {days}일 이전 결과가 없습니다.")
            return
        
        print(f"🗑️ {days}일 이전 결과 {len(old_pairs)}개 발견:")
        for pair_id, name, created_at in old_pairs:
            print(f"   - {pair_id}: {name} ({created_at})")
        
        response = input("\n정말 삭제하시겠습니까? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("취소됨")
            return
        
        # 데이터베이스에서 삭제
        for pair_id, _, _ in old_pairs:
            cursor.execute("DELETE FROM pipeline_results WHERE pair_id = ?", (pair_id,))
            cursor.execute("DELETE FROM xml_pairs WHERE pair_id = ?", (pair_id,))
            
            # 결과 디렉토리 삭제
            result_dirs = list(self.output_dir.glob(f"{pair_id}_*"))
            for result_dir in result_dirs:
                if result_dir.is_dir():
                    import shutil
                    shutil.rmtree(result_dir)
                    print(f"   📁 삭제됨: {result_dir}")
        
        conn.commit()
        conn.close()
        
        print(f"✅ {len(old_pairs)}개 결과가 삭제되었습니다.")
    
    def _print_results(self, pair_id: str, results: Dict):
        """상세 결과 출력"""
        
        print(f"✅ 처리 완료: {pair_id}")
        print("="*50)
        
        # 안전한 처리시간 계산
        total_time = 0
        success_count = 0
        
        if 'stages' in results:
            for stage_data in results['stages'].values():
                if isinstance(stage_data, dict) and 'time' in stage_data:
                    total_time += stage_data['time']
                if isinstance(stage_data, dict) and stage_data.get('status') == 'success':
                    success_count += 1
        
        print(f"전체 처리 시간: {total_time:.2f}초")
        print(f"성공한 단계: {success_count}/{len(results)}")
        print()
        
        print("단계별 결과:")
        print("-"*50)
        
        # stages 딕셔너리가 있는 경우 출력
        if 'stages' in results:
            for stage, stage_data in results['stages'].items():
                if isinstance(stage_data, dict):
                    status = stage_data.get('status', 'unknown')
                    stage_time = stage_data.get('time', 0)
                    status_emoji = "✅" if status == 'success' else "❌"
                    
                    print(f"{status_emoji} {stage.upper()}: {status} ({stage_time:.2f}초)")
                    
                    if 'output_file' in stage_data:
                        print(f"   📄 출력: {stage_data['output_file']}")
                    
                    if 'accuracy_score' in stage_data and stage_data['accuracy_score']:
                        print(f"   🎯 정확도: {stage_data['accuracy_score']:.3f}")
                    
                    if 'error' in stage_data:
                        print(f"   ❌ 오류: {stage_data['error']}")
                    
                    print()
        
        # 결과 디렉토리 표시
        if 'result_folder' in results:
            print(f"📁 결과 디렉토리: {results['result_folder']}")
        else:
            result_dirs = list(self.output_dir.glob(f"{pair_id}_*"))
            if result_dirs:
                latest_dir = max(result_dirs, key=os.path.getmtime)
                print(f"📁 결과 디렉토리: {latest_dir}")
    
    def _print_brief_results(self, pair_id: str, results: Dict):
        """간단한 결과 출력"""
        
        # 안전한 처리시간 및 성공 카운트 계산
        total_time = 0
        success_count = 0
        total_stages = 0
        
        if 'stages' in results:
            for stage_data in results['stages'].values():
                if isinstance(stage_data, dict):
                    total_stages += 1
                    if 'time' in stage_data:
                        total_time += stage_data['time']
                    if stage_data.get('status') == 'success':
                        success_count += 1
        
        print(f"✅ {pair_id}: {success_count}/{total_stages} 성공 ({total_time:.1f}초)")
        
        # 실패한 단계가 있으면 표시
        failed_stages = []
        if 'stages' in results:
            failed_stages = [stage for stage, stage_data in results['stages'].items() 
                           if isinstance(stage_data, dict) and stage_data.get('status') != 'success']
        if failed_stages:
            print(f"   ❌ 실패: {', '.join(failed_stages)}")

    def smart_select_and_process(self, xml_dir: str, performance_mode: bool = False, 
                                   batch_size: int = 50, max_workers: int = 4):
        """대화형 파일 선택으로 XML 쌍 처리 (성능 최적화 지원)"""
        
        # 성능 최적화 활성화 시 설정 적용
        if performance_mode:
            print("🚀 성능 최적화 모드 활성화!")
            print(f"   - 배치 크기: {batch_size}")
            print(f"   - 최대 워커: {max_workers}")
            print(f"   - GPU 가속: 활성화")
            print("=" * 60)
            
            # 성능 최적화 도구 자동 실행
            try:
                from .performance_optimizer import apply_performance_settings
                apply_performance_settings(batch_size=batch_size, max_workers=max_workers)
            except ImportError:
                print("⚠️ 성능 최적화 모듈을 찾을 수 없습니다.")
        
        try:
            # 고도화된 XML 파일 브라우저 사용
            sys.path.insert(0, str(Path(__file__).parent / "utils"))
            from .xml_file_browser import XMLFileBrowser
            
            print("🐳 Docker XML Smart Pipeline - 대화형 파일 선택")
            print("=" * 60)
            
            browser = XMLFileBrowser(xml_dir)
            
            # 파일 스캔
            print("🔍 XML 파일 스캔 중...")
            pairs = browser.scan_xml_pairs()
            
            if not pairs:
                print("❌ XML 쌍을 찾을 수 없습니다.")
                return
            
            # 대화형 선택
            result = browser.interactive_select()
            
            if not result:
                print("❌ 선택이 취소되었습니다.")
                return
            
            original_file, translation_file = result
            
            # 처리 확인
            print(f"\n🎯 선택된 XML 쌍:")
            print(f"원문: {Path(original_file).name}")
            print(f"번역: {Path(translation_file).name}")
            
            confirm = input("\n이 XML 쌍을 처리하시겠습니까? (y/N): ").strip().lower()
            
            if confirm not in ['y', 'yes']:
                print("❌ 처리가 취소되었습니다.")
                return
            
            # pair_id 생성
            book_name = Path(original_file).stem
            if '원문' in book_name:
                pair_id = book_name.replace('원문', '').replace('_x-C', '').replace('-', '').strip('_')
            else:
                pair_id = book_name[:20]
            
            # 파이프라인 실행
            print(f"\n🚀 XML 파이프라인 처리 시작...")
            
            result = self.process_single_pair(
                original_xml=original_file,
                translation_xml=translation_file,
                pair_id=pair_id
            )
            
            print(f"\n🎉 처리 완료!")
            if isinstance(result, dict):
                print(f"📁 결과 위치: {result.get('result_folder', 'N/A')}")
            
        except ImportError:
            print("⚠️ 고도화된 브라우저를 사용할 수 없습니다. 기본 모드로 실행합니다.")
            self._fallback_smart_select(xml_dir)
        except Exception as e:
            print(f"❌ 스마트 선택 중 오류: {e}")
            print("기본 모드로 실행합니다.")
            self._fallback_smart_select(xml_dir)

    def process_single_xml(self, xml_file: str, pair_id: str = None, performance_mode: bool = False, 
                          batch_size: int = 50, max_workers: int = 4):
        """단일 XML 파일 처리 (원문+번역문 통합 파일)"""
        
        # 성능 최적화 활성화 시 설정 적용
        if performance_mode:
            print("🚀 성능 최적화 모드 활성화!")
            print(f"   - 배치 크기: {batch_size}")
            print(f"   - 최대 워커: {max_workers}")
            print(f"   - GPU 가속: 활성화")
            print("=" * 60)
        
        print(f"🔄 단일 XML 파일 처리 시작...")
        print(f"📄 입력 파일: {Path(xml_file).name}")
        
        # XML 구조 분석
        xml_type = self._analyze_xml_structure(xml_file)
        
        if xml_type != 'merged':
            print(f"❌ 이 XML 파일은 통합 파일이 아닙니다. (구조: {xml_type})")
            print("   원문과 번역문이 모두 포함된 XML 파일을 사용하세요.")
            return None
        
        try:
            # 단일 XML을 임시 분리 파일로 변환
            temp_dir = Path(self.output_dir) / "temp"
            temp_dir.mkdir(exist_ok=True)
            
            # pair_id 생성
            if not pair_id:
                pair_id = Path(xml_file).stem
            
            temp_orig = temp_dir / f"{pair_id}_원문.xml"
            temp_trans = temp_dir / f"{pair_id}_번역문.xml"
            
            print("🔧 단일 XML을 임시 분리 파일로 변환 중...")
            
            # XML 분리 작업 수행
            self._split_merged_xml(xml_file, str(temp_orig), str(temp_trans))
            
            # 기존 XML 파이프라인으로 처리
            print("🚀 분리된 XML로 파이프라인 실행...")
            result = self.process_single_pair(
                original_xml=str(temp_orig),
                translation_xml=str(temp_trans), 
                pair_id=pair_id
            )
            
            # 임시 파일 정리
            try:
                temp_orig.unlink()
                temp_trans.unlink()
                if not list(temp_dir.iterdir()):  # 빈 디렉토리면 삭제
                    temp_dir.rmdir()
            except:
                pass
            
            print(f"🎉 단일 XML 처리 완료!")
            return result
            
        except Exception as e:
            print(f"❌ 단일 XML 처리 중 오류: {e}")
            return None

    def _analyze_xml_structure(self, xml_file: str) -> str:
        """XML 파일 구조 분석: 통합형 vs 분리형"""
        try:
            import xml.etree.ElementTree as ET
            
            if not Path(xml_file).exists():
                return 'unknown'
                
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # 통합형: 원문과 번역문이 같은 파일에 있는 경우
            has_original = any(elem.tag in ['원문', 'original'] for elem in root.iter())
            has_translation = any(elem.tag in ['번역문', 'translation'] for elem in root.iter())
            
            if has_original and has_translation:
                return 'merged'  # 통합 XML (원문+번역문)
            else:
                return 'separate'  # 분리 XML (원문만 or 번역문만)
                
        except Exception as e:
            print(f"⚠️ XML 구조 분석 실패: {e}")
            return 'unknown'

    def _split_merged_xml(self, merged_xml: str, orig_output: str, trans_output: str):
        """통합 XML을 원문/번역문 분리 파일로 분할"""
        try:
            import xml.etree.ElementTree as ET
            import copy
            
            tree = ET.parse(merged_xml)
            root = tree.getroot()
            
            # 원문 XML 생성
            orig_root = ET.Element(root.tag)
            orig_root.attrib = root.attrib.copy()
            
            # 번역문 XML 생성  
            trans_root = ET.Element(root.tag)
            trans_root.attrib = root.attrib.copy()
            
            def deep_copy_element(source_elem):
                """ElementTree 요소의 깊은 복사"""
                new_elem = ET.Element(source_elem.tag)
                new_elem.attrib = source_elem.attrib.copy()
                new_elem.text = source_elem.text
                new_elem.tail = source_elem.tail
                
                for child in source_elem:
                    new_elem.append(deep_copy_element(child))
                
                return new_elem
            
            # 원문과 번역문 요소 분리
            for elem in root:
                if elem.tag in ['원문', 'original']:
                    # 원문 요소의 내부 단락들을 루트 레벨로 이동 (깊은 복사)
                    for child in elem:
                        orig_root.append(deep_copy_element(child))
                elif elem.tag in ['번역문', 'translation']:
                    # 번역문 요소의 내부 단락들을 루트 레벨로 이동 (깊은 복사)
                    for child in elem:
                        trans_root.append(deep_copy_element(child))
                else:
                    # 기타 요소는 둘 다에 복사 (깊은 복사)
                    orig_root.append(deep_copy_element(elem))
                    trans_root.append(deep_copy_element(elem))
            
            # 파일 저장
            orig_tree = ET.ElementTree(orig_root)
            trans_tree = ET.ElementTree(trans_root)
            
            orig_tree.write(orig_output, encoding='utf-8', xml_declaration=True)
            trans_tree.write(trans_output, encoding='utf-8', xml_declaration=True)
            
            print(f"   ✅ 원문 파일: {Path(orig_output).name}")
            print(f"   ✅ 번역문 파일: {Path(trans_output).name}")
            
        except Exception as e:
            raise Exception(f"XML 분리 실패: {e}")
    
    def _fallback_smart_select(self, xml_dir: str):
        """기본 스마트 선택 (폴백 모드)"""
        xml_path = Path(xml_dir)
        if not xml_path.is_absolute():
            xml_path = Path.cwd() / xml_dir
        
        print(f"🔍 XML 파일 검색 중: {xml_path}")
        
        if not xml_path.exists():
            print(f"❌ 디렉토리를 찾을 수 없습니다: {xml_path}")
            return
        
        # 원문 파일들 검색
        original_files = []
        for pattern in ['*원문*.txt', '*원문*.xml']:
            original_files.extend(list(xml_path.glob(pattern)))
        
        if not original_files:
            print(f"❌ 원문 파일을 찾을 수 없습니다: {xml_path}")
            return
        
        # 매칭되는 쌍들 찾기
        pairs = []
        for orig_file in original_files:
            trans_file = None
            orig_name = orig_file.stem
            
            trans_name = orig_name.replace('원문', '번역문')
            for ext in ['.txt', '.xml']:
                potential_trans = orig_file.parent / (trans_name + ext)
                if potential_trans.exists():
                    trans_file = potential_trans
                    break
            
            if trans_file:
                pair_name = orig_name.replace('_원문', '').replace('-원문', '')
                pairs.append({
                    'name': pair_name,
                    'original': str(orig_file),
                    'translation': str(trans_file)
                })
        
        if not pairs:
            print(f"❌ 매칭되는 XML 쌍을 찾을 수 없습니다.")
            return
        
        # 대화형 선택
        print(f"\n📋 발견된 XML 쌍 ({len(pairs)}개):")
        print("-" * 60)
        for i, pair in enumerate(pairs, 1):
            print(f"{i:2d}. {pair['name']}")
        print("-" * 60)
        
        while True:
            try:
                choice = input(f"\n번호를 선택하세요 (1-{len(pairs)}, 'q'=종료): ").strip()
                
                if choice.lower() == 'q':
                    print("종료됨")
                    return
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(pairs):
                    selected_pair = pairs[choice_num - 1]
                    
                    print(f"\n✅ 선택됨: {selected_pair['name']}")
                    print(f"   원문: {selected_pair['original']}")
                    print(f"   번역: {selected_pair['translation']}")
                    
                    # 처리 확인
                    confirm = input("\n처리하시겠습니까? (Y/n): ").strip()
                    if confirm.lower() in ['', 'y', 'yes']:
                        # 처리 실행
                        self.process_single_pair(
                            selected_pair['original'],
                            selected_pair['translation'],
                            selected_pair['name']
                        )
                        
                        # 계속 여부 확인
                        continue_choice = input("\n다른 쌍을 처리하시겠습니까? (Y/n): ").strip()
                        if continue_choice.lower() not in ['', 'y', 'yes']:
                            break
                    else:
                        continue
                else:
                    print(f"❌ 잘못된 번호입니다. 1-{len(pairs)} 사이의 번호를 입력하세요.")
                    
            except ValueError:
                print("❌ 숫자를 입력하세요.")
            except KeyboardInterrupt:
                print("\n\n사용자에 의해 중단됨")
                return

def main():
    """메인 CLI 함수"""
    
    parser = argparse.ArgumentParser(description="XML 파이프라인 관리 CLI")
    subparsers = parser.add_subparsers(dest='command', help='사용 가능한 명령어')
    
    # process 명령어 - 단일 쌍
    process_parser = subparsers.add_parser('process', help='단일 XML 쌍 처리')
    process_parser.add_argument('--original', required=True, help='원문 XML 파일')
    process_parser.add_argument('--translation', required=True, help='번역문 XML 파일')
    process_parser.add_argument('--pair-id', help='XML 쌍 ID (자동 생성 가능)')
    process_parser.add_argument('--output-dir', default='xml_pipeline_results', help='출력 디렉토리')
    
    # batch 명령어 - 디렉토리 일괄 처리
    batch_parser = subparsers.add_parser('batch', help='디렉토리 일괄 처리')
    batch_parser.add_argument('--xml-dir', required=True, help='XML 파일 디렉토리')
    batch_parser.add_argument('--pattern', default='*원문*.xml', help='원문 파일 패턴')
    batch_parser.add_argument('--output-dir', default='xml_pipeline_results', help='출력 디렉토리')
    
    # list 명령어 - 결과 목록
    list_parser = subparsers.add_parser('list', help='최근 처리 결과 목록')
    list_parser.add_argument('--limit', type=int, default=10, help='조회 개수')
    list_parser.add_argument('--output-dir', default='xml_pipeline_results', help='출력 디렉토리')
    
    # show 명령어 - 상세 결과
    show_parser = subparsers.add_parser('show', help='특정 XML 쌍 상세 결과')
    show_parser.add_argument('pair_id', help='XML 쌍 ID')
    show_parser.add_argument('--output-dir', default='xml_pipeline_results', help='출력 디렉토리')
    
    # cleanup 명령어 - 결과 정리
    cleanup_parser = subparsers.add_parser('cleanup', help='오래된 결과 정리')
    cleanup_parser.add_argument('--days', type=int, default=7, help='보관 기간 (일)')
    cleanup_parser.add_argument('--output-dir', default='xml_pipeline_results', help='출력 디렉토리')
    
    # smart 명령어 - 대화형 파일 선택
    smart_parser = subparsers.add_parser('smart', help='대화형 파일 선택으로 처리')
    smart_parser.add_argument('--xml-dir', default='sources', help='XML 파일 디렉토리')
    smart_parser.add_argument('--output-dir', default='xml_pipeline_results', help='출력 디렉토리')
    smart_parser.add_argument('--performance', action='store_true', help='성능 최적화 모드 활성화')
    smart_parser.add_argument('--batch-size', type=int, default=50, help='배치 크기 (성능 최적화용)')
    smart_parser.add_argument('--max-workers', type=int, default=4, help='최대 워커 수 (성능 최적화용)')

    # single 명령어 - 단일 XML 파일 처리 (원문+번역문 통합)
    single_parser = subparsers.add_parser('single', help='단일 XML 파일 처리 (원문+번역문 통합)')
    single_parser.add_argument('xml_file', help='처리할 단일 XML 파일')
    single_parser.add_argument('--output-dir', default='xml_pipeline_results', help='출력 디렉토리')
    single_parser.add_argument('--performance', action='store_true', help='성능 최적화 모드 활성화')
    single_parser.add_argument('--batch-size', type=int, default=50, help='배치 크기 (성능 최적화용)')
    single_parser.add_argument('--max-workers', type=int, default=4, help='최대 워커 수 (성능 최적화용)')
    single_parser.add_argument('--pair-id', help='XML 쌍 식별자 (미지정시 파일명에서 자동생성)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = XMLPipelineManager(args.output_dir)
    
    try:
        if args.command == 'process':
            # 단일 XML 쌍 처리
            manager.process_single_pair(
                args.original,
                args.translation,
                args.pair_id
            )
        
        elif args.command == 'batch':
            # 디렉토리 일괄 처리
            manager.process_directory(args.xml_dir, args.pattern)
        
        elif args.command == 'list':
            # 결과 목록
            manager.list_recent_results(args.limit)
        
        elif args.command == 'show':
            # 상세 결과
            manager.show_pair_details(args.pair_id)
        
        elif args.command == 'cleanup':
            # 결과 정리
            manager.cleanup_old_results(args.days)
        
        elif args.command == 'smart':
            # 대화형 파일 선택 + 성능 최적화
            performance_settings = {
                'performance_mode': getattr(args, 'performance', False),
                'batch_size': getattr(args, 'batch_size', 50),
                'max_workers': getattr(args, 'max_workers', 4)
            }
            manager.smart_select_and_process(args.xml_dir, **performance_settings)
        
        elif args.command == 'single':
            # 단일 XML 파일 처리
            performance_settings = {
                'performance_mode': getattr(args, 'performance', False),
                'batch_size': getattr(args, 'batch_size', 50),
                'max_workers': getattr(args, 'max_workers', 4)
            }
            result = manager.process_single_xml(
                xml_file=args.xml_file,
                pair_id=getattr(args, 'pair_id', None),
                **performance_settings
            )
            if result:
                print(f"📁 결과 위치: {result.get('result_folder', 'N/A')}")
    
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨")
    except Exception as e:
        print(f"❌ 오류: {e}")

if __name__ == "__main__":
    main()