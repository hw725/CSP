#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
통합 전체서종 누적분석기 v3.0 (정리된 버전)
현재까지 완료된 모든 책을 포괄하고, 앞으로 추가되는 책도 자동 누적 분석

작성자: AI Assistant
수정일: 2025년 1월
"""

import os
import json
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# 작가/역자 정보 추출기 추가
from book_metadata_extractor import BookMetadataExtractor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cumulative_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CumulativeBookAnalyzer:
    """누적 도서 분석기"""
    
    def __init__(self, base_dir: str = "xml_pipeline_results", preserve_manual_csv: bool = True):
        # analytics 디렉토리에서 실행되므로 상위 디렉토리 참조
        self.base_dir = Path("..") / base_dir
        self.db_path = "cumulative_analysis.db"
        self.preserve_manual_csv = preserve_manual_csv
        
        # CSV 파일 경로 설정 (수동 편집 보호)
        if preserve_manual_csv and Path("cumulative_analysis_results_manual.csv").exists():
            self.csv_export_path = f"cumulative_analysis_results_auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            print(f"⚠️  수동 편집된 CSV 파일을 보호하기 위해 새로운 파일로 저장합니다: {self.csv_export_path}")
        else:
            self.csv_export_path = "cumulative_analysis_results.csv"
            
        self.metadata_extractor = BookMetadataExtractor()  # 작가/역자 정보 추출기
        self._init_database()
        logger.info(f"누적 분석기 초기화 완료 - 기본 디렉토리: {self.base_dir}")
    
    def _init_database(self):
        """SQLite 데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS book_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_name TEXT NOT NULL UNIQUE,
                    author TEXT,
                    translator TEXT,
                    analysis_date TEXT NOT NULL,
                    total_paragraphs INTEGER,
                    pa_accuracy REAL,
                    sa_accuracy REAL,
                    embedding_similarity_avg REAL,
                    processing_time_seconds REAL,
                    quality_grade TEXT,
                    notes TEXT,
                    global_source_similarity REAL,
                    global_target_similarity REAL,
                    phrase_count INTEGER,
                    sa_result_count INTEGER,
                    length_accuracy REAL,
                    -- PA 세부 메트릭
                    pa_precision REAL,
                    pa_recall REAL,
                    pa_f1_score REAL,
                    pa_avg_similarity REAL,
                    pa_combined_similarity REAL,
                    -- SA 세부 메트릭
                    sa_precision REAL,
                    sa_recall REAL,
                    sa_f1_score REAL,
                    sa_set_similarity REAL,
                    sa_source_similarity REAL,
                    sa_target_similarity REAL,
                    sibu_classification TEXT,
                    period TEXT
                )
            ''')
            
            # 기존 테이블에 author, translator 컬럼이 없으면 추가
            cursor.execute("PRAGMA table_info(book_analysis)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'author' not in columns:
                cursor.execute("ALTER TABLE book_analysis ADD COLUMN author TEXT")
                logger.info("author 컬럼 추가됨")
                
            if 'translator' not in columns:
                cursor.execute("ALTER TABLE book_analysis ADD COLUMN translator TEXT")
                logger.info("translator 컬럼 추가됨")
                
            if 'sibu_classification' not in columns:
                cursor.execute("ALTER TABLE book_analysis ADD COLUMN sibu_classification TEXT")
                logger.info("sibu_classification 컬럼 추가됨")
                
            if 'period' not in columns:
                cursor.execute("ALTER TABLE book_analysis ADD COLUMN period TEXT")
                logger.info("period 컬럼 추가됨")
            
            conn.commit()
            conn.close()
            logger.info("데이터베이스 초기화 완료")
            
        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {e}")
            raise
    
    def scan_for_results(self):
        """결과 폴더들을 자동 스캔"""
        result_folders = []
        
        if not self.base_dir.exists():
            logger.warning(f"기본 디렉토리가 존재하지 않습니다: {self.base_dir}")
            return result_folders
        
        for item in self.base_dir.iterdir():
            if item.is_dir():
                accuracy_file = item / "accuracy_report.json"
                accuracy_file_in_folder = item / "accuracy" / "accuracy_report.json"
                
                if accuracy_file.exists() or accuracy_file_in_folder.exists():
                    result_folders.append(item)
                    logger.info(f"유효한 결과 폴더 발견: {item.name}")
        
        logger.info(f"총 {len(result_folders)}개의 결과 폴더 발견")
        return sorted(result_folders)
    
    def extract_book_data(self, result_folder: Path):
        """개별 책의 분석 데이터 추출"""
        try:
            book_name = result_folder.name
            
            # accuracy_report.json 읽기 (두 위치 모두 확인)
            accuracy_file = result_folder / "accuracy_report.json"
            accuracy_file_in_folder = result_folder / "accuracy" / "accuracy_report.json"
            
            if accuracy_file.exists():
                target_file = accuracy_file
            elif accuracy_file_in_folder.exists():
                target_file = accuracy_file_in_folder
            else:
                logger.warning(f"정확도 보고서가 없습니다: {book_name}")
                return None
            
            with open(target_file, 'r', encoding='utf-8') as f:
                accuracy_data = json.load(f)
            
            # xml_level_similarity.json에서 PA/SA 분석 결과 읽기
            xml_similarity_file = result_folder / "accuracy" / "xml_level_similarity.json"
            pa_analysis = {}
            sa_analysis = {}
            
            if xml_similarity_file.exists():
                try:
                    with open(xml_similarity_file, 'r', encoding='utf-8') as f:
                        xml_similarity_data = json.load(f)
                        pa_analysis = xml_similarity_data.get('pa_analysis', {})
                        sa_analysis = xml_similarity_data.get('sa_analysis', {})
                except Exception as e:
                    logger.warning(f"XML 레벨 유사도 파일 읽기 오류 ({book_name}): {e}")
            
            # 기본 메트릭 추출
            comparison_results = accuracy_data.get('comparison_results', {})
            global_integrity = accuracy_data.get('global_integrity', {})
            
            # XML 레벨 분석에서 PA/SA 세부 메트릭 추출
            xml_level_analysis = accuracy_data.get('xml_level_analysis', {})
            pa_analysis = xml_level_analysis.get('pa_analysis', {})
            sa_analysis = xml_level_analysis.get('sa_analysis', {})
            
            # PA 정확도 계산: xml_level_similarity.json의 실제 PA 분석 결과 사용
            pa_f1_score = pa_analysis.get('f1_score', 0.0)
            pa_avg_similarity = pa_analysis.get('avg_similarity', 0.0)
            # PA 정확도 = F1 Score(60%) + 평균 유사도(40%)
            calculated_pa_accuracy = pa_f1_score * 0.6 + pa_avg_similarity * 0.4
            
            # SA 정확도 계산: xml_level_similarity.json의 실제 SA 분석 결과 사용
            sa_f1_score = sa_analysis.get('f1_score', 0.0)
            sa_avg_similarity = sa_analysis.get('avg_similarity', 0.0)
            # SA 정확도 = F1 Score(60%) + 평균 유사도(40%) 
            calculated_sa_accuracy = sa_f1_score * 0.6 + sa_avg_similarity * 0.4
            
            # 작가/역자 정보 추출
            author, translator = self.metadata_extractor.extract_author_translator(book_name)
            
            # 4부분류 정보 추출
            sibu_classification = self.metadata_extractor.get_sibu_classification(book_name)
            
            # 시대 정보 추출
            period = self.metadata_extractor.get_period_from_author(author)
            
            book_data = {
                'book_name': book_name,
                'author': author,
                'translator': translator,
                'analysis_date': datetime.now().isoformat(),
                'total_paragraphs': accuracy_data.get('총_문단수', 
                    comparison_results.get('xml_phrase_count', 0)),
                'pa_accuracy': accuracy_data.get('PA_정확도', calculated_pa_accuracy),
                'sa_accuracy': accuracy_data.get('SA_정확도', calculated_sa_accuracy),
                'embedding_similarity_avg': accuracy_data.get('임베딩_유사도_평균', 
                    global_integrity.get('global_target_text_similarity', 0.0)),
                'processing_time_seconds': accuracy_data.get('처리_시간_초', 0.0),
                # 세부 항목
                'global_source_similarity': global_integrity.get('global_source_text_similarity', 0.0),
                'global_target_similarity': global_integrity.get('global_target_text_similarity', 0.0),
                'phrase_count': comparison_results.get('xml_phrase_count', 0),
                'sa_result_count': comparison_results.get('sa_result_count', 0),
                'length_accuracy': comparison_results.get('length_based_accuracy', 0.0),
                # PA 세부 메트릭
                'pa_precision': pa_analysis.get('precision', 0.0),
                'pa_recall': pa_analysis.get('recall', 0.0),
                'pa_f1_score': pa_analysis.get('f1_score', 0.0),
                'pa_avg_similarity': pa_analysis.get('avg_similarity', 0.0),
                'pa_combined_similarity': pa_analysis.get('avg_combined_similarity', 0.0),
                # SA 세부 메트릭
                'sa_precision': sa_analysis.get('precision', 0.0),
                'sa_recall': sa_analysis.get('recall', 0.0),
                'sa_f1_score': sa_analysis.get('f1_score', 0.0),
                'sa_set_similarity': sa_analysis.get('avg_combined_similarity', 0.0),
                'sa_source_similarity': sa_analysis.get('avg_original_similarity', 0.0),
                'sa_target_similarity': sa_analysis.get('avg_translation_similarity', 0.0),
                'sibu_classification': sibu_classification,
                'period': period,
            }
            
            # 품질 등급 계산
            book_data['quality_grade'] = self._calculate_quality_grade(
                book_data['pa_accuracy'],
                book_data['sa_accuracy'],
                book_data['embedding_similarity_avg']
            )
            
            # 추가 메모
            notes = []
            if book_data['total_paragraphs'] > 0:
                notes.append(f"총 {book_data['total_paragraphs']}개 문단 처리")
            if book_data['processing_time_seconds'] > 0:
                notes.append(f"처리시간: {book_data['processing_time_seconds']:.1f}초")
            
            book_data['notes'] = "; ".join(notes)
            
            logger.info(f"{book_name} 데이터 추출 완료 - 품질등급: {book_data['quality_grade']}")
            return book_data
            
        except Exception as e:
            logger.error(f"{result_folder.name} 데이터 추출 실패: {e}")
            return None
    
    def _calculate_quality_grade(self, pa_acc: float, sa_acc: float, embed_sim: float):
        """
        품질 등급 계산 (2025년 9월 간소화된 공식)
        
        계산 공식: (PA정확도 × 0.5 + SA정확도 × 0.5) × 100
        - PA정확도: 문단 정렬 정확도 (50% 가중치) - F1 Score(60%) + 평균유사도(40%)
        - SA정확도: 구 정렬 정확도 (50% 가중치) - F1 Score(60%) + 평균유사도(40%)
        
        개선 사항:
        • 핵심 지표 집중: PA/SA 성능만으로 품질 평가
        • SA 분석: XML <s> 태그 기반 문장별 그룹핑으로 0% → 60%+ 개선
        • 텍스트 정제: 괄호([]) 및 하이픈(-) 일관 제거로 매칭 정확도 향상
        
        등급 기준 (현실적 조정):
        A+ (85-100): 최고 품질 | A (75-84): 우수 | B+ (65-74): 양호+
        B (60-64): 양호 | C+ (55-59): 보통+ | C (50-54): 보통
        D (45-49): 부족 | F (0-44): 불량
        """
        total_score = (pa_acc * 0.5 + sa_acc * 0.5) * 100
        
        if total_score >= 85:
            return "A+"
        elif total_score >= 75:
            return "A"
        elif total_score >= 65:
            return "B+"
        elif total_score >= 60:
            return "B"
        elif total_score >= 55:
            return "C+"
        elif total_score >= 50:
            return "C"
        elif total_score >= 45:
            return "D"
        else:
            return "F"
    
    def store_analysis_data(self, book_data):
        """분석 데이터를 데이터베이스에 저장 (중복 제거)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 기존 데이터 확인
            cursor.execute('SELECT COUNT(*) FROM book_analysis WHERE book_name = ?', (book_data['book_name'],))
            exists = cursor.fetchone()[0] > 0
            
            if exists:
                # 기존 데이터 업데이트
                cursor.execute('''
                    UPDATE book_analysis SET
                    author = ?, translator = ?, analysis_date = ?, total_paragraphs = ?, pa_accuracy = ?, sa_accuracy = ?, 
                    embedding_similarity_avg = ?, processing_time_seconds = ?, quality_grade = ?, notes = ?,
                    global_source_similarity = ?, global_target_similarity = ?, phrase_count = ?, 
                    sa_result_count = ?, length_accuracy = ?,
                    pa_precision = ?, pa_recall = ?, pa_f1_score = ?, pa_avg_similarity = ?, pa_combined_similarity = ?,
                    sa_precision = ?, sa_recall = ?, sa_f1_score = ?, sa_set_similarity = ?, sa_source_similarity = ?, sa_target_similarity = ?
                    WHERE book_name = ?
                ''', (
                    book_data['author'],
                    book_data['translator'],
                    book_data['analysis_date'],
                    book_data['total_paragraphs'],
                    book_data['pa_accuracy'],
                    book_data['sa_accuracy'],
                    book_data['embedding_similarity_avg'],
                    book_data['processing_time_seconds'],
                    book_data['quality_grade'],
                    book_data['notes'],
                    book_data['global_source_similarity'],
                    book_data['global_target_similarity'],
                    book_data['phrase_count'],
                    book_data['sa_result_count'],
                    book_data['length_accuracy'],
                    book_data['pa_precision'],
                    book_data['pa_recall'],
                    book_data['pa_f1_score'],
                    book_data['pa_avg_similarity'],
                    book_data['pa_combined_similarity'],
                    book_data['sa_precision'],
                    book_data['sa_recall'],
                    book_data['sa_f1_score'],
                    book_data['sa_set_similarity'],
                    book_data['sa_source_similarity'],
                    book_data['sa_target_similarity'],
                    book_data['book_name']
                ))
            else:
                # 새 데이터 삽입
                cursor.execute('''
                    INSERT INTO book_analysis 
                    (book_name, author, translator, analysis_date, total_paragraphs, pa_accuracy, sa_accuracy, 
                     embedding_similarity_avg, processing_time_seconds, quality_grade, notes,
                     global_source_similarity, global_target_similarity, phrase_count, 
                     sa_result_count, length_accuracy,
                     pa_precision, pa_recall, pa_f1_score, pa_avg_similarity, pa_combined_similarity,
                     sa_precision, sa_recall, sa_f1_score, sa_set_similarity, sa_source_similarity, sa_target_similarity,
                     sibu_classification, period)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    book_data['book_name'],
                    book_data['author'],
                    book_data['translator'],
                    book_data['analysis_date'],
                    book_data['total_paragraphs'],
                    book_data['pa_accuracy'],
                    book_data['sa_accuracy'],
                    book_data['embedding_similarity_avg'],
                    book_data['processing_time_seconds'],
                    book_data['quality_grade'],
                    book_data['notes'],
                    book_data['global_source_similarity'],
                    book_data['global_target_similarity'],
                    book_data['phrase_count'],
                    book_data['sa_result_count'],
                    book_data['length_accuracy'],
                    book_data['pa_precision'],
                    book_data['pa_recall'],
                    book_data['pa_f1_score'],
                    book_data['pa_avg_similarity'],
                    book_data['pa_combined_similarity'],
                    book_data['sa_precision'],
                    book_data['sa_recall'],
                    book_data['sa_f1_score'],
                    book_data['sa_set_similarity'],
                    book_data['sa_source_similarity'],
                    book_data['sa_target_similarity'],
                    book_data['sibu_classification'],
                    book_data['period']
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"{book_data['book_name']} 데이터 저장 완료")
            return True
            
        except Exception as e:
            logger.error(f"데이터 저장 실패 ({book_data['book_name']}): {e}")
            return False
    
    def generate_cumulative_report(self):
        """누적 분석 보고서 생성"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            df = pd.read_sql_query('SELECT * FROM book_analysis ORDER BY analysis_date DESC', conn)
            
            if df.empty:
                logger.warning("분석 데이터가 없습니다")
                return {}
            
            # 통계 계산
            report = {
                'generated_at': datetime.now().isoformat(),
                'total_books': len(df),
                'total_paragraphs': int(df['total_paragraphs'].sum()),
                'average_pa_accuracy': float(df['pa_accuracy'].mean()),
                'average_sa_accuracy': float(df['sa_accuracy'].mean()),
                'average_embedding_similarity': float(df['embedding_similarity_avg'].mean()),
                'total_processing_time': float(df['processing_time_seconds'].sum()),
                'quality_distribution': df['quality_grade'].value_counts().to_dict(),
                'books_summary': []
            }
            
            # 개별 책 요약 (세부 메트릭 포함)
            for _, row in df.iterrows():
                book_summary = {
                    'book_name': row['book_name'],
                    'quality_grade': row['quality_grade'],
                    'pa_accuracy': float(row['pa_accuracy']),
                    'sa_accuracy': float(row['sa_accuracy']),
                    'total_paragraphs': int(row['total_paragraphs']),
                    'analysis_date': row['analysis_date'],
                    # PA 세부 메트릭
                    'pa_details': {
                        'precision': float(row['pa_precision']) if pd.notna(row['pa_precision']) else 0.0,
                        'recall': float(row['pa_recall']) if pd.notna(row['pa_recall']) else 0.0,
                        'f1_score': float(row['pa_f1_score']) if pd.notna(row['pa_f1_score']) else 0.0,
                        'avg_similarity': float(row['pa_avg_similarity']) if pd.notna(row['pa_avg_similarity']) else 0.0,
                        'combined_similarity': float(row['pa_combined_similarity']) if pd.notna(row['pa_combined_similarity']) else 0.0
                    },
                    # SA 세부 메트릭
                    'sa_details': {
                        'precision': float(row['sa_precision']) if pd.notna(row['sa_precision']) else 0.0,
                        'recall': float(row['sa_recall']) if pd.notna(row['sa_recall']) else 0.0,
                        'f1_score': float(row['sa_f1_score']) if pd.notna(row['sa_f1_score']) else 0.0,
                        'set_similarity': float(row['sa_set_similarity']) if pd.notna(row['sa_set_similarity']) else 0.0,
                        'source_similarity': float(row['sa_source_similarity']) if pd.notna(row['sa_source_similarity']) else 0.0,
                        'target_similarity': float(row['sa_target_similarity']) if pd.notna(row['sa_target_similarity']) else 0.0
                    }
                }
                report['books_summary'].append(book_summary)
            
            conn.close()
            
            logger.info(f"누적 보고서 생성 완료 - 총 {report['total_books']}권 분석")
            return report
            
        except Exception as e:
            logger.error(f"누적 보고서 생성 실패: {e}")
            return {}
    
    def export_to_csv(self):
        """분석 결과를 CSV로 내보내기"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            df = pd.read_sql_query('''
                SELECT book_name as '책명',
                       author as '작가',
                       translator as '역자',
                       quality_grade as '품질등급',
                       total_paragraphs as '총문단수',
                       phrase_count as '원문구수',
                       sa_result_count as '번역구수',
                       ROUND(length_accuracy * 100, 2) as '길이정확도(%)',
                       ROUND(global_source_similarity * 100, 2) as '원문유사도(%)',
                       ROUND(global_target_similarity * 100, 2) as '번역유사도(%)',
                       ROUND(pa_accuracy * 100, 2) as 'PA정확도(%)',
                       ROUND(sa_accuracy * 100, 2) as 'SA정확도(%)',
                       ROUND(embedding_similarity_avg * 100, 2) as '임베딩유사도(%)',
                       ROUND(pa_precision * 100, 2) as 'PA_Precision(%)',
                       ROUND(pa_recall * 100, 2) as 'PA_Recall(%)',
                       ROUND(pa_f1_score * 100, 2) as 'PA_F1Score(%)',
                       ROUND(pa_avg_similarity * 100, 2) as 'PA_평균유사도(%)',
                       ROUND(sa_precision * 100, 2) as 'SA_Precision(%)',
                       ROUND(sa_recall * 100, 2) as 'SA_Recall(%)',
                       ROUND(sa_f1_score * 100, 2) as 'SA_F1Score(%)',
                       ROUND(sa_set_similarity * 100, 2) as 'SA_한세트유사도(%)',
                       ROUND(sa_source_similarity * 100, 2) as 'SA_원문유사도(%)',
                       ROUND(sa_target_similarity * 100, 2) as 'SA_번역문유사도(%)',
                       ROUND(processing_time_seconds, 1) as '처리시간(초)',
                       analysis_date as '분석일시',
                       sibu_classification as '4부분류',
                       period as '시대'
                FROM book_analysis 
                ORDER BY quality_grade, pa_accuracy DESC
            ''', conn)
            
            conn.close()
            
            df.to_csv(self.csv_export_path, index=False, encoding='utf-8-sig')
            
            logger.info(f"CSV 내보내기 완료: {self.csv_export_path}")
            
            # 수동 편집된 CSV가 있다면 병합/동기화 완료 안내
            manual_csv_path = "cumulative_analysis_results_manual.csv"
            if Path(manual_csv_path).exists() and self.csv_export_path != "cumulative_analysis_results.csv":
                print(f"\n📝 수동 편집된 CSV: {manual_csv_path} (자동 동기화 완료)")
                print(f"💾 최신 자동 생성 파일: {self.csv_export_path}")
                print("ℹ️  수동 편집 내용이 DB와 모든 분석 도구에 반영되었습니다.")
            
            return True
            
        except Exception as e:
            logger.error(f"CSV 내보내기 실패: {e}")
            return False

    def merge_manual_csv(self, manual_csv_path: str = "cumulative_analysis_results_manual.csv"):
        """수동 편집된 CSV와 새로운 데이터를 병합"""
        try:
            if not Path(manual_csv_path).exists():
                logger.warning(f"수동 편집 CSV 파일이 존재하지 않습니다: {manual_csv_path}")
                return False
            
            # 수동 편집된 CSV 읽기
            manual_df = pd.read_csv(manual_csv_path, encoding='utf-8-sig')
            
            # 새로운 자동 생성 CSV 읽기
            auto_df = pd.read_csv(self.csv_export_path, encoding='utf-8-sig')
            
            # 병합 전략: 수동 편집된 것을 우선하고, 새로운 책만 추가
            manual_books = set(manual_df['책명'].values)
            new_books_df = auto_df[~auto_df['책명'].isin(manual_books)]
            
            if not new_books_df.empty:
                # 새로운 책이 있으면 병합
                merged_df = pd.concat([manual_df, new_books_df], ignore_index=True)
                
                # 백업 생성
                backup_path = f"{manual_csv_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                manual_df.to_csv(backup_path, index=False, encoding='utf-8-sig')
                
                # 병합된 결과 저장
                merged_df.to_csv(manual_csv_path, index=False, encoding='utf-8-sig')
                
                print(f"✅ CSV 병합 완료!")
                print(f"📁 기존 수동 편집 파일 백업: {backup_path}")
                print(f"📁 병합된 파일: {manual_csv_path}")
                print(f"📊 새로 추가된 책: {len(new_books_df)}권")
                
                # 새로 추가된 책 목록 출력
                if len(new_books_df) <= 10:
                    print("새로 추가된 책 목록:")
                    for book in new_books_df['책명'].values:
                        print(f"  - {book}")
                
                return True
            else:
                print("ℹ️  새로 추가된 책이 없습니다. 병합할 내용이 없습니다.")
                return True
                
        except Exception as e:
            logger.error(f"CSV 병합 실패: {e}")
            return False

    def sync_manual_csv_to_db(self, manual_csv_path: str = "cumulative_analysis_results_manual.csv"):
        """수동 편집된 CSV 내용을 데이터베이스에 반영"""
        try:
            if not Path(manual_csv_path).exists():
                logger.warning(f"수동 편집 CSV 파일이 존재하지 않습니다: {manual_csv_path}")
                return False
            
            print(f"🔄 수동 편집된 CSV를 DB에 반영 중: {manual_csv_path}")
            
            # 수동 편집된 CSV 읽기
            manual_df = pd.read_csv(manual_csv_path, encoding='utf-8-sig')
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            updated_count = 0
            for _, row in manual_df.iterrows():
                # CSV 컬럼명을 DB 컬럼명으로 매핑
                book_data = {
                    'book_name': row['책명'],
                    'author': row['작가'] if '작가' in row else '미상',
                    'translator': row['역자'] if '역자' in row else '한국고전번역원',
                    'quality_grade': row['품질등급'],
                    'total_paragraphs': int(row['총문단수']) if pd.notna(row['총문단수']) else 0,
                    'phrase_count': int(row['원문구수']) if pd.notna(row['원문구수']) else 0,
                    'sa_result_count': int(row['번역구수']) if pd.notna(row['번역구수']) else 0,
                    'length_accuracy': float(row['길이정확도(%)']) / 100 if pd.notna(row['길이정확도(%)']) else 0.0,
                    'global_source_similarity': float(row['원문유사도(%)']) / 100 if pd.notna(row['원문유사도(%)']) else 0.0,
                    'global_target_similarity': float(row['번역유사도(%)']) / 100 if pd.notna(row['번역유사도(%)']) else 0.0,
                    'pa_accuracy': float(row['PA정확도(%)']) / 100 if pd.notna(row['PA정확도(%)']) else 0.0,
                    'sa_accuracy': float(row['SA정확도(%)']) / 100 if pd.notna(row['SA정확도(%)']) else 0.0,
                    'embedding_similarity_avg': float(row['임베딩유사도(%)']) / 100 if pd.notna(row['임베딩유사도(%)']) else 0.0,
                    'pa_precision': float(row['PA_Precision(%)']) / 100 if 'PA_Precision(%)' in row and pd.notna(row['PA_Precision(%)']) else 0.0,
                    'pa_recall': float(row['PA_Recall(%)']) / 100 if 'PA_Recall(%)' in row and pd.notna(row['PA_Recall(%)']) else 0.0,
                    'pa_f1_score': float(row['PA_F1Score(%)']) / 100 if 'PA_F1Score(%)' in row and pd.notna(row['PA_F1Score(%)']) else 0.0,
                    'pa_avg_similarity': float(row['PA_평균유사도(%)']) / 100 if 'PA_평균유사도(%)' in row and pd.notna(row['PA_평균유사도(%)']) else 0.0,
                    'sa_precision': float(row['SA_Precision(%)']) / 100 if 'SA_Precision(%)' in row and pd.notna(row['SA_Precision(%)']) else 0.0,
                    'sa_recall': float(row['SA_Recall(%)']) / 100 if 'SA_Recall(%)' in row and pd.notna(row['SA_Recall(%)']) else 0.0,
                    'sa_f1_score': float(row['SA_F1Score(%)']) / 100 if 'SA_F1Score(%)' in row and pd.notna(row['SA_F1Score(%)']) else 0.0,
                    'sa_set_similarity': float(row['SA_한세트유사도(%)']) / 100 if 'SA_한세트유사도(%)' in row and pd.notna(row['SA_한세트유사도(%)']) else 0.0,
                    'sa_source_similarity': float(row['SA_원문유사도(%)']) / 100 if 'SA_원문유사도(%)' in row and pd.notna(row['SA_원문유사도(%)']) else 0.0,
                    'sa_target_similarity': float(row['SA_번역문유사도(%)']) / 100 if 'SA_번역문유사도(%)' in row and pd.notna(row['SA_번역문유사도(%)']) else 0.0,
                    'processing_time_seconds': float(row['처리시간(초)']) if '처리시간(초)' in row and pd.notna(row['처리시간(초)']) else 0.0,
                    'analysis_date': row['분석일시'] if '분석일시' in row else datetime.now().isoformat(),
                    'notes': f"수동 편집됨 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    'sibu_classification': row['4부분류'] if '4부분류' in row and pd.notna(row['4부분류']) else self.metadata_extractor.get_sibu_classification(row['책명']),
                    'period': row['시대'] if '시대' in row and pd.notna(row['시대']) else self.metadata_extractor.get_period_from_author(row['작가'] if '작가' in row and pd.notna(row['작가']) else '미상')
                }
                
                # 기존 데이터 확인 후 업데이트
                cursor.execute('SELECT COUNT(*) FROM book_analysis WHERE book_name = ?', (book_data['book_name'],))
                exists = cursor.fetchone()[0] > 0
                
                if exists:
                    # 기존 데이터 업데이트 (수동 편집 우선)
                    cursor.execute('''
                        UPDATE book_analysis SET
                        author = ?, translator = ?, quality_grade = ?, total_paragraphs = ?, 
                        phrase_count = ?, sa_result_count = ?, length_accuracy = ?,
                        global_source_similarity = ?, global_target_similarity = ?,
                        pa_accuracy = ?, sa_accuracy = ?, embedding_similarity_avg = ?,
                        pa_precision = ?, pa_recall = ?, pa_f1_score = ?, pa_avg_similarity = ?,
                        sa_precision = ?, sa_recall = ?, sa_f1_score = ?, sa_set_similarity = ?,
                        sa_source_similarity = ?, sa_target_similarity = ?,
                        processing_time_seconds = ?, analysis_date = ?, notes = ?, sibu_classification = ?, period = ?
                        WHERE book_name = ?
                    ''', (
                        book_data['author'], book_data['translator'], book_data['quality_grade'], 
                        book_data['total_paragraphs'], book_data['phrase_count'], book_data['sa_result_count'],
                        book_data['length_accuracy'], book_data['global_source_similarity'], 
                        book_data['global_target_similarity'], book_data['pa_accuracy'], book_data['sa_accuracy'],
                        book_data['embedding_similarity_avg'], book_data['pa_precision'], book_data['pa_recall'],
                        book_data['pa_f1_score'], book_data['pa_avg_similarity'], book_data['sa_precision'],
                        book_data['sa_recall'], book_data['sa_f1_score'], book_data['sa_set_similarity'],
                        book_data['sa_source_similarity'], book_data['sa_target_similarity'],
                        book_data['processing_time_seconds'], book_data['analysis_date'], book_data['notes'],
                        book_data['sibu_classification'], book_data['period'], book_data['book_name']
                    ))
                    updated_count += 1
                else:
                    logger.warning(f"DB에 존재하지 않는 책: {book_data['book_name']} (수동 추가된 항목일 수 있음)")
            
            conn.commit()
            conn.close()
            
            print(f"✅ DB 동기화 완료: {updated_count}개 항목 업데이트됨")
            logger.info(f"수동 CSV → DB 동기화 완료: {updated_count}개 항목")
            return True
            
        except Exception as e:
            logger.error(f"수동 CSV → DB 동기화 실패: {e}")
            return False

    def create_manual_csv_template(self):
        """수동 편집용 CSV 템플릿 생성"""
        manual_csv_path = "cumulative_analysis_results_manual.csv"
        
        if Path(manual_csv_path).exists():
            print(f"⚠️  {manual_csv_path} 파일이 이미 존재합니다.")
            return False
        
        # 현재 CSV를 수동 편집용으로 복사
        if Path(self.csv_export_path).exists():
            import shutil
            shutil.copy(self.csv_export_path, manual_csv_path)
            print(f"📝 수동 편집용 CSV 템플릿 생성: {manual_csv_path}")
            print("💡 이 파일을 편집하시면 다음 분석 시 자동으로 보호됩니다.")
            return True
        else:
            logger.error("원본 CSV 파일이 존재하지 않습니다.")
            return False
    
    def run_full_analysis(self):
        """전체 누적 분석 실행"""
        try:
            logger.info("=== 전체 누적 분석 시작 ===")
            
            # 1. 결과 폴더 스캔
            result_folders = self.scan_for_results()
            
            if not result_folders:
                logger.warning("처리할 결과 폴더가 없습니다")
                return False
            
            # 2. 각 책 데이터 추출 및 저장
            success_count = 0
            for folder in result_folders:
                book_data = self.extract_book_data(folder)
                if book_data and self.store_analysis_data(book_data):
                    success_count += 1
            
            logger.info(f"총 {len(result_folders)}개 중 {success_count}개 책 처리 완료")
            
            # 3. 누적 보고서 생성
            report = self.generate_cumulative_report()
            if report:
                with open('cumulative_analysis_report.json', 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                logger.info("누적 분석 보고서 저장 완료: cumulative_analysis_report.json")
            
            # 4. CSV 내보내기
            self.export_to_csv()
            
            # 5. 수동 편집된 CSV가 있다면 DB에 동기화
            manual_csv_path = "cumulative_analysis_results_manual.csv"
            if Path(manual_csv_path).exists():
                print(f"\n🔄 수동 편집된 CSV 발견: {manual_csv_path}")
                print("💡 DB에 동기화를 진행합니다...")
                if self.sync_manual_csv_to_db():
                    print("✅ 수동 편집 내용이 DB에 반영되었습니다!")
                    
                    # 동기화 후 최종 CSV 다시 생성
                    print("📊 동기화된 내용으로 최종 CSV를 재생성합니다...")
                    self.export_to_csv()
                else:
                    print("⚠️ DB 동기화에 실패했습니다.")
            
            logger.info("=== 전체 누적 분석 완료 ===")
            return True
            
        except Exception as e:
            logger.error(f"전체 누적 분석 실패: {e}")
            return False


def main():
    """메인 함수"""
    print("📚 통합 전체서종 누적분석기 v3.0 (수동 편집 보호)")
    print("=" * 60)
    
    try:
        analyzer = CumulativeBookAnalyzer()
        success = analyzer.run_full_analysis()
        
        if success:
            print("\n✅ 누적 분석이 성공적으로 완료되었습니다!")
            print(f"📊 결과 파일:")
            print(f"   - 데이터베이스: cumulative_analysis.db")
            print(f"   - JSON 보고서: cumulative_analysis_report.json")
            print(f"   - CSV 내보내기: {analyzer.csv_export_path}")
            print(f"   - 로그 파일: cumulative_analysis.log")
            
            # 수동 편집 가이드 (동기화 상태에 따라 다른 메시지)
            manual_csv_path = "cumulative_analysis_results_manual.csv"
            if Path(manual_csv_path).exists():
                print("\n💡 수동 편집 시스템 상태:")
                print("   ✅ 수동 편집된 CSV가 감지되어 DB에 자동 동기화되었습니다!")
                print("   ✅ 대시보드와 클러스터링 분석이 동일한 데이터를 사용합니다.")
                print("\n� 추가 수동 편집 시:")
                print("   1. 'cumulative_analysis_results_manual.csv' 파일을 편집")
                print("   2. 이 스크립트를 다시 실행 (자동 동기화됨)")
            else:
                print("\n💡 수동 편집 가이드:")
                print("   1. 수동 편집용 템플릿 생성: analyzer.create_manual_csv_template()")
                print("   2. 'cumulative_analysis_results_manual.csv' 파일을 편집")
                print("   3. 이 스크립트를 다시 실행하면 자동으로 동기화됩니다!")
                print("\n🔧 고급 사용법:")
                print("   - 개별 동기화: analyzer.sync_manual_csv_to_db()")
                print("   - CSV 병합: analyzer.merge_manual_csv()")
            
        else:
            print("\n❌ 누적 분석 중 오류가 발생했습니다.")
            print("로그 파일을 확인해주세요: cumulative_analysis.log")
    
    except Exception as e:
        print(f"\n💥 심각한 오류 발생: {e}")
        logger.error(f"메인 함수 실행 실패: {e}")


if __name__ == "__main__":
    main()