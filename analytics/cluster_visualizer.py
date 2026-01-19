#!/usr/bin/env python3
"""
통합 데이터 시각화 및 클러스터링 분석기 (메타데이터 포함)
- PA/SA 정확도 분포 시각화
- 비지도 클러스터링 분석 (K-means, DBSCAN, Hierarchical)  
- 작가/역자/시대/분류 메타데이터 기반 분석
- Pretendard Gov Variable 폰트 사용
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
import os
from pathlib import Path
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# 폰트 설정 - 한글 문제 해결
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 도커 환경에서 한글 폰트 설치 및 설정
import subprocess
import sys
import matplotlib.font_manager as fm

def setup_korean_font():
    """Docker 환경에서 한글/한자 폰트 설정 (더 안전한 방식)"""
    import subprocess
    import os
    import warnings
    warnings.filterwarnings('ignore')
    
    print("🚀 Docker 환경에서 한글/한자 폰트 설정 중...")
    
    try:
        # CJK 폰트 패키지 확실히 설치
        try:
            print("📦 CJK 폰트 패키지 강제 설치...")
            subprocess.run(['apt-get', 'update'], capture_output=True, check=False)
            subprocess.run(['apt-get', 'install', '-y', 
                           'fonts-noto-cjk-kr',      # 한국어 CJK 
                           'fonts-noto-cjk',         # 전체 CJK
                           'fonts-nanum',            # 나눔폰트
                           'fontconfig'], 
                          capture_output=True, check=False)
            
            # 폰트 캐시 강제 재생성
            subprocess.run(['fc-cache', '-f'], capture_output=True, check=False)
            print("✅ CJK 폰트 패키지 설치 완료")
        except Exception as e:
            print(f"⚠️ 폰트 설치 오류: {e}")
        
        # matplotlib 폰트 캐시 완전 재생성
        try:
            fm._rebuild()
            print("🔄 matplotlib 폰트 캐시 재생성 완료")
        except Exception as e:
            print(f"⚠️ matplotlib 캐시 재생성 실패: {e}")
        
        # 시스템에서 CJK 폰트 직접 검색
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        print(f"📋 총 폰트 개수: {len(available_fonts)}")
        
        # 한글+한자 지원 폰트를 우선순위로 검색
        korean_cjk_fonts = []
        for font_name in available_fonts:
            if any(keyword in font_name for keyword in ['Noto Sans CJK', 'Noto Serif CJK', 'NanumGothic', 'NanumMyeongjo']):
                korean_cjk_fonts.append(font_name)
        
        print(f"🔍 한국어 CJK 폰트 발견: {korean_cjk_fonts[:3]}")
        
        # 폰트 선택 (한국어 우선)
        selected_font = None
        font_priority = [
            'Noto Sans CJK KR Regular',
            'Noto Sans CJK KR',  
            'Noto Serif CJK KR',
            'NanumGothic',
            'NanumMyeongjo'
        ]
        
        # 우선순위에 따라 폰트 선택
        for priority_font in font_priority:
            for available_font in available_fonts:
                if priority_font in available_font:
                    selected_font = available_font
                    break
            if selected_font:
                break
        
        # 찾지 못했으면 CJK 폰트 중 첫 번째 사용
        if not selected_font and korean_cjk_fonts:
            selected_font = korean_cjk_fonts[0]
        
        # 그래도 없으면 시스템 기본 폰트
        if not selected_font:
            selected_font = 'DejaVu Sans'
        
        # matplotlib에 폰트 설정 적용
        plt.rcParams['font.family'] = [selected_font]
        plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 10
        
        print(f"✅ 최종 선택된 폰트: {selected_font}")
        print(f"🧪 테스트 - 한글: 경서사서제자서문집")
        print(f"🧪 테스트 - 한자: 經史子集")
        
        return True
        
    except Exception as e:
        print(f"❌ 폰트 설정 완전 실패: {e}")
        print("🔧 안전 모드: 기본 폰트 사용")
        plt.rcParams['font.family'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return False

# 한글 폰트 설정 실행
font_installed = setup_korean_font()

class NpEncoder(json.JSONEncoder):
    """NumPy 객체를 JSON으로 직렬화하기 위한 사용자 정의 인코더"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# 로깅 설정 함수
def setup_logging():
    """로깅 설정을 초기화합니다."""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 로그 파일명에 타임스탬프 추가
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"visualization_analysis_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # 로깅 포맷 설정
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # 로거 설정
    logger = logging.getLogger('VisualizationAnalyzer')
    logger.setLevel(logging.INFO)
    
    # 파일 핸들러 추가
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # 콘솔 핸들러 추가 
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_path

# 로깅 설정 초기화
logger, log_path = setup_logging()

class IntegratedAnalyzer:
    """통합 데이터 분석기"""
    
    def __init__(self, csv_file: str = None, 
                 json_file: str = "cumulative_analysis_report.json"):
        # CSV 파일 자동 선택: manual 파일이 있으면 우선 사용
        if csv_file is None:
            if Path("cumulative_analysis_results_manual.csv").exists():
                self.csv_file = "cumulative_analysis_results_manual.csv"
                print("📝 수동 편집된 CSV 파일을 사용합니다: cumulative_analysis_results_manual.csv")
            else:
                self.csv_file = "cumulative_analysis_results.csv"
                print("📊 기본 CSV 파일을 사용합니다: cumulative_analysis_results.csv")
        else:
            self.csv_file = csv_file
            
        self.json_file = json_file
        self.logger = logging.getLogger('VisualizationAnalyzer')
        self.df = None
        self.clustering_df = None
        self.output_dir = "visualization_results"
        
        # 출력 디렉토리 생성
        Path(self.output_dir).mkdir(exist_ok=True)
        
    def load_data(self):
        """CSV 및 JSON 데이터 로드"""
        self.logger.info("데이터 로드 시작")
        
        # CSV 데이터 로드
        try:
            self.df = pd.read_csv(self.csv_file, encoding='utf-8')
            msg = f"CSV 데이터 로드 완료: {len(self.df)}권"
            print(f"✅ {msg}")
            self.logger.info(msg)
            csv_loaded = True
        except Exception as e:
            msg = f"CSV 데이터 로드 실패: {e}"
            print(f"❌ {msg}")
            self.logger.error(msg)
            csv_loaded = False
        
        # JSON 데이터 로드 (클러스터링용)
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            # CSV 데이터가 있으면 CSV에서 메타데이터를 가져오고, 없으면 JSON에서 추출
            if csv_loaded and self.df is not None:
                self.clustering_df = self._prepare_clustering_data_from_csv()
            else:
                self.clustering_df = self._prepare_clustering_data_from_json(json_data)
            msg = f"클러스터링 데이터 준비 완료: {len(self.clustering_df)}권"
            print(f"✅ {msg}")
            self.logger.info(msg)
            json_loaded = True
        except Exception as e:
            msg = f"클러스터링 데이터 준비 실패: {e}"
            print(f"❌ {msg}")
            self.logger.error(msg)
            json_loaded = False
            
        return csv_loaded or json_loaded
    
    def _prepare_clustering_data_from_csv(self):
        """CSV 데이터에서 클러스터링 데이터 준비 (4부분류/시대 메타데이터 포함)"""
        # 4부분류 한자-한글 매핑 (직접 한자 사용으로 강화)
        sibu_mapping = {
            '經': '경서(經)',  # 經 - 직접 한자 사용
            '史': '사서(史)',  # 史
            '子': '제자서(子)',  # 子  
            '集': '문집(集)',  # 集
            '집': '문집(集)',  # 集 - 소문자도 처리
            '未詳': '미상',
            'unknown': '미상',
            '미상': '미상',
            '': '미상'
        }
        
        clustering_data = []
        
        # 디버깅: CSV 파일의 컬럼 정보 출력
        print(f"📊 CSV 컬럼 목록: {list(self.df.columns)}")
        print(f"📊 CSV 총 행 수: {len(self.df)}")
        
        # 시대 정보가 있는 첫 번째 행 몇 개 확인
        for i in range(min(3, len(self.df))):
            row = self.df.iloc[i]
            print(f"📊 Row {i+1}: 책명={row.get('책명', '없음')}, 시대={row.get('시대', '없음')}, 4부분류={row.get('4부분류', '없음')}")
        
        for _, row in self.df.iterrows():
            book_name = row.get('책명', '')
            pa_accuracy = row.get('PA정확도(%)', 0)
            sa_accuracy = row.get('SA정확도(%)', 0)
            author = row.get('작가', '미상')
            translator = row.get('역자', '한국고전번역원')
            
            # CSV에서 직접 4부분류와 시대 정보 읽기
            sibu_classification = row.get('4부분류', '未詳')
            period = row.get('시대', '미상')
            
            # 시대에서 간단한 형태 추출 (괄호 앞 부분만)
            dynasty = period.split('(')[0] if '(' in period and period != '미상' else period
            
            # 4부분류를 한글-한자 병기로 변환
            genre = sibu_mapping.get(sibu_classification, '미상')
            
            # 유효한 데이터만 포함
            if pa_accuracy > 0 and sa_accuracy > 0:
                clustering_data.append({
                    'filename': book_name,
                    'pa_accuracy': pa_accuracy,
                    'sa_accuracy': sa_accuracy,
                    'quality_grade': (pa_accuracy + sa_accuracy) / 2,
                    'author': author if author and author.strip() else '미상',
                    'translator': translator if translator and translator.strip() else '한국고전번역원',
                    'dynasty': dynasty,
                    'genre': genre
                })
        
        return pd.DataFrame(clustering_data)
    
    def _prepare_clustering_data_from_json(self, data):
        """JSON 데이터에서 클러스터링 데이터 준비 (메타데이터 포함)"""
        # BookMetadataExtractor 로드
        try:
            from book_metadata_extractor import BookMetadataExtractor
            metadata_extractor = BookMetadataExtractor()
            self.logger.info("BookMetadataExtractor 로드 성공 (JSON)")
        except ImportError:
            metadata_extractor = None
            self.logger.warning("BookMetadataExtractor를 로드할 수 없습니다. 메타데이터 없이 진행합니다.")
        
        clustering_data = []
        
        # JSON 구조가 cumulative_analysis_report.json 형식인 경우
        if 'books_summary' in data:
            for book in data['books_summary']:
                pa_accuracy = book.get('pa_accuracy', 0) * 100  # 0-1 범위를 0-100으로 변환
                sa_accuracy = book.get('sa_accuracy', 0) * 100  # 0-1 범위를 0-100으로 변환
                book_name = book.get('book_name', '')
                
                # 메타데이터 추출
                author = '미상'
                translator = '한국고전번역원'
                dynasty = '미상'
                genre = '기타'
                
                if metadata_extractor and book_name:
                    try:
                        metadata = metadata_extractor.extract_metadata(book_name)
                        author = metadata.get('author', '미상')
                        translator = metadata.get('translator', '한국고전번역원')
                        
                        # 상세 정보 가져오기 (시대, 분류)
                        detailed_info = metadata_extractor.get_detailed_author_info(book_name)
                        dynasty = detailed_info.get('dynasty', '미상')
                        genre = detailed_info.get('genre', '기타')
                    except Exception as e:
                        self.logger.debug(f"메타데이터 추출 실패 ({book_name}): {e}")
                
                if pa_accuracy > 0 and sa_accuracy > 0:
                    clustering_data.append({
                        'filename': book_name,
                        'pa_accuracy': pa_accuracy,
                        'sa_accuracy': sa_accuracy,
                        'quality_grade': (pa_accuracy + sa_accuracy) / 2,
                        'author': author,
                        'translator': translator,
                        'dynasty': dynasty,
                        'genre': genre
                    })
        else:
            # 기존 형식 처리
            for filename, analysis in data.items():
                # xml_level_similarity에서 PA/SA 데이터 추출
                xml_data = analysis.get('xml_level_similarity', {})
                
                # PA 정확도 계산 (F1 Score 60% + avg_similarity 40%)
                pa_f1 = xml_data.get('pa_analysis', {}).get('f1_score', 0)
                pa_similarity = xml_data.get('pa_analysis', {}).get('avg_similarity', 0)
                pa_accuracy = (pa_f1 * 0.6 + pa_similarity * 0.4) * 100
                
                # SA 정확도 계산 (F1 Score 60% + avg_similarity 40%)
                sa_f1 = xml_data.get('sa_analysis', {}).get('f1_score', 0)
                sa_similarity = xml_data.get('sa_analysis', {}).get('avg_similarity', 0)
                sa_accuracy = (sa_f1 * 0.6 + sa_similarity * 0.4) * 100
                
                # 메타데이터 추출
                author = '미상'
                translator = '한국고전번역원'
                dynasty = '미상'
                genre = '기타'
                
                if metadata_extractor and filename:
                    try:
                        metadata = metadata_extractor.extract_metadata(filename)
                        author = metadata.get('author', '미상')
                        translator = metadata.get('translator', '한국고전번역원')
                        
                        # 상세 정보 가져오기 (시대, 분류)
                        detailed_info = metadata_extractor.get_detailed_author_info(filename)
                        dynasty = detailed_info.get('dynasty', '미상')
                        genre = detailed_info.get('genre', '기타')
                    except Exception as e:
                        self.logger.debug(f"메타데이터 추출 실패 ({filename}): {e}")
                
                # 유효한 데이터만 포함
                if pa_accuracy > 0 and sa_accuracy > 0:
                    clustering_data.append({
                        'filename': filename,
                        'pa_accuracy': pa_accuracy,
                        'sa_accuracy': sa_accuracy,
                        'quality_grade': (pa_accuracy + sa_accuracy) / 2,
                        'author': author,
                        'translator': translator,
                        'dynasty': dynasty,
                        'genre': genre
                    })
        
        return pd.DataFrame(clustering_data)

    def create_metadata_analysis(self):
        """메타데이터 기반 분석 시각화"""
        if self.clustering_df is None or len(self.clustering_df) == 0:
            print("⚠️ 클러스터링 데이터가 없습니다.")
            return
            
        # 메타데이터 컬럼 확인
        required_cols = ['author', 'dynasty', 'genre']
        if not all(col in self.clustering_df.columns for col in required_cols):
            print("⚠️ 메타데이터가 없습니다. 메타데이터 분석을 건너뜁니다.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('메타데이터 기반 분석', fontsize=16, fontweight='bold')
        
        # 1. 작가별 PA/SA 분포
        author_counts = self.clustering_df['author'].value_counts()
        top_authors = author_counts.head(8).index  # 상위 8명 작가
        author_data = self.clustering_df[self.clustering_df['author'].isin(top_authors)]
        
        for i, author in enumerate(top_authors):
            author_subset = author_data[author_data['author'] == author]
            ax1.scatter(author_subset['pa_accuracy'], author_subset['sa_accuracy'], 
                       label=author, alpha=0.7, s=60)
        
        ax1.set_xlabel('PA 정확도 (%)')
        ax1.set_ylabel('SA 정확도 (%)')
        ax1.set_title('작가별 PA/SA 정확도 분포')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        ax1.grid(True, alpha=0.3)
        
        # 2. 시대별 품질 분포
        dynasty_quality = self.clustering_df.groupby('dynasty')['quality_grade'].mean().sort_values(ascending=False)
        colors = plt.cm.Set3(np.arange(len(dynasty_quality)))
        
        bars = ax2.bar(range(len(dynasty_quality)), dynasty_quality.values, color=colors, alpha=0.8)
        ax2.set_xlabel('시대')
        ax2.set_ylabel('평균 품질 점수')
        ax2.set_title('시대별 평균 품질 점수')
        ax2.set_xticks(range(len(dynasty_quality)))
        ax2.set_xticklabels(dynasty_quality.index, rotation=45, ha='right')
        
        # 막대 위에 수치 표시
        for bar, value in zip(bars, dynasty_quality.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=10)
        
        # 3. 분류별 품질 분포 (4부분류)
        genre_quality = self.clustering_df.groupby('genre')['quality_grade'].mean().sort_values(ascending=False)
        colors = plt.cm.Pastel1(np.arange(len(genre_quality)))
        
        bars = ax3.bar(range(len(genre_quality)), genre_quality.values, color=colors, alpha=0.8)
        ax3.set_xlabel('4부분류')
        ax3.set_ylabel('평균 품질 점수')
        ax3.set_title('4부분류별 평균 품질 점수')
        ax3.set_xticks(range(len(genre_quality)))
        ax3.set_xticklabels(genre_quality.index, rotation=45, ha='right', fontsize=10)
        
        # 막대 위에 수치 표시
        for bar, value in zip(bars, genre_quality.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=10)
        
        # 4. 시대-분류 매트릭스 히트맵
        pivot_table = self.clustering_df.pivot_table(
            values='quality_grade', 
            index='dynasty', 
            columns='genre', 
            aggfunc='mean'
        )
        
        import seaborn as sns
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax4, 
                   cbar_kws={'label': '평균 품질 점수'})
        ax4.set_title('시대별-4부분류별 품질 점수 히트맵')
        ax4.set_xlabel('4부분류')
        ax4.set_ylabel('시대')
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/metadata_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 메타데이터 분석 차트 저장: {output_path}")
        
    def create_author_clustering_analysis(self, results):
        """작가별 클러스터링 분석"""
        if self.clustering_df is None or 'author' not in self.clustering_df.columns:
            print("⚠️ 작가 정보가 없습니다.")
            return
            
        # K-means 결과를 데이터프레임에 추가
        df_with_clusters = self.clustering_df.copy()
        df_with_clusters['cluster'] = results['kmeans']['labels']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('작가별 클러스터링 분석', fontsize=16, fontweight='bold')
        
        # 1. 클러스터별 작가 분포
        cluster_author = df_with_clusters.groupby(['cluster', 'author']).size().unstack(fill_value=0)
        cluster_author.plot(kind='bar', stacked=True, ax=ax1, colormap='Set3')
        ax1.set_xlabel('클러스터')
        ax1.set_ylabel('서종 수')
        ax1.set_title('클러스터별 작가 분포')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
        
        # 2. 작가별 클러스터 분포
        author_counts = df_with_clusters['author'].value_counts()
        top_authors = author_counts.head(8).index
        top_author_data = df_with_clusters[df_with_clusters['author'].isin(top_authors)]
        
        author_cluster = top_author_data.groupby(['author', 'cluster']).size().unstack(fill_value=0)
        author_cluster.plot(kind='bar', ax=ax2, colormap='viridis')
        ax2.set_xlabel('작가')
        ax2.set_ylabel('서종 수')
        ax2.set_title('주요 작가별 클러스터 분포')
        ax2.legend(title='클러스터', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/author_clustering_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 작가별 클러스터링 분석 차트 저장: {output_path}")

    def create_quality_grade_distribution(self):
        """품질 등급 분포 차트"""
        if self.df is None:
            print("❌ CSV 데이터가 없어 품질 등급 분포를 생성할 수 없습니다.")
            return
        
        # 한글 폰트 확실히 설정
        setup_korean_font()
        
        plt.figure(figsize=(12, 8))
        
        # 품질 등급별 색상 설정
        grade_colors = {
            'A+': '#28a745', 'A': '#20c997', 'B+': '#17a2b8', 
            'B': '#007bff', 'C+': '#ffc107', 'C': '#fd7e14', 
            'D': '#dc3545', 'F': '#6c757d'
        }
        
        grade_counts = self.df['품질등급'].value_counts().sort_index()
        
        # 서브플롯 생성
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('품질 등급 분포 분석', fontsize=16, fontweight='bold')
        
        # 1. 막대 그래프
        colors = [grade_colors.get(grade, '#6c757d') for grade in grade_counts.index]
        bars = ax1.bar(grade_counts.index, grade_counts.values, color=colors, alpha=0.8)
        ax1.set_title('품질 등급별 서종 수', fontweight='bold')
        ax1.set_xlabel('품질 등급')
        ax1.set_ylabel('서종 수')
        
        # 막대 위에 수치 표시
        for bar, count in zip(bars, grade_counts.values):
            height = bar.get_height()
            ax1.annotate(f'{count}권', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        # 2. 도넛 차트
        wedges, texts, autotexts = ax2.pie(grade_counts.values, labels=grade_counts.index, 
                                          colors=colors, autopct='%1.1f%%', startangle=90,
                                          wedgeprops=dict(width=0.5))
        ax2.set_title('품질 등급 비율', fontweight='bold')
        
        # 3. PA vs SA 산점도
        scatter = ax3.scatter(self.df['PA정확도(%)'], self.df['SA정확도(%)'], 
                             c=[grade_colors.get(grade, '#6c757d') for grade in self.df['품질등급']], 
                             alpha=0.7, s=60)
        ax3.set_xlabel('PA 정확도 (%)')
        ax3.set_ylabel('SA 정확도 (%)')
        ax3.set_title('PA vs SA 정확도 분포', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 평균선 추가
        pa_mean = self.df['PA정확도(%)'].mean()
        sa_mean = self.df['SA정확도(%)'].mean()
        ax3.axvline(pa_mean, color='red', linestyle='--', alpha=0.7, label=f'PA 평균: {pa_mean:.1f}%')
        ax3.axhline(sa_mean, color='blue', linestyle='--', alpha=0.7, label=f'SA 평균: {sa_mean:.1f}%')
        ax3.legend()
        
        # 4. 히스토그램
        ax4.hist(self.df['PA정확도(%)'], bins=15, alpha=0.7, color='skyblue', 
                label=f'PA (평균: {pa_mean:.1f}%)', edgecolor='black')
        ax4.hist(self.df['SA정확도(%)'], bins=15, alpha=0.7, color='lightcoral', 
                label=f'SA (평균: {sa_mean:.1f}%)', edgecolor='black')
        ax4.set_xlabel('정확도 (%)')
        ax4.set_ylabel('빈도')
        ax4.set_title('PA/SA 정확도 분포', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = f"{self.output_dir}/quality_grade_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        msg = f"품질 등급 분포 차트 저장: {output_path}"
        print(f"📊 {msg}")
        self.logger.info(msg)
        
    def create_performance_correlation(self):
        """성능 상관관계 히트맵"""
        if self.df is None:
            print("❌ CSV 데이터가 없어 상관관계 분석을 수행할 수 없습니다.")
            return
            
        plt.figure(figsize=(12, 8))
        
        # 수치형 컬럼만 선택
        numeric_cols = ['PA정확도(%)', 'SA정확도(%)', '임베딩유사도(%)', '총문단수', 
                       'PA_Precision(%)', 'PA_Recall(%)', 'PA_F1Score(%)', 'PA_평균유사도(%)',
                       'SA_Precision(%)', 'SA_Recall(%)', 'SA_F1Score(%)', 'SA_한세트유사도(%)']
        
        # 존재하는 컬럼만 필터링
        available_cols = [col for col in numeric_cols if col in self.df.columns]
        correlation_matrix = self.df[available_cols].corr()
        
        # 히트맵 생성
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5})
        
        plt.title('성능 지표 간 상관관계', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output_path = f"{self.output_dir}/performance_correlation.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"📊 성능 상관관계 히트맵 저장: {output_path}")

    # ========== 클러스터링 분석 함수들 ==========
    
    def find_optimal_k(self, X, max_k=8):
        """엘보우 방법으로 최적 k 찾기 - 여러 지표 고려"""
        inertias = []
        silhouette_scores = []
        calinski_harabasz_scores = []
        davies_bouldin_scores = []
        k_range = range(2, min(max_k + 1, len(X)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, labels))
            
            # 추가 평가 지표
            from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
            calinski_harabasz_scores.append(calinski_harabasz_score(X, labels))
            davies_bouldin_scores.append(davies_bouldin_score(X, labels))
        
        return k_range, inertias, silhouette_scores, calinski_harabasz_scores, davies_bouldin_scores
    
    def find_elbow_point(self, k_range, inertias):
        """엘보우 지점을 찾는 함수"""
        k_range = list(k_range)
        
        if len(inertias) < 3:
            return k_range[0] if k_range else 2
        
        # 1차 차분 계산 (기울기)
        first_diff = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        
        # 2차 차분 계산 (기울기 변화율)
        second_diff = [first_diff[i] - first_diff[i+1] for i in range(len(first_diff)-1)]
        
        # 2차 차분이 가장 큰 지점이 엘보우 지점 (곡률이 가장 큰 지점)
        if second_diff:
            elbow_idx = np.argmax(second_diff) + 1  # 인덱스 보정
            if elbow_idx < len(k_range):
                return k_range[elbow_idx]
        
        # 대안: 감소율이 50% 이하로 떨어지는 지점
        for i in range(1, len(first_diff)):
            if len(first_diff) > i and first_diff[i] < first_diff[0] * 0.5:
                return k_range[i+1]
                
        return k_range[0] if k_range else 2
    
    def perform_clustering_analysis(self):
        """클러스터링 분석 수행"""
        if self.clustering_df is None or len(self.clustering_df) < 3:
            print("❌ 클러스터링을 위한 충분한 데이터가 없습니다.")
            return None
            
        # 데이터 준비
        X = self.clustering_df[['pa_accuracy', 'sa_accuracy']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        results = {}
        
        # 1. K-means 클러스터링 - 여러 k값 분석
        print("🔍 K-means 최적 클러스터 수 찾는 중...")
        k_range, inertias, silhouette_scores, calinski_scores, davies_scores = self.find_optimal_k(X_scaled)
        
        # 여러 지표로 최적 k 결정
        silhouette_optimal = k_range[np.argmax(silhouette_scores)]
        calinski_optimal = k_range[np.argmax(calinski_scores)]
        davies_optimal = k_range[np.argmin(davies_scores)]  # 낮을수록 좋음
        
        print(f"📊 클러스터 수 후보:")
        print(f"   실루엣 점수 최적: k={silhouette_optimal} (점수: {max(silhouette_scores):.3f})")
        print(f"   Calinski-Harabasz 최적: k={calinski_optimal} (점수: {max(calinski_scores):.1f})")
        print(f"   Davies-Bouldin 최적: k={davies_optimal} (점수: {min(davies_scores):.3f})")
        
        # 지표별 의미 설명
        print(f"\n💡 지표 해석:")
        print(f"   🔹 실루엣 점수: 클러스터 내 응집도 vs 클러스터 간 분리도")
        print(f"      → k={silhouette_optimal}에서 클러스터가 가장 잘 구분됨")
        print(f"   🔹 Calinski-Harabasz: 클러스터 간 분산 vs 클러스터 내 분산")  
        print(f"      → k={calinski_optimal}에서 클러스터 구조가 가장 명확함")
        print(f"   🔹 Davies-Bouldin: 클러스터 내 거리 vs 클러스터 간 거리")
        print(f"      → k={davies_optimal}에서 클러스터가 가장 컴팩트함")
        
        # 엘보우 방법 설명 추가
        print(f"\n📈 엘보우 방법:")
        elbow_k = self.find_elbow_point(k_range, inertias)
        print(f"   📍 엘보우 지점: k={elbow_k} (급격한 감소가 완만해지는 지점)")
        print(f"   💭 엘보우 방법은 클러스터 내 분산의 감소율이 둔화되는 지점을 찾음")
        
        # 종합 권장사항 제시
        print(f"\n🎯 종합 권장사항:")
        if silhouette_optimal == elbow_k:
            print(f"   ✅ 실루엣 점수와 엘보우 방법이 일치: k={silhouette_optimal} 강력 권장")
        elif abs(silhouette_optimal - elbow_k) <= 1:
            print(f"   ✅ 실루엣 점수(k={silhouette_optimal})와 엘보우 방법(k={elbow_k})이 유사: 둘 다 고려 가능")
        else:
            print(f"   ⚠️  지표들이 다른 결과를 제시: 도메인 지식으로 판단 필요")
            print(f"      - 큰 그룹 구분: k={min(silhouette_optimal, elbow_k)}")  
            print(f"      - 세부 분석: k={max(silhouette_optimal, calinski_optimal, davies_optimal)}")
        
        # 실루엣 점수를 주 지표로 사용하되, 여러 k값으로 분석 저장
        optimal_k = silhouette_optimal
        print(f"\n✅ 선택된 클러스터 수: {optimal_k}")
        print(f"   📋 선택 근거: 실루엣 점수가 가장 높아 클러스터 구분이 명확함")
        
        # 여러 k값으로 K-means 수행
        kmeans_results = {}
        for k in [2, 3, 4]:  # 주요 k값들 테스트
            if k <= len(X_scaled):
                kmeans_k = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels_k = kmeans_k.fit_predict(X_scaled)
                kmeans_results[f'k{k}'] = {
                    'labels': labels_k,
                    'centers': scaler.inverse_transform(kmeans_k.cluster_centers_),
                    'silhouette_score': silhouette_score(X_scaled, labels_k)
                }
        
        # 최적 k 결과를 메인으로 저장
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        
        results['kmeans'] = {
            'labels': kmeans_labels,
            'centers': scaler.inverse_transform(kmeans.cluster_centers_),
            'optimal_k': optimal_k,
            'silhouette_score': silhouette_score(X_scaled, kmeans_labels),
            'k_analysis': {
                'k_range': list(k_range),
                'silhouette_scores': silhouette_scores,
                'calinski_scores': calinski_scores,
                'davies_scores': davies_scores
            },
            'multiple_k_results': kmeans_results
        }
        
        # 2. DBSCAN 클러스터링
        print("🔍 DBSCAN 클러스터링 수행 중...")
        dbscan = DBSCAN(eps=0.5, min_samples=3)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = list(dbscan_labels).count(-1)
        
        print(f"✅ DBSCAN 클러스터 수: {n_clusters_dbscan}, 노이즈 포인트: {n_noise}")
        
        if n_clusters_dbscan > 1:
            dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels)
        else:
            dbscan_silhouette = -1
        
        results['dbscan'] = {
            'labels': dbscan_labels,
            'n_clusters': n_clusters_dbscan,
            'n_noise': n_noise,
            'silhouette_score': dbscan_silhouette
        }
        
        # 2-1. DBSCAN 하이퍼파라미터 탐색 (노이즈 최소화 및 품질 균형)
        def evaluate_dbscan_params(Xs, eps, min_samples):
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(Xs)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            sil = -1
            if n_clusters > 1 and len(set(labels)) > 1:
                try:
                    sil = silhouette_score(Xs, labels)
                except Exception:
                    sil = -1
            return {
                'eps': eps,
                'min_samples': min_samples,
                'labels': labels,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette': sil
            }

        # eps 범위는 데이터 스케일에 따라 0.1~1.5 사이를 탐색, min_samples는 2~6 범위
        eps_candidates = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0]
        min_samples_candidates = [2, 3, 4, 5, 6]
        dbscan_grid_results = []
        for eps in eps_candidates:
            for ms in min_samples_candidates:
                res = evaluate_dbscan_params(X_scaled, eps, ms)
                dbscan_grid_results.append(res)

        # 스코어링: (노이즈를 줄이되, 클러스터가 1개만 나오지 않고, 실루엣도 반영)
        # 점수 = -노이즈비율 + 0.5*실루엣 (실루엣이 -1이면 0 처리) + 클러스터 수 보너스(1 초과시)
        n_total = len(X_scaled)
        def score_row(r):
            sil = max(0, r['silhouette'])
            cluster_bonus = 0.2 if r['n_clusters'] >= 2 else -0.5
            return - (r['n_noise'] / max(1, n_total)) + 0.5 * sil + cluster_bonus

        # 각 후보에 점수 부여
        for r in dbscan_grid_results:
            r['score'] = score_row(r)

        best_dbscan = max(dbscan_grid_results, key=lambda r: r['score'])
        print("\n🛠 DBSCAN 하이퍼파라미터 제안:")
        print(f"   • 권장 eps: {best_dbscan['eps']}, min_samples: {best_dbscan['min_samples']}")
        print(f"   • 예상 클러스터 수: {best_dbscan['n_clusters']}, 예상 노이즈: {best_dbscan['n_noise']}")
        if best_dbscan['silhouette'] >= 0:
            print(f"   • 예상 실루엣 점수: {best_dbscan['silhouette']:.3f}")
        else:
            print("   • 실루엣 점수 계산 불가 (클러스터가 1개 또는 전부 노이즈)")

        results['dbscan_tuning'] = {
            'candidates': dbscan_grid_results,
            'recommended': best_dbscan
        }

        # 3. 계층적 클러스터링
        print("🔍 계층적 클러스터링 수행 중...")
        hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
        hierarchical_labels = hierarchical.fit_predict(X_scaled)
        
        results['hierarchical'] = {
            'labels': hierarchical_labels,
            'silhouette_score': silhouette_score(X_scaled, hierarchical_labels)
        }
        
        return results, X, X_scaled, scaler
    
    def plot_clustering_results(self, results, X, X_scaled, scaler):
        """클러스터링 결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('PA/SA 점수 기반 비지도 클러스터링 분석', fontsize=16, fontweight='bold')
        
        # 1. 원본 데이터 분포
        ax1 = axes[0, 0]
        scatter = ax1.scatter(self.clustering_df['pa_accuracy'], self.clustering_df['sa_accuracy'], 
                             c=self.clustering_df['quality_grade'], cmap='viridis', 
                             s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('PA Accuracy (%)' if not font_installed else 'PA 정확도 (%)')
        ax1.set_ylabel('SA Accuracy (%)' if not font_installed else 'SA 정확도 (%)')
        ax1.set_title('Original Data Distribution' if not font_installed else '원본 데이터 분포 (품질 등급별 색상)')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Quality Grade' if not font_installed else '품질 등급')
        
        # 2. K-means 클러스터링
        ax2 = axes[0, 1]
        kmeans_labels = results['kmeans']['labels']
        centers = results['kmeans']['centers']
        
        scatter = ax2.scatter(self.clustering_df['pa_accuracy'], self.clustering_df['sa_accuracy'], 
                             c=kmeans_labels, cmap='Set3', 
                             s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax2.scatter(centers[:, 0], centers[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='중심점')
        ax2.set_xlabel('PA 정확도 (%)')
        ax2.set_ylabel('SA 정확도 (%)')
        ax2.set_title(f'K-means 클러스터링 (k={results["kmeans"]["optimal_k"]})\n'
                      f'실루엣 점수: {results["kmeans"]["silhouette_score"]:.3f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. DBSCAN 클러스터링
        ax3 = axes[1, 0]
        dbscan_labels = results['dbscan']['labels']
        unique_labels = set(dbscan_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # 노이즈 포인트는 검은색으로
                col = 'black'
            
            class_member_mask = (dbscan_labels == k)
            xy = X[class_member_mask]
            ax3.scatter(self.clustering_df.loc[class_member_mask, 'pa_accuracy'], 
                       self.clustering_df.loc[class_member_mask, 'sa_accuracy'],
                       c=[col], s=60, alpha=0.7, 
                       edgecolors='black', linewidth=0.5,
                       label=f'클러스터 {k}' if k != -1 else '노이즈')
        
        ax3.set_xlabel('PA 정확도 (%)')
        ax3.set_ylabel('SA 정확도 (%)')
        ax3.set_title(f'DBSCAN 클러스터링\n클러스터: {results["dbscan"]["n_clusters"]}, '
                      f'노이즈: {results["dbscan"]["n_noise"]}')
        ax3.grid(True, alpha=0.3)
        if len(unique_labels) <= 10:  # 범례가 너무 많지 않을 때만 표시
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. 계층적 클러스터링
        ax4 = axes[1, 1]
        hierarchical_labels = results['hierarchical']['labels']
        
        scatter = ax4.scatter(self.clustering_df['pa_accuracy'], self.clustering_df['sa_accuracy'], 
                             c=hierarchical_labels, cmap='Set3', 
                             s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax4.set_xlabel('PA 정확도 (%)')
        ax4.set_ylabel('SA 정확도 (%)')
        ax4.set_title(f'계층적 클러스터링\n'
                      f'실루엣 점수: {results["hierarchical"]["silhouette_score"]:.3f}')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/clustering_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 클러스터링 분석 차트 저장: {output_path}")
    
    def plot_elbow_method(self, X_scaled):
        """엘보우 방법 및 다중 지표 시각화"""
        k_range, inertias, silhouette_scores, calinski_scores, davies_scores = self.find_optimal_k(X_scaled)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 엘보우 차트
        ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Cluster Number (k)' if not font_installed else '클러스터 수 (k)')
        ax1.set_ylabel('Inertia (WSS)')
        ax1.set_title('Elbow Method - Inertia' if not font_installed else '엘보우 방법 - Inertia')
        ax1.grid(True, alpha=0.3)
        
        # 2. 실루엣 점수 차트
        ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Cluster Number (k)' if not font_installed else '클러스터 수 (k)')
        ax2.set_ylabel('Silhouette Score' if not font_installed else '실루엣 점수')
        ax2.set_title('Optimal k - Silhouette Score' if not font_installed else '최적 클러스터 수 - 실루엣 점수')
        ax2.grid(True, alpha=0.3)
        
        # 최적 k 표시
        optimal_k = k_range[np.argmax(silhouette_scores)]
        ax2.axvline(x=optimal_k, color='green', linestyle='--', 
                    label=f'Optimal k={optimal_k}' if not font_installed else f'최적 k={optimal_k}')
        ax2.legend()
        
        # 3. Calinski-Harabasz 점수
        ax3.plot(k_range, calinski_scores, 'go-', linewidth=2, markersize=8)
        ax3.set_xlabel('Cluster Number (k)' if not font_installed else '클러스터 수 (k)')
        ax3.set_ylabel('Calinski-Harabasz Score')
        ax3.set_title('Calinski-Harabasz Index')
        ax3.grid(True, alpha=0.3)
        
        calinski_optimal = k_range[np.argmax(calinski_scores)]
        ax3.axvline(x=calinski_optimal, color='orange', linestyle='--', 
                    label=f'Optimal k={calinski_optimal}' if not font_installed else f'최적 k={calinski_optimal}')
        ax3.legend()
        
        # 4. Davies-Bouldin 점수 (낮을수록 좋음)
        ax4.plot(k_range, davies_scores, 'mo-', linewidth=2, markersize=8)
        ax4.set_xlabel('Cluster Number (k)' if not font_installed else '클러스터 수 (k)')
        ax4.set_ylabel('Davies-Bouldin Score')
        ax4.set_title('Davies-Bouldin Index (Lower is Better)' if not font_installed else 'Davies-Bouldin 지수 (낮을수록 좋음)')
        ax4.grid(True, alpha=0.3)
        
        davies_optimal = k_range[np.argmin(davies_scores)]
        ax4.axvline(x=davies_optimal, color='red', linestyle='--', 
                    label=f'Optimal k={davies_optimal}' if not font_installed else f'최적 k={davies_optimal}')
        ax4.legend()
        
        plt.suptitle('K-means Clustering Optimization' if not font_installed else 'K-means 클러스터링 최적화', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = f'{self.output_dir}/elbow_method.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 엘보우 방법 차트 저장: {output_path}")
        
    def plot_multiple_k_comparison(self, results, X, X_scaled, scaler):
        """여러 k값 비교 시각화"""
        multiple_k = results['kmeans']['multiple_k_results']
        
        n_plots = len(multiple_k)
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        if n_plots == 1:
            axes = [axes]
            
        fig.suptitle('K-means Clustering Comparison (Different k values)' if not font_installed else 
                    'K-means 클러스터링 비교 (다양한 k값)', fontsize=14, fontweight='bold')
        
        for i, (k_name, k_result) in enumerate(multiple_k.items()):
            ax = axes[i]
            labels = k_result['labels']
            centers = k_result['centers']
            silhouette = k_result['silhouette_score']
            
            # 산점도
            scatter = ax.scatter(self.clustering_df['pa_accuracy'], self.clustering_df['sa_accuracy'], 
                               c=labels, cmap='Set3', s=60, alpha=0.7, 
                               edgecolors='black', linewidth=0.5)
            
            # 중심점 표시
            ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', 
                      s=200, linewidths=3, label='Centers' if not font_installed else '중심점')
            
            ax.set_xlabel('PA Accuracy (%)' if not font_installed else 'PA 정확도 (%)')
            ax.set_ylabel('SA Accuracy (%)' if not font_installed else 'SA 정확도 (%)')
            ax.set_title(f'{k_name}: Silhouette={silhouette:.3f}' if not font_installed else 
                        f'{k_name}: 실루엣={silhouette:.3f}')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        output_path = f'{self.output_dir}/multiple_k_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 다중 k값 비교 차트 저장: {output_path}")
    
    def find_cluster_representatives(self, cluster_data, cluster_centers, cluster_id, scaler):
        """클러스터 중심점에 가장 가까운 대표 서목 찾기"""
        # 현재 클러스터의 데이터만 선택
        cluster_mask = cluster_data['cluster'] == cluster_id
        cluster_subset = cluster_data[cluster_mask].copy()
        
        if len(cluster_subset) == 0:
            return None
            
        # 클러스터 데이터의 특성 벡터 (PA, SA 정확도)
        features = cluster_subset[['pa_accuracy', 'sa_accuracy']].values
        features_scaled = scaler.transform(features)
        
        # 클러스터 중심점
        center = cluster_centers[cluster_id].reshape(1, -1)
        
        # 각 데이터 포인트와 중심점 간의 유클리드 거리 계산
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(features_scaled, center).flatten()
        
        # 거리가 가장 가까운 순으로 정렬
        cluster_subset = cluster_subset.copy()
        cluster_subset['distance_to_center'] = distances
        closest_representatives = cluster_subset.nsmallest(3, 'distance_to_center')
        
        return closest_representatives

    def analyze_clusters(self, results):
        """클러스터별 특성 분석"""
        print("\n" + "="*60)
        print("📊 클러스터별 특성 분석")
        print("="*60)
        
        # K-means 클러스터 분석
        print("\n🔸 K-means 클러스터 분석:")
        df_kmeans = self.clustering_df.copy()
        df_kmeans['cluster'] = results['kmeans']['labels']
        
        # 스케일러 준비 (클러스터 중심점 계산용)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features = df_kmeans[['pa_accuracy', 'sa_accuracy']].values
        scaler.fit(features)

        # 대표 서목 저장을 위한 레코드 수집
        representatives_records = []
        
        for cluster_id in sorted(df_kmeans['cluster'].unique()):
            cluster_data = df_kmeans[df_kmeans['cluster'] == cluster_id]
            print(f"\n클러스터 {cluster_id} ({len(cluster_data)}개 서종):")
            print(f"  PA 정확도: {cluster_data['pa_accuracy'].mean():.1f}% ± {cluster_data['pa_accuracy'].std():.1f}%")
            print(f"  SA 정확도: {cluster_data['sa_accuracy'].mean():.1f}% ± {cluster_data['sa_accuracy'].std():.1f}%")
            print(f"  품질 등급: {cluster_data['quality_grade'].mean():.1f}% ± {cluster_data['quality_grade'].std():.1f}%")
            
            # 📚 메타데이터 기반 특성 분석
            if 'author' in cluster_data.columns:
                # 주요 작가 분포
                top_authors = cluster_data['author'].value_counts().head(3)
                if len(top_authors) > 0:
                    print(f"  📝 주요 작가: {', '.join([f'{author}({count}권)' for author, count in top_authors.items()])}")
                
                # 시대 분포
                if 'dynasty' in cluster_data.columns:
                    dynasty_dist = cluster_data['dynasty'].value_counts()
                    if len(dynasty_dist) > 0:
                        print(f"  🏛️ 시대 분포: {', '.join([f'{dynasty}({count})' for dynasty, count in dynasty_dist.head(3).items()])}")
                
                # 분류 분포
                if 'genre' in cluster_data.columns:
                    genre_dist = cluster_data['genre'].value_counts()
                    if len(genre_dist) > 0:
                        print(f"  📚 분류 분포: {', '.join([f'{genre}({count})' for genre, count in genre_dist.head(3).items()])}")
            
            # 🎯 중심점에 가장 가까운 대표 서목
            representatives = self.find_cluster_representatives(
                df_kmeans, results['kmeans']['centers'], cluster_id, scaler
            )
            
            if representatives is not None and len(representatives) > 0:
                print(f"  🎯 중심점 대표 서목 (거리순):")
                for idx, (_, rep) in enumerate(representatives.iterrows(), 1):
                    filename = rep['filename'].replace('.xml', '')
                    distance = rep['distance_to_center']
                    pa_acc = rep['pa_accuracy']
                    sa_acc = rep['sa_accuracy']
                    print(f"     {idx}. {filename}")
                    print(f"        PA: {pa_acc:.1f}%, SA: {sa_acc:.1f}% (중심거리: {distance:.3f})")

                    # CSV 저장용 레코드
                    representatives_records.append({
                        'cluster': cluster_id,
                        'rank': idx,
                        'filename': filename,
                        'pa_accuracy': pa_acc,
                        'sa_accuracy': sa_acc,
                        'distance_to_center': float(distance)
                    })
            
            # 품질 기준 우수 서종
            top_quality = cluster_data.nlargest(3, 'quality_grade')
            print(f"  🏆 품질 우수 서종: {', '.join(top_quality['filename'].str.replace('.xml', '').tolist())}")

        # 대표 서목 CSV로 저장
        if representatives_records:
            reps_df = pd.DataFrame(representatives_records)
            reps_out = f"{self.output_dir}/cluster_representatives.csv"
            reps_df.to_csv(reps_out, index=False, encoding='utf-8-sig')
            print(f"\n🗂️ 대표 서목 요약 저장: {reps_out}")
            self.logger.info(f"대표 서목 요약 저장: {reps_out}")

        # DBSCAN 튜닝 요약 및 개선 제안
        if 'dbscan_tuning' in results:
            rec = results['dbscan_tuning']['recommended']
            total = len(self.clustering_df)
            noise_ratio = rec['n_noise'] / max(1, total)
            print("\n🧪 DBSCAN 하이퍼파라미터 권장안 요약")
            print(f"   • eps={rec['eps']}, min_samples={rec['min_samples']}")
            print(f"   • 예상 클러스터 수={rec['n_clusters']}, 노이즈={rec['n_noise']} ({noise_ratio*100:.1f}%)")
            if rec['silhouette'] >= 0:
                print(f"   • 예상 실루엣 점수={rec['silhouette']:.3f}")
            
            # 상황별 제안 문구
            print("\n📝 개선 제안:")
            if noise_ratio > 0.3:
                print("   - 노이즈가 많은 편입니다. eps를 소폭 증가(+0.05~0.1)하거나 min_samples를 완화(감소)해보세요.")
            elif rec['n_clusters'] <= 1:
                print("   - 클러스터가 하나만 형성됩니다. eps를 확대하거나 min_samples를 줄여 분할을 유도해보세요.")
            else:
                print("   - 현재 파라미터로 적정 분할이 이루어졌습니다. 도메인 기준으로 k-means 결과와 병행 검토를 권장합니다.")

            # 권장안 및 후보 저장
            try:
                import json
                tuning_dir = self.output_dir
                # 후보 상위 10개 선별
                candidates = results['dbscan_tuning']['candidates']
                top_candidates = sorted(candidates, key=lambda r: r.get('score', 0), reverse=True)[:10]
                
                # JSON 저장을 위해 numpy 배열 제거
                def clean_for_json(data):
                    if isinstance(data, dict):
                        cleaned = {}
                        for k, v in data.items():
                            if k != 'labels':  # labels 필드 제거 (numpy 배열이므로)
                                cleaned[k] = clean_for_json(v)
                        return cleaned
                    elif isinstance(data, list):
                        return [clean_for_json(item) for item in data]
                    else:
                        return data
                
                cleaned_rec = clean_for_json(rec)
                cleaned_top_candidates = clean_for_json(top_candidates)
                
                reco_path = f"{tuning_dir}/dbscan_tuning_recommendations.json"
                with open(reco_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'recommended': cleaned_rec,
                        'top_candidates': cleaned_top_candidates
                    }, f, ensure_ascii=False, indent=2, cls=NpEncoder)
                print(f"🗂️ DBSCAN 권장안 저장: {reco_path}")
                self.logger.info(f"DBSCAN 권장안 저장: {reco_path}")

                # 후보 CSV 저장
                cand_df = pd.DataFrame(candidates)
                cand_csv = f"{tuning_dir}/dbscan_tuning_candidates.csv"
                cand_df.to_csv(cand_csv, index=False, encoding='utf-8-sig')
                print(f"🗂️ DBSCAN 후보 목록 저장: {cand_csv}")
                self.logger.info(f"DBSCAN 후보 목록 저장: {cand_csv}")
            except Exception as e:
                print(f"⚠️ DBSCAN 튜닝 결과 저장 중 오류: {e}")
                self.logger.error(f"DBSCAN 튜닝 결과 저장 중 오류: {e}")
    
    def create_statistical_summary(self):
        """통계 요약 보고서"""
        print("=" * 60)
        print("📈 통계 요약 보고서")
        print("=" * 60)
        
        if self.df is not None:
            # 기본 통계 (CSV 데이터 기반)
            print(f"📚 총 서종 수 (CSV): {len(self.df)}권")
            print(f"📊 PA 정확도: 평균 {self.df['PA정확도(%)'].mean():.1f}% (최소 {self.df['PA정확도(%)'].min():.1f}%, 최대 {self.df['PA정확도(%)'].max():.1f}%)")
            print(f"📊 SA 정확도: 평균 {self.df['SA정확도(%)'].mean():.1f}% (최소 {self.df['SA정확도(%)'].min():.1f}%, 최대 {self.df['SA정확도(%)'].max():.1f}%)")
            
            # 품질 등급 분포
            print("\n🏆 품질 등급 분포:")
            grade_counts = self.df['품질등급'].value_counts().sort_index()
            for grade, count in grade_counts.items():
                percentage = (count / len(self.df)) * 100
                print(f"   {grade}: {count}권 ({percentage:.1f}%)")
        
        if self.clustering_df is not None:
            print(f"\n📚 클러스터링 데이터 (JSON): {len(self.clustering_df)}권")
            print(f"📊 PA 정확도: 평균 {self.clustering_df['pa_accuracy'].mean():.1f}%")
            print(f"📊 SA 정확도: 평균 {self.clustering_df['sa_accuracy'].mean():.1f}%")
    
    def run_all_analysis(self):
        """모든 분석 실행"""
        start_time = datetime.now()
        self.logger.info(f"통합 데이터 시각화 및 클러스터링 분석 시작 - 시작시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("🚀 통합 데이터 시각화 및 클러스터링 분석 시작")
        print("=" * 60)
        
        if not self.load_data():
            self.logger.error("데이터 로드 실패로 분석을 중단합니다")
            return
        
        # 통계 요약
        self.logger.info("통계 요약 생성 시작")
        self.create_statistical_summary()
        
        # 기본 시각화 (CSV 데이터 기반)
        if self.df is not None:
            print("\n📊 기본 시각화 생성 중...")
            self.logger.info("기본 시각화 생성 시작")
            self.create_quality_grade_distribution()
            self.create_performance_correlation()
            self.logger.info("기본 시각화 생성 완료")
        
        # 클러스터링 분석 (JSON 데이터 기반)
        if self.clustering_df is not None:
            print("\n🔍 클러스터링 분석 수행 중...")
            self.logger.info("클러스터링 분석 시작")
            clustering_results = self.perform_clustering_analysis()
            
            if clustering_results is not None:
                results, X, X_scaled, scaler = clustering_results
                
                # 엘보우 방법 및 다중 지표 시각화
                self.plot_elbow_method(X_scaled)
                
                # 여러 k값 비교 시각화
                self.plot_multiple_k_comparison(results, X, X_scaled, scaler)
                
                # 클러스터링 결과 시각화
                self.plot_clustering_results(results, X, X_scaled, scaler)
                
                # 📊 메타데이터 기반 분석 추가
                print("\n📚 메타데이터 기반 분석 수행 중...")
                self.logger.info("메타데이터 분석 시작")
                self.create_metadata_analysis()
                self.create_author_clustering_analysis(results)
                self.logger.info("메타데이터 분석 완료")
                
                # 클러스터별 특성 분석
                self.analyze_clusters(results)
                
                self.logger.info("클러스터링 분석 완료")
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        success_msg = "모든 분석이 완료되었습니다!"
        result_msg = f"결과 파일들이 '{self.output_dir}' 폴더에 저장되었습니다."
        duration_msg = f"총 소요시간: {duration.total_seconds():.2f}초"
        
        print(f"\n✅ {success_msg}")
        print(f"📁 {result_msg}")
        print(f"⏱️ {duration_msg}")
        
        self.logger.info(f"분석 완료 - 종료시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"총 소요시간: {duration.total_seconds():.2f}초")
        self.logger.info(f"결과 파일 저장 위치: {self.output_dir}")
        
        # 로그 파일 위치 알림
        print(f"📝 상세 로그는 '{log_path}'에 저장되었습니다.")

def main():
    """메인 함수"""
    logger.info("="*60)
    logger.info("통합 시각화 클러스터링 분석기 실행 시작")
    logger.info(f"로그 파일: {log_path}")
    logger.info("="*60)
    
    analyzer = IntegratedAnalyzer()
    analyzer.run_all_analysis()

if __name__ == "__main__":
    main()