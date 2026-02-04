#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
실시간 모니터링 대시보드 v3.0 (정리된 버전)
누적 분석 결과를 기반으로 HTML 대시보드 생성

작성자: AI Assistant
수정일: 2025년 1월
"""

import sqlite3
from datetime import datetime
import logging
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeDashboard:
    """실시간 모니터링 대시보드"""

    def __init__(self, db_path: str = "cumulative_analysis.db"):
        self.db_path = db_path
        self.output_file = "monitoring_dashboard.html"

    def get_detailed_metrics_from_json(self, book_name):
        """JSON 파일에서 PA/SA 세부 결과를 추출"""
        try:
            # 1. 먼저 cumulative_analysis_report.json에서 찾기 (새로운 구조)
            cumulative_report_path = "cumulative_analysis_report.json"
            if os.path.exists(cumulative_report_path):
                with open(cumulative_report_path, "r", encoding="utf-8") as f:
                    cumulative_data = json.load(f)

                # books_summary에서 해당 책 찾기
                for book in cumulative_data.get("books_summary", []):
                    if book["book_name"] == book_name:
                        if "pa_details" in book and "sa_details" in book:
                            return {
                                "pa": {
                                    "precision": book["pa_details"].get("precision", 0),
                                    "recall": book["pa_details"].get("recall", 0),
                                    "f1_score": book["pa_details"].get("f1_score", 0),
                                    "avg_similarity": book["pa_details"].get(
                                        "avg_similarity", 0
                                    ),
                                    "combined_similarity": book["pa_details"].get(
                                        "combined_similarity", 0
                                    ),
                                },
                                "sa": {
                                    "precision": book["sa_details"].get("precision", 0),
                                    "recall": book["sa_details"].get("recall", 0),
                                    "f1_score": book["sa_details"].get("f1_score", 0),
                                    "set_similarity": book["sa_details"].get(
                                        "set_similarity", 0
                                    ),
                                    "source_only_similarity": book["sa_details"].get(
                                        "source_similarity", 0
                                    ),
                                    "target_only_similarity": book["sa_details"].get(
                                        "target_similarity", 0
                                    ),
                                },
                            }

            # 2. 백업: xml_pipeline_results 폴더에서 찾기 (기존 방식)
            json_path = (
                f"../xml_pipeline_results/{book_name}/accuracy/accuracy_report.json"
            )

            if not os.path.exists(json_path):
                logger.warning(f"세부 분석 데이터를 찾을 수 없음: {json_path}")
                return None

            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            # xml_level_analysis에서 PA/SA 데이터 추출
            if "xml_level_analysis" in json_data:
                xml_analysis = json_data["xml_level_analysis"]
                details = {}

                # PA 메트릭
                if "pa_analysis" in xml_analysis:
                    pa_data = xml_analysis["pa_analysis"]
                    details["pa"] = {
                        "precision": pa_data.get("precision", 0),
                        "recall": pa_data.get("recall", 0),
                        "f1_score": pa_data.get("f1_score", 0),
                        "avg_similarity": pa_data.get("avg_similarity", 0),
                        "combined_similarity": pa_data.get(
                            "avg_combined_similarity", 0
                        ),
                    }

                # SA 메트릭
                if "sa_analysis" in xml_analysis:
                    sa_data = xml_analysis["sa_analysis"]
                    details["sa"] = {
                        "precision": sa_data.get("precision", 0),
                        "recall": sa_data.get("recall", 0),
                        "f1_score": sa_data.get("f1_score", 0),
                        "set_similarity": sa_data.get("avg_combined_similarity", 0),
                        "source_only_similarity": sa_data.get(
                            "avg_original_similarity", 0
                        ),
                        "target_only_similarity": sa_data.get(
                            "avg_translation_similarity", 0
                        ),
                    }

                return details

            return None

        except Exception as e:
            logger.error(f"JSON 메트릭 추출 오류 ({book_name}): {e}")
            return None

    def generate_dashboard(self):
        """HTML 대시보드 생성"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 전체 통계 조회
            cursor.execute("""
                SELECT
                    COUNT(*) as total_books,
                    SUM(total_paragraphs) as total_paragraphs,
                    AVG(pa_accuracy) as avg_pa_accuracy,
                    AVG(sa_accuracy) as avg_sa_accuracy,
                    AVG(embedding_similarity_avg) as avg_embedding_similarity
                FROM book_analysis
            """)

            stats = cursor.fetchone()

            # 개별 책 데이터 조회 (작가/역자/4부분류/시대 정보 포함)
            cursor.execute("""
                SELECT book_name, author, translator, quality_grade, pa_accuracy, sa_accuracy,
                       embedding_similarity_avg, total_paragraphs, analysis_date,
                       phrase_count, sa_result_count, global_source_similarity,
                       global_target_similarity, length_accuracy, sibu_classification, period
                FROM book_analysis
                ORDER BY analysis_date DESC
            """)

            books = cursor.fetchall()

            # 작가별/4부분류별/시대별 분포 조회
            cursor.execute("""
                SELECT author, COUNT(*) as count
                FROM book_analysis
                WHERE author IS NOT NULL AND author != '미상'
                GROUP BY author
                ORDER BY count DESC
                LIMIT 10
            """)
            author_dist = cursor.fetchall()

            # 4부분류별 분포 조회
            cursor.execute("""
                SELECT sibu_classification, COUNT(*) as count
                FROM book_analysis
                WHERE sibu_classification IS NOT NULL
                GROUP BY sibu_classification
                ORDER BY count DESC
            """)
            sibu_dist = cursor.fetchall()

            # 시대별 분포 조회
            cursor.execute("""
                SELECT period, COUNT(*) as count
                FROM book_analysis
                WHERE period IS NOT NULL AND period != '미상'
                GROUP BY period
                ORDER BY count DESC
                LIMIT 10
            """)
            period_dist = cursor.fetchall()

            # 품질 등급별 분포 조회
            cursor.execute("""
                SELECT quality_grade, COUNT(*) as count
                FROM book_analysis
                GROUP BY quality_grade
                ORDER BY
                    CASE quality_grade
                        WHEN 'A+' THEN 1 WHEN 'A' THEN 2 WHEN 'B+' THEN 3
                        WHEN 'B' THEN 4 WHEN 'C+' THEN 5 WHEN 'C' THEN 6
                        WHEN 'D' THEN 7 ELSE 8
                    END
            """)

            quality_dist = cursor.fetchall()
            conn.close()

            # HTML 생성 (새로운 메타데이터 포함)
            html_content = self._generate_html_content(
                stats, books, quality_dist, author_dist, sibu_dist, period_dist
            )

            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(html_content)

            logger.info(f"대시보드 생성 완료: {self.output_file}")
            return True

        except Exception as e:
            logger.error(f"대시보드 생성 실패: {e}")
            return False

    def _generate_html_content(
        self, stats, books, quality_dist, author_dist, sibu_dist, period_dist
    ):
        """HTML 콘텐츠 생성"""

        # BookMetadataExtractor 임포트 및 초기화
        try:
            from book_metadata_extractor import BookMetadataExtractor

            metadata_extractor = BookMetadataExtractor()
        except ImportError:
            metadata_extractor = None

        # 통계 데이터 처리
        total_books = stats[0] if stats[0] else 0
        total_paragraphs = stats[1] if stats[1] else 0
        avg_pa = (stats[2] * 100) if stats[2] else 0
        avg_sa = (stats[3] * 100) if stats[3] else 0
        avg_embedding = (stats[4] * 100) if stats[4] else 0

        current_time = datetime.now().strftime("%Y년 %m월 %d일 %H:%M:%S")

        # 품질 등급별 색상 매핑
        grade_colors = {
            "A+": "#28a745",
            "A": "#20c997",
            "B+": "#17a2b8",
            "B": "#007bff",
            "C+": "#ffc107",
            "C": "#fd7e14",
            "D": "#dc3545",
            "F": "#6c757d",
        }

        # 개별 책 행 생성 (4부분류/시대 정보 포함)
        book_rows = ""
        for book in books:
            (
                book_name,
                author,
                translator,
                grade,
                pa_acc,
                sa_acc,
                embed_sim,
                paragraphs,
                analysis_date,
                phrase_count,
                sa_result_count,
                global_source_sim,
                global_target_sim,
                length_acc,
                sibu_classification,
                period,
            ) = book

            grade_color = grade_colors.get(grade, "#6c757d")

            # 작가/역자 정보 처리
            display_author = author if author and author != "미상" else "미상"
            display_translator = translator if translator else "한국고전번역원"

            # 4부분류/시대 정보 처리
            display_sibu = sibu_classification if sibu_classification else "未詳"
            display_period = period if period and period != "미상" else "미상"

            # 복수 역자 처리 (세미콜론 구분)
            translator_display = display_translator
            if "; " in display_translator:
                translator_parts = display_translator.split("; ")
                if len(translator_parts) > 2:
                    translator_display = (
                        f"{translator_parts[0]} 외 {len(translator_parts)-1}명"
                    )
                else:
                    translator_display = " & ".join(translator_parts)

            # 세부 항목들이 None일 경우 기본값 설정
            phrase_count = phrase_count or 0
            sa_result_count = sa_result_count or 0
            global_source_sim = global_source_sim or 0.0
            global_target_sim = global_target_sim or 0.0
            length_acc = length_acc or 0.0

            # 세부 내역 HTML 생성 (4부분류/시대 정보 포함)
            detail_content = f"""
                <div class="metadata-section">
                    <h4>📖 서지 정보</h4>
                    <div class="separator">--------------------------------------------------</div>
                    <div class="metric-row"><strong>작가:</strong> {display_author} ({display_period})</div>
                    <div class="metric-row"><strong>역자:</strong> {display_translator}</div>
                    <div class="metric-row"><strong>4부분류:</strong> {display_sibu}</div>
                    <div class="metric-row"><strong>원문구수:</strong> {phrase_count:,}개</div>
                    <div class="metric-row"><strong>번역구수:</strong> {sa_result_count:,}개</div>
                    <div class="metric-row"><strong>길이 정확도:</strong> {length_acc * 100:.1f}%</div>
                    <div class="metric-row"><strong>원문 유사도:</strong> {global_source_sim * 100:.1f}%</div>
                    <div class="metric-row"><strong>번역 유사도:</strong> {global_target_sim * 100:.1f}%</div>
                </div>
            """

            # JSON에서 세부 메트릭 가져오기
            detail_metrics = self.get_detailed_metrics_from_json(book_name)

            # 세부 내역 HTML 생성
            detail_content = ""
            if detail_metrics:
                # PA 결과
                if "pa" in detail_metrics:
                    pa = detail_metrics["pa"]
                    detail_content += f"""
                        <div class="analysis-section">
                            <h4>📊 PA (Paragraph Analysis) 결과</h4>
                            <div class="separator">--------------------------------------------------</div>
                            <div class="metric-row">• Precision: {pa.get('precision', 0):.4f}</div>
                            <div class="metric-row">• Recall: {pa.get('recall', 0):.4f}</div>
                            <div class="metric-row">• F1 Score: {pa.get('f1_score', 0):.4f}</div>
                    """
                    if "avg_similarity" in pa:
                        detail_content += f"""
                            <div class="similarity-section">
                                <div class="metric-subsection">📈 유사도 세부 분석:</div>
                                <div class="metric-indent">- 평균 유사도: {pa.get('avg_similarity', 0):.4f}</div>
                                <div class="metric-indent">- 결합 유사도: {pa.get('combined_similarity', 0):.4f}</div>
                            </div>
                        """
                    detail_content += "</div>"

                # SA 결과
                if "sa" in detail_metrics:
                    sa = detail_metrics["sa"]
                    detail_content += f"""
                        <div class="analysis-section">
                            <h4>📊 SA (Sentence Analysis) 결과</h4>
                            <div class="separator">--------------------------------------------------</div>
                            <div class="metric-row">• Precision: {sa.get('precision', 0):.4f}</div>
                            <div class="metric-row">• Recall: {sa.get('recall', 0):.4f}</div>
                            <div class="metric-row">• F1 Score: {sa.get('f1_score', 0):.4f}</div>
                    """
                    if "set_similarity" in sa:
                        detail_content += f"""
                            <div class="similarity-section">
                                <div class="metric-subsection">📈 유사도 세부 분석:</div>
                                <div class="metric-indent">- 한 세트 유사도: {sa.get('set_similarity', 0):.4f}</div>
                                <div class="metric-indent">- 원문만 유사도: {sa.get('source_only_similarity', 0):.4f}</div>
                                <div class="metric-indent">- 번역문만 유사도: {sa.get('target_only_similarity', 0):.4f}</div>
                            </div>
                        """
                    detail_content += "</div>"
            else:
                detail_content = (
                    "<div class='no-data'>세부 분석 데이터를 찾을 수 없습니다.</div>"
                )

            book_rows += f"""
                <tr onclick="toggleDetails('{book_name}')" style="cursor: pointer;">
                    <td><strong>{book_name}</strong></td>
                    <td style="font-size: 0.8rem;">{display_author}<br><small style="color: #666;">({display_period})</small></td>
                    <td style="font-size: 0.8rem;">{translator_display}</td>
                    <td><span style="background-color: #e3f2fd; color: #1976d2; padding: 2px 6px; border-radius: 10px; font-size: 0.75rem;">{display_sibu}</span></td>
                    <td><span class="badge" style="background-color: {grade_color}; color: white;">{grade}</span></td>
                    <td>{pa_acc * 100:.1f}%</td>
                    <td>{sa_acc * 100:.1f}%</td>
                    <td>{embed_sim * 100:.1f}%</td>
                    <td>{paragraphs:,}</td>
                    <td>{analysis_date[:10]}</td>
                </tr>
                <tr id="details_{book_name}" class="detail-row" style="display: none;">
                    <td colspan="10" class="detail-content">
                        {detail_content}
                    </td>
                </tr>
            """

        # 작가별 통계 생성
        author_stats = ""
        for author, count in author_dist[:10]:  # 상위 10명만 표시
            author_stats += f"""
                <div class="performance-indicator">
                    <span class="indicator-label">{author}</span>
                    <span class="indicator-value">{count}권</span>
                </div>
            """

        # 4부분류별 통계 생성
        sibu_stats = ""
        for sibu, count in sibu_dist:
            sibu_stats += f"""
                <div class="performance-indicator">
                    <span class="indicator-label">{sibu}</span>
                    <span class="indicator-value">{count}권</span>
                </div>
            """

        # 시대별 통계 생성
        period_stats = ""
        for period, count in period_dist[:10]:  # 상위 10개 시대만 표시
            period_display = (
                period.split("(")[0] if "(" in period else period
            )  # 괄호 내용 제거하여 간단히 표시
            period_stats += f"""
                <div class="performance-indicator">
                    <span class="indicator-label">{period_display}</span>
                    <span class="indicator-value">{count}권</span>
                </div>
            """

        # 작가별 분포 차트 데이터
        author_chart_data = []
        author_chart_labels = []
        author_colors = [
            "#FF6384",
            "#36A2EB",
            "#FFCE56",
            "#4BC0C0",
            "#9966FF",
            "#FF9F40",
            "#FF6384",
            "#C9CBCF",
            "#4BC0C0",
            "#36A2EB",
        ]

        for i, (author, count) in enumerate(author_dist[:8]):  # 상위 8명만
            author_chart_labels.append(f"'{author}'")
            author_chart_data.append(str(count))

        # 나머지 작가들은 '기타'로 묶기
        if len(author_dist) > 8:
            other_count = sum(count for _, count in author_dist[8:])
            if other_count > 0:
                author_chart_labels.append("'기타'")
                author_chart_data.append(str(other_count))
        chart_data = []
        chart_labels = []
        chart_colors = []

        for grade, count in quality_dist:
            chart_labels.append(f"'{grade}'")
            chart_data.append(str(count))
            chart_colors.append(f"'{grade_colors.get(grade, '#6c757d')}'")

        html_template = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>한문 처리 파이프라인 모니터링 대시보드</title>
    <link href="https://fonts.googleapis.com/css2?family=Sarasa+Fixed+K:wght@400;500;600;700&family=Sarasa+Gothic+K:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Sarasa Fixed K', 'Sarasa Gothic K', 'D2Coding', 'Consolas', monospace;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }}

        .container {{ max-width: 95vw; margin: 0 auto; padding: 15px; }}

        .header {{
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }}

        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}

        .last-updated {{ font-size: 1.1rem; opacity: 0.9; }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .stat-card {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.3s ease;
        }}

        .stat-card:hover {{ transform: translateY(-5px); }}

        .stat-value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}

        .stat-label {{
            font-size: 1.1rem;
            color: #7f8c8d;
            font-weight: 500;
        }}

        .content-grid {{
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 30px;
            margin-bottom: 30px;
        }}

        .main-content {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }}

        .sidebar {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }}

        .section-title {{
            font-size: 1.5rem;
            margin-bottom: 20px;
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}

        .table-container {{
            overflow-x: auto;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        table {{ width: 100%; border-collapse: collapse; background: white; }}

        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            font-size: 0.9rem;
            position: relative;
            cursor: pointer;
            user-select: none;
        }}

        th:hover {{ background: linear-gradient(135deg, #5a6fd8 0%, #6b4190 100%); }}

        th.sortable::after {{
            content: '⇅';
            position: absolute;
            right: 10px;
            opacity: 0.5;
        }}

        th.sort-asc::after {{ content: '▲'; opacity: 1; }}
        th.sort-desc::after {{ content: '▼'; opacity: 1; }}

        .quality-grade {{
            cursor: help;
            position: relative;
        }}

        .tooltip {{
            visibility: hidden;
            width: 300px;
            background-color: #333;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1000;
            bottom: 125%;
            left: 50%;
            margin-left: -150px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.8rem;
            line-height: 1.4;
        }}

        .tooltip::after {{
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #333 transparent transparent transparent;
        }}

        .quality-grade:hover .tooltip {{ visibility: visible; opacity: 1; }}

        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
            font-size: 0.9rem;
        }}

        tr:hover {{ background-color: #e3f2fd !important; }}

        .badge {{
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .chart-container {{ position: relative; height: 300px; margin-top: 20px; }}

        .performance-indicator {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 15px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #3498db;
        }}

        .indicator-label {{ font-weight: 600; color: #2c3e50; }}

        .indicator-value {{
            font-size: 1.2rem;
            font-weight: bold;
            color: #27ae60;
        }}

        .detail-row {{ background: #f8f9fa !important; }}

        .detail-content {{
            padding: 15px 20px !important;
            font-family: 'Sarasa Fixed K', 'Sarasa Gothic K', 'D2Coding', 'Monaco', 'Consolas', monospace;
            background-color: #f8f9fa;
        }}

        .analysis-section {{
            margin-bottom: 20px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}

        .analysis-section h4 {{
            margin: 0 0 10px 0;
            color: #2c3e50;
            font-size: 1.1rem;
        }}

        .separator {{
            color: #7f8c8d;
            margin: 8px 0;
            font-size: 0.9rem;
        }}

        .metric-row {{
            margin: 5px 0;
            color: #2c3e50;
            font-weight: 500;
        }}

        .similarity-section {{
            margin-top: 10px;
            padding-left: 10px;
        }}

        .metric-subsection {{
            margin: 8px 0 5px 0;
            color: #27ae60;
            font-weight: 600;
        }}

        .metric-indent {{
            margin-left: 20px;
            color: #34495e;
        }}

        .no-data {{
            color: #e74c3c;
            font-style: italic;
            text-align: center;
            padding: 20px;
        }}

        .detail-row:hover {{ background-color: #f8f9fa !important; }}

        @media (max-width: 1200px) {{
            .content-grid {{ grid-template-columns: 1fr; }}
        }}

        @media (max-width: 768px) {{
            .stats-grid {{ grid-template-columns: 1fr; }}
            .container {{ padding: 10px; }}
            .header h1 {{ font-size: 2rem; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📚 한문 처리 파이프라인 모니터링 대시보드</h1>
            <div class="last-updated">마지막 업데이트: {current_time}</div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{total_books:,}</div>
                <div class="stat-label">처리 완료 서종</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_paragraphs:,}</div>
                <div class="stat-label">총 처리 문단</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_pa:.1f}%</div>
                <div class="stat-label">평균 PA 정확도</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_sa:.1f}%</div>
                <div class="stat-label">평균 SA 정확도</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_embedding:.1f}%</div>
                <div class="stat-label">평균 임베딩 유사도</div>
            </div>
        </div>

        <div class="content-grid">
            <div class="main-content">
                <h2 class="section-title">📖 처리 완료 서종 목록 (클릭하여 세부 정보 확인)</h2>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th class="sortable" onclick="sortTable(0)">서종명</th>
                                <th class="sortable" onclick="sortTable(1)">작가 (시대)</th>
                                <th class="sortable" onclick="sortTable(2)">역자</th>
                                <th class="sortable" onclick="sortTable(3)">4부분류</th>
                                <th class="sortable quality-grade" onclick="sortTable(4)">
                                    품질등급
                                    <div class="tooltip">
                                        <strong>품질등급 계산 공식:</strong><br>
                                        (PA정확도 × 0.5 + SA정확도 × 0.5) × 100<br><br>
                                        <strong>등급 기준:</strong><br>
                                        A+ (85-100): 최고 품질 | A (75-84): 우수<br>
                                        B+ (65-74): 양호+ | B (60-64): 양호<br>
                                        C+ (55-59): 보통+ | C (50-54): 보통<br>
                                        D (45-49): 부족 | F (0-44): 불량
                                    </div>
                                </th>
                                <th class="sortable" onclick="sortTable(5)">PA 정확도</th>
                                <th class="sortable" onclick="sortTable(6)">SA 정확도</th>
                                <th class="sortable" onclick="sortTable(7)">임베딩 유사도</th>
                                <th class="sortable" onclick="sortTable(8)">문단수</th>
                                <th class="sortable" onclick="sortTable(9)">분석일</th>
                            </tr>
                        </thead>
                        <tbody>
                            {book_rows}
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="sidebar">
                <h2 class="section-title">📊 품질 등급 분포</h2>
                <div class="chart-container">
                    <canvas id="qualityChart"></canvas>
                </div>

                <h3 style="margin-top: 30px; margin-bottom: 15px; color: #2c3e50;">🎯 성능 지표</h3>
                <div class="performance-indicator">
                    <span class="indicator-label">전체 평균 정확도</span>
                    <span class="indicator-value">{(avg_pa + avg_sa) / 2:.1f}%</span>
                </div>
                <div class="performance-indicator">
                    <span class="indicator-label">처리 서종 수</span>
                    <span class="indicator-value">{total_books}권</span>
                </div>
                <div class="performance-indicator">
                    <span class="indicator-label">총 처리 문단</span>
                    <span class="indicator-value">{total_paragraphs:,}</span>
                </div>

                <h3 style="margin-top: 30px; margin-bottom: 15px; color: #2c3e50;">👥 주요 작가별 서종 수</h3>
                {author_stats}

                <h3 style="margin-top: 30px; margin-bottom: 15px; color: #2c3e50;">📈 작가별 분포</h3>
                <div class="chart-container">
                    <canvas id="authorChart"></canvas>
                </div>

                <h3 style="margin-top: 30px; margin-bottom: 15px; color: #2c3e50;">📚 4부분류별 서종 수</h3>
                {sibu_stats}

                <h3 style="margin-top: 30px; margin-bottom: 15px; color: #2c3e50;">🏛️ 주요 시대별 서종 수</h3>
                {period_stats}
            </div>
            </div>
        </div>
    </div>

    <script>
        // 품질 등급 차트
        const ctx = document.getElementById('qualityChart').getContext('2d');
        const qualityChart = new Chart(ctx, {{
            type: 'doughnut',
            data: {{
                labels: [{', '.join(chart_labels)}],
                datasets: [{{
                    data: [{', '.join(chart_data)}],
                    backgroundColor: [{', '.join(chart_colors)}],
                    borderWidth: 3,
                    borderColor: '#fff'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'bottom',
                        labels: {{
                            padding: 20,
                            font: {{ size: 12 }}
                        }}
                    }}
                }}
            }}
        }});

        // 세부 정보 토글 함수
        function toggleDetails(bookName) {{
            const detailRow = document.getElementById('details_' + bookName);
            if (detailRow.style.display === 'none') {{
                detailRow.style.display = 'table-row';
            }} else {{
                detailRow.style.display = 'none';
            }}
        }}

        // 테이블 정렬 함수
        let currentSort = {{ column: -1, ascending: true }};

        function sortTable(columnIndex) {{
            const table = document.querySelector('table tbody');
            const rows = Array.from(table.querySelectorAll('tr')).filter(row => !row.id.startsWith('details_'));
            const headers = document.querySelectorAll('th');

            // 정렬 방향 결정
            const ascending = currentSort.column !== columnIndex || !currentSort.ascending;
            currentSort = {{ column: columnIndex, ascending: ascending }};

            // 헤더 스타일 업데이트
            headers.forEach(header => {{
                header.classList.remove('sort-asc', 'sort-desc');
            }});
            headers[columnIndex].classList.add(ascending ? 'sort-asc' : 'sort-desc');

            // 정렬 수행
            rows.sort((a, b) => {{
                const aValue = getCellValue(a, columnIndex);
                const bValue = getCellValue(b, columnIndex);

                if (columnIndex === 1) {{ // 품질등급
                    return compareGrades(aValue, bValue) * (ascending ? 1 : -1);
                }} else if (columnIndex === 2 || columnIndex === 3 || columnIndex === 4) {{ // PA, SA, 임베딩 유사도
                    return (parseFloat(aValue) - parseFloat(bValue)) * (ascending ? 1 : -1);
                }} else if (columnIndex === 5) {{ // 문단수
                    return (parseInt(aValue) - parseInt(bValue)) * (ascending ? 1 : -1);
                }} else if (columnIndex === 6) {{ // 분석일
                    return (new Date(aValue) - new Date(bValue)) * (ascending ? 1 : -1);
                }} else {{ // 서종명
                    return aValue.localeCompare(bValue) * (ascending ? 1 : -1);
                }}
            }});

            // 정렬된 행들을 테이블에 다시 추가
            rows.forEach(row => {{
                table.appendChild(row);
                // 세부 정보 행도 함께 이동
                const bookName = row.cells[0].textContent.trim();
                const detailRow = document.getElementById('details_' + bookName);
                if (detailRow) {{
                    table.appendChild(detailRow);
                }}
            }});
        }}

        function getCellValue(row, columnIndex) {{
            const cell = row.cells[columnIndex];
            return cell ? cell.textContent.trim() : '';
        }}

        function compareGrades(a, b) {{
            const gradeOrder = {{ 'A+': 8, 'A': 7, 'B+': 6, 'B': 5, 'C+': 4, 'C': 3, 'D': 2, 'F': 1 }};
            return (gradeOrder[a] || 0) - (gradeOrder[b] || 0);
        }}

        // 자동 새로고침 (5분마다)
        setTimeout(() => {{
            location.reload();
        }}, 300000);
    </script>
</body>
</html>
        """

        return html_template

def main():
    """메인 함수"""
    print("🖥️  실시간 모니터링 대시보드 생성기 v3.0")
    print("=" * 40)

    try:
        dashboard = RealTimeDashboard()

        if dashboard.generate_dashboard():
            print("✅ 대시보드 생성 완료!")
            print(f"📊 파일: {dashboard.output_file}")
            print("🌐 웹브라우저에서 열어서 확인하세요.")
        else:
            print("❌ 대시보드 생성 실패")

    except Exception as e:
        print(f"💥 오류 발생: {e}")

if __name__ == "__main__":
    main()
