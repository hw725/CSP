#!/usr/bin/env python3
"""
마크다운 보고서를 HTML로 변환하여 대시보드에서 직접 표시 가능하게 함
"""

import markdown
from pathlib import Path

def md_to_html(md_path, html_path):
    """마크다운을 HTML로 변환"""
    
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    
    # 마크다운을 HTML로 변환
    html_content = markdown.markdown(md_content, extensions=['tables', 'extra'])
    
    # HTML 템플릿으로 감싸기
    full_html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>마커 분류 분석 보고서</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            margin: 20px 0;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin: 30px 0 15px 0;
        }}
        h3 {{
            color: #7f8c8d;
            margin: 15px 0 10px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.95em;
        }}
        th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        ul {{
            margin: 10px 0 10px 20px;
        }}
        li {{
            margin: 5px 0;
        }}
        strong {{
            color: #2c3e50;
        }}
        .reference {{
            background-color: #ecf0f1;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 20px 0;
            font-size: 0.95em;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    <div class="container">
        {html_content}
        <div class="reference">
            <strong>📚 근거:</strong> 임규직, 박문호, 이삼환 등 한문 구두학 전문가 분류<br>
            <strong>🔗 출처:</strong> classified_markers.json + compound_tags.json
        </div>
    </div>
</body>
</html>"""
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    
    print(f"✅ HTML 보고서 생성: {html_path}")

def main():
    base_path = Path("hyeonto/report_1-1")
    
    files = [
        (
            "sentence_k3_normalized/sentence_clusters_재분류_보고서.md",
            "sentence_k3_normalized/sentence_clusters_재분류_보고서.html"
        ),
        (
            "phrase_k3_normalized/phrase_clusters_재분류_보고서.md",
            "phrase_k3_normalized/phrase_clusters_재분류_보고서.html"
        ),
    ]
    
    for md_file, html_file in files:
        md_path = base_path / md_file
        html_path = base_path / html_file
        
        if md_path.exists():
            md_to_html(md_path, html_path)
        else:
            print(f"⚠️ {md_path} 파일 없음")

if __name__ == "__main__":
    main()
