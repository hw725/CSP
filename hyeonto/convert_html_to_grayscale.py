#!/usr/bin/env python3
"""
기존 HTML 시각화를 흑백 인쇄용으로 변환

모든 HTML 시각화 파일의 색상을 흑백으로 변환합니다.
원본은 유지하고 _bw.html 파일을 새로 생성합니다.
"""

import re
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
REPORTS_DIR = BASE_DIR / "reports"

# 컬러 → 흑백 매핑 (4단계 명도: 검정 0%, 진회색 40%, 밝은회색 75%, 흰색 100%)
# 클러스터별 명확한 구분을 위해 명도 차이 극대화
COLOR_REPLACEMENTS = {
    # 클러스터 0 (빨강/핑크) → 검정 (0%)
    '#e94560': '#000000',
    'rgba(233, 69, 96, 0.8)': 'rgba(0, 0, 0, 0.9)',
    'rgba(233, 69, 96, 0.6)': 'rgba(0, 0, 0, 0.7)',
    
    # 클러스터 1 (녹색/청록) → 진회색 (40%)
    '#4ecca3': '#666666',
    'rgba(78, 204, 163, 0.8)': 'rgba(102, 102, 102, 0.9)',
    'rgba(78, 204, 163, 0.6)': 'rgba(102, 102, 102, 0.7)',
    
    # 클러스터 2 (노랑/주황) → 밝은회색 (75%)
    '#ffc93c': '#BBBBBB',
    '#ff8c42': '#BBBBBB',
    'rgba(255, 201, 60, 0.8)': 'rgba(187, 187, 187, 0.9)',
    'rgba(255, 140, 66, 0.8)': 'rgba(187, 187, 187, 0.9)',
    
    # 클러스터 3 (파랑/보라) → 흰색 (100%, 검정 테두리로 구분)
    '#22577a': '#EEEEEE',
    '#ff6b6b': '#EEEEEE', 
    'rgba(34, 87, 122, 0.8)': 'rgba(238, 238, 238, 0.9)',
    'rgba(255, 107, 107, 0.8)': 'rgba(238, 238, 238, 0.9)',
    
    # 추가 색상들
    '#45b7aa': '#888888',
    '#38a3a5': '#AAAAAA',
    'rgba(69, 183, 170, 0.8)': 'rgba(136, 136, 136, 0.9)',
    'rgba(56, 163, 165, 0.8)': 'rgba(170, 170, 170, 0.9)',
    
    # 배경 그라데이션 → 흰색
    'linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%)': '#ffffff',
    
    # 텍스트 색상
    'color: #fff': 'color: #000',
    'color: #888': 'color: #333',
    "color: '#fff'": "color: '#000'",
    'font: { color: \'#fff\'': 'font: { color: \'#000\'',
    
    # 링크/오버레이 색상
    'rgba(255,255,255,0.2)': 'rgba(80,80,80,0.4)',
    'rgba(255,255,255,0.3)': 'rgba(0,0,0,0.3)',
    'rgba(255,255,255,0.05)': 'rgba(0,0,0,0.08)',
    'rgba(255,255,255,0.1)': 'rgba(0,0,0,0.15)',
    
    # 타이틀 그라데이션 스타일 제거
    'background: linear-gradient(90deg, #e94560, #4ecca3); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;': 'color: #222;',
}



def convert_to_grayscale(html_content: str) -> str:
    """HTML 콘텐츠의 색상을 흑백으로 변환 + 마커 심볼 변경"""
    result = html_content
    
    # 1. 색상 변환
    for color_from, color_to in COLOR_REPLACEMENTS.items():
        result = result.replace(color_from, color_to)
    
    # paper_bgcolor 투명 → 흰색
    result = result.replace("paper_bgcolor: 'rgba(0,0,0,0)'", "paper_bgcolor: '#ffffff'")
    
    # 2. Plotly 마커 심볼 추가 (클러스터별 다른 모양)
    # 클러스터 0~3에 대해 다른 심볼 지정
    symbol_mapping = ['circle', 'square', 'diamond', 'triangle-up', 'cross', 'star', 'hexagon', 'pentagon']
    
    # 산점도 trace에 symbol 속성 추가
    # 패턴: mode: "markers" 가 있는 trace 찾기
    import re
    
    # Plotly.newPlot 호출 후 심볼+색상 변환 스크립트 삽입
    symbol_script = '''
    <script>
    // 흑백 인쇄 최적화: 클러스터별 마커 심볼 + 명도 차이 극대화
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(function() {
            const plots = document.querySelectorAll('.plotly-graph-div, [id^="sankey"], #plot, .js-plotly-plot');
            plots.forEach(function(plot) {
                if (plot._fullData) {
                    // 심볼: 원, 사각, 다이아, 삼각, X, 별, 육각, 오각
                    const symbols = ['circle', 'square', 'diamond', 'triangle-up', 'cross', 'star', 'hexagon', 'pentagon'];
                    // 명도: 검정(0%), 진회색(40%), 밝은회색(75%), 흰색(95%)
                    const colors = ['#000000', '#666666', '#BBBBBB', '#EEEEEE', '#333333', '#999999', '#DDDDDD', '#444444'];
                    
                    plot._fullData.forEach(function(trace, idx) {
                        if (trace.mode && trace.mode.includes('markers')) {
                            const update = {
                                'marker.symbol': symbols[idx % symbols.length],
                                'marker.color': colors[idx % colors.length],
                                'marker.size': 10,
                                'marker.line.width': 2,
                                'marker.line.color': '#000000',
                                'marker.opacity': 0.9
                            };
                            Plotly.restyle(plot, update, [idx]);
                        }
                    });
                }
            });
        }, 500);
    });
    </script>
    '''

    
    # </body> 앞에 스크립트 삽입
    result = result.replace('</body>', symbol_script + '\n</body>')
    
    # 3. 타이틀에 (흑백) 표시 추가
    result = re.sub(r'<title>([^<]+)</title>', r'<title>\1 (흑백)</title>', result)
    
    # 4. 배경색 밝게 (인쇄용)
    result = result.replace('background: #0f0c29', 'background: #ffffff')
    result = result.replace('background: #1a1a2e', 'background: #ffffff')
    result = result.replace('background: #16213e', 'background: #ffffff')
    
    return result



def process_file(html_path: Path) -> bool:
    """단일 HTML 파일 변환 (원본을 _color로 백업 후 덮어쓰기)"""
    try:
        content = html_path.read_text(encoding='utf-8')
        grayscale_content = convert_to_grayscale(content)
        
        # 원본을 _color.html로 백업
        color_path = html_path.parent / (html_path.stem + '_color.html')
        color_path.write_text(content, encoding='utf-8')
        
        # 원본 파일을 흑백으로 덮어쓰기
        html_path.write_text(grayscale_content, encoding='utf-8')
        
        print(f"  ? {html_path.name} (컬러 백업: {color_path.name})")
        return True
    except Exception as e:
        print(f"  ? {html_path.name}: {e}")
        return False



def main():
    print("="*70)
    print("? HTML 시각화 흑백 변환")
    print("="*70)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 처리할 HTML 파일 목록 (이미 _bw.html인 것 제외)
    html_files = []
    
    # 주요 시각화 파일들
    main_viz = [
        REPORTS_DIR / "k4_embedding_overlay_2d.html",
        REPORTS_DIR / "k4_embedding_overlay_3d.html",
        REPORTS_DIR / "k4_sentence_phrase_overlay_normalized.html",
        REPORTS_DIR / "dashboard.html",
    ]
    
    # exploratory 폴더의 모든 HTML
    exploratory_viz = list((REPORTS_DIR / "exploratory").rglob("*.html"))
    
    # sankey_diagrams (이미 _bw가 있으면 제외)
    sankey_viz = [f for f in (REPORTS_DIR / "sankey_diagrams").glob("*.html") 
                  if '_bw' not in f.name]
    
    html_files = main_viz + exploratory_viz + sankey_viz
    html_files = [f for f in html_files if f.exists() and '_bw' not in f.name]
    
    print(f"\n? 변환할 파일: {len(html_files)}개")
    
    success_count = 0
    for html_file in html_files:
        if process_file(html_file):
            success_count += 1
    
    print(f"\n{'='*70}")
    print(f"? 변환 완료: {success_count}/{len(html_files)}개")
    print("="*70)


if __name__ == "__main__":
    main()
