"""
가중치 민감도 및 영가설 검증 HTML 시각화 생성

run_validation_analysis.py의 결과를 시각화하는 HTML 파일 생성:
1. 가중치 민감도 분석 차트 (바 차트)
2. 영가설 검정 결과 (효과 크기 시각화)
"""
import json
from pathlib import Path
from datetime import datetime

VALIDATION_DIR = Path(__file__).parent / "reports" / "validation"

# 흑백 인쇄용 스타일
BW_STYLE = """
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: 'Pretendard', 'Noto Sans KR', sans-serif;
        background: #ffffff;
        min-height: 100vh;
        color: #333;
        padding: 20px;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    h1 {
        text-align: center;
        font-size: 2rem;
        margin-bottom: 10px;
        color: #333;
    }
    h2 {
        font-size: 1.4rem;
        margin: 30px 0 15px 0;
        color: #333;
        border-bottom: 2px solid #333;
        padding-bottom: 5px;
    }
    .subtitle { text-align: center; color: #666; margin-bottom: 20px; }
    .chart-container {
        background: #f9f9f9;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 20px;
        margin: 20px 0;
    }
    .chart { width: 100%; height: 400px; }
    .result-box {
        background: #f0f0f0;
        border: 2px solid #333;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
    }
    .verdict-pass {
        background: #e8f5e9;
        border-color: #2e7d32;
    }
    .verdict-fail {
        background: #ffebee;
        border-color: #c62828;
    }
    .metric { margin: 10px 0; }
    .metric-label { font-weight: bold; color: #333; }
    .metric-value { font-size: 1.2rem; }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
    }
    th, td {
        border: 1px solid #333;
        padding: 10px;
        text-align: center;
    }
    th { background: #e0e0e0; }
    @media print {
        body { background: white !important; }
        .chart-container { break-inside: avoid; }
    }
"""


def generate_weight_sensitivity_html(output_path: Path):
    """가중치 민감도 분석 HTML 시각화 생성"""
    
    # 데이터 로드
    json_path = VALIDATION_DIR / "weight_sensitivity" / "weight_sensitivity_summary.json"
    if not json_path.exists():
        print(f"❌ 데이터 파일 없음: {json_path}")
        return False
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scenarios = data['scenarios']
    conclusion = data['conclusion']
    
    # 시나리오 이름, 값 추출
    names = [s['name'] for s in scenarios]
    avg_canon = [s['avg_weighted_canonicity'] for s in scenarios]
    max_canon = [s['max_weighted_canonicity'] for s in scenarios]
    entropy = [s['avg_genre_entropy'] for s in scenarios]
    
    html_content = f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>가중치 민감도 분석 시각화 (흑백)</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>{BW_STYLE}</style>
</head>
<body>
    <div class="container">
        <h1>📊 가중치 민감도 분석 시각화</h1>
        <p class="subtitle">Sentence/Phrase K=4 클러스터 | {datetime.now().strftime("%Y-%m-%d")}</p>
        
        <h2>1. 시나리오별 Canonicity 비교</h2>
        <div class="chart-container">
            <div id="canonicity-chart" class="chart"></div>
        </div>
        
        <h2>2. 시나리오별 장르 엔트로피</h2>
        <div class="chart-container">
            <div id="entropy-chart" class="chart"></div>
        </div>
        
        <h2>3. 핵심 발견</h2>
        <div class="result-box">
            <div class="metric">
                <span class="metric-label">Uniform→Strong Canonicity 변화:</span>
                <span class="metric-value">+{conclusion['canonicity_delta_uniform_to_strong']:.2f}%p</span>
            </div>
            <div class="metric">
                <span class="metric-label">Uniform→Strong 엔트로피 변화:</span>
                <span class="metric-value">{conclusion['entropy_delta_uniform_to_strong']:.4f}</span>
            </div>
            <div class="metric">
                <span class="metric-label">권장 시나리오:</span>
                <span class="metric-value">{data['recommended_scenario'].capitalize()}</span>
            </div>
        </div>
        
        <h2>4. 시나리오 상세</h2>
        <table>
            <tr>
                <th>시나리오</th>
                <th>사서 가중치</th>
                <th>평균 Canonicity</th>
                <th>최대 Canonicity</th>
                <th>장르 엔트로피</th>
            </tr>
            {"".join(f'''
            <tr>
                <td>{s['name']}</td>
                <td>-</td>
                <td>{s['avg_weighted_canonicity']:.2f}%</td>
                <td>{s['max_weighted_canonicity']:.2f}%</td>
                <td>{s['avg_genre_entropy']:.4f}</td>
            </tr>''' for s in scenarios)}
        </table>
    </div>
    
    <script>
        // Canonicity 바 차트
        const names = {json.dumps(names)};
        const avgCanon = {json.dumps(avg_canon)};
        const maxCanon = {json.dumps(max_canon)};
        
        Plotly.newPlot('canonicity-chart', [
            {{
                x: names,
                y: avgCanon,
                type: 'bar',
                name: '평균 Canonicity',
                marker: {{ color: '#333' }}
            }},
            {{
                x: names,
                y: maxCanon,
                type: 'bar',
                name: '최대 Canonicity',
                marker: {{ color: '#999', pattern: {{ shape: '/' }} }}
            }}
        ], {{
            barmode: 'group',
            paper_bgcolor: '#f9f9f9',
            plot_bgcolor: '#f9f9f9',
            xaxis: {{ title: '시나리오', color: '#333' }},
            yaxis: {{ title: 'Canonicity (%)', color: '#333' }},
            legend: {{ font: {{ color: '#333' }} }},
            margin: {{ t: 30, b: 60, l: 60, r: 30 }}
        }}, {{ responsive: true }});
        
        // 엔트로피 라인 차트
        const entropy = {json.dumps(entropy)};
        
        Plotly.newPlot('entropy-chart', [{{
            x: names,
            y: entropy,
            type: 'scatter',
            mode: 'lines+markers',
            name: '장르 엔트로피',
            line: {{ color: '#333', width: 3 }},
            marker: {{ size: 12, color: '#333', symbol: 'diamond' }}
        }}], {{
            paper_bgcolor: '#f9f9f9',
            plot_bgcolor: '#f9f9f9',
            xaxis: {{ title: '시나리오', color: '#333' }},
            yaxis: {{ title: '엔트로피', color: '#333' }},
            margin: {{ t: 30, b: 60, l: 60, r: 30 }}
        }}, {{ responsive: true }});
    </script>
</body>
</html>'''

    output_path.write_text(html_content, encoding='utf-8')
    print(f"✅ 가중치 민감도 시각화 저장: {output_path}")
    return True


def generate_hypothesis_test_html(output_path: Path):
    """영가설 검정 결과 HTML 시각화 생성"""
    
    # 데이터 로드
    json_path = VALIDATION_DIR / "hypothesis_test_summary.json"
    if not json_path.exists():
        print(f"❌ 데이터 파일 없음: {json_path}")
        return False
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    null_h = data['null_hypothesis']
    inverse = data['inverse_weighting']
    alt = data['alternative_centrality']
    
    verdict_class = 'verdict-pass' if null_h['verdict'] == 'REJECTED' else 'verdict-fail'
    
    html_content = f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>영가설 검정 결과 시각화 (흑백)</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>{BW_STYLE}</style>
</head>
<body>
    <div class="container">
        <h1>🔬 영가설 검정 결과 시각화</h1>
        <p class="subtitle">Sentence/Phrase K=4 클러스터 | {datetime.now().strftime("%Y-%m-%d")}</p>
        
        <h2>1. 영가설 테스트 (Null Hypothesis)</h2>
        <p><strong>H0:</strong> 사서 중심성은 우연의 결과</p>
        
        <div class="result-box {verdict_class}">
            <div class="metric">
                <span class="metric-label">원본 Canonicity:</span>
                <span class="metric-value">{null_h['observed_canonicity']:.2f}%</span>
            </div>
            <div class="metric">
                <span class="metric-label">랜덤 평균:</span>
                <span class="metric-value">{null_h['random_mean']:.2f}%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Effect Size (Cohen's d):</span>
                <span class="metric-value">{null_h['effect_size']:.3f}</span>
            </div>
            <div class="metric">
                <span class="metric-label">p-value:</span>
                <span class="metric-value">{null_h['p_value']:.6f}</span>
            </div>
            <div class="metric">
                <span class="metric-label">판정:</span>
                <span class="metric-value">{'✅ 영가설 기각 (사서 중심성은 실제 언어 패턴)' if null_h['verdict'] == 'REJECTED' else '❌ 영가설 채택'}</span>
            </div>
        </div>
        
        <div class="chart-container">
            <div id="effect-size-chart" class="chart"></div>
        </div>
        
        <h2>2. 반대가설 테스트 (Inverse Weighting)</h2>
        <div class="chart-container">
            <div id="inverse-chart" class="chart"></div>
        </div>
        <p><strong>해석:</strong> {inverse['interpretation']}</p>
        
        <h2>3. 대립가설 테스트 (Alternative Centrality)</h2>
        <div class="chart-container">
            <div id="alt-chart" class="chart"></div>
        </div>
        
        <div class="result-box {'verdict-pass' if alt['verdict'] == 'SASEO_DOMINANT' else 'verdict-fail'}">
            <div class="metric">
                <span class="metric-label">판정:</span>
                <span class="metric-value">{'✅ 사서가 유의하게 더 중심적' if alt['verdict'] == 'SASEO_DOMINANT' else '❌ 삼경이 더 중심적'}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Effect Size:</span>
                <span class="metric-value">{alt['effect_size']:.3f}</span>
            </div>
        </div>
        
        <h2>4. 종합 판정</h2>
        <div class="result-box verdict-pass">
            <div class="metric">
                <span class="metric-label">최종 판정:</span>
                <span class="metric-value">{data['final_verdict']}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Bias Level:</span>
                <span class="metric-value">{data['bias_level']}</span>
            </div>
        </div>
    </div>
    
    <script>
        // Effect Size 시각화
        Plotly.newPlot('effect-size-chart', [{{
            x: ['원본 Canonicity', '랜덤 평균'],
            y: [{null_h['observed_canonicity']}, {null_h['random_mean']}],
            type: 'bar',
            marker: {{ color: ['#333', '#999'] }},
            error_y: {{
                type: 'data',
                array: [0, {null_h['random_std']}],
                visible: true,
                color: '#000'
            }}
        }}], {{
            paper_bgcolor: '#f9f9f9',
            plot_bgcolor: '#f9f9f9',
            title: {{ text: 'Effect Size 시각화 (Cohen\\'s d = {null_h["effect_size"]:.2f})', font: {{ color: '#333' }} }},
            yaxis: {{ title: 'Canonicity (%)', color: '#333' }},
            margin: {{ t: 50, b: 60, l: 60, r: 30 }}
        }}, {{ responsive: true }});
        
        // 반대가설 테스트
        Plotly.newPlot('inverse-chart', [{{
            x: ['Strong (5.0x)', 'Uniform (1.0x)', 'Inverse (0.2x)'],
            y: [{inverse['strong']}, {inverse['uniform']}, {inverse['inverse']}],
            type: 'bar',
            marker: {{ color: '#333' }}
        }}], {{
            paper_bgcolor: '#f9f9f9',
            plot_bgcolor: '#f9f9f9',
            title: {{ text: '가중치별 가중 Canonicity', font: {{ color: '#333' }} }},
            yaxis: {{ title: '가중 비율 (%)', color: '#333' }},
            margin: {{ t: 50, b: 60, l: 60, r: 30 }}
        }}, {{ responsive: true }});
        
        // 대립가설 테스트
        Plotly.newPlot('alt-chart', [{{
            x: ['사서 (四書)', '삼경 (三經)', '기타'],
            y: [{alt['saseo_ratio']}, {alt['samgyeong_ratio']}, {alt['other_ratio']}],
            type: 'bar',
            marker: {{ color: ['#000', '#666', '#bbb'] }}
        }}], {{
            paper_bgcolor: '#f9f9f9',
            plot_bgcolor: '#f9f9f9',
            title: {{ text: '텍스트 집단별 비율', font: {{ color: '#333' }} }},
            yaxis: {{ title: '비율 (%)', color: '#333' }},
            margin: {{ t: 50, b: 60, l: 60, r: 30 }}
        }}, {{ responsive: true }});
    </script>
</body>
</html>'''

    output_path.write_text(html_content, encoding='utf-8')
    print(f"✅ 영가설 검정 시각화 저장: {output_path}")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("검증 분석 HTML 시각화 생성")
    print("=" * 60)
    
    # 가중치 민감도 시각화
    ws_html = VALIDATION_DIR / "weight_sensitivity" / "weight_sensitivity_visualization_bw.html"
    generate_weight_sensitivity_html(ws_html)
    
    # 영가설 검정 시각화
    ht_html = VALIDATION_DIR / "hypothesis_test_visualization_bw.html"
    generate_hypothesis_test_html(ht_html)
    
    print("\n✅ 모든 시각화 생성 완료!")
