"""
원문 문자 손실 상세 분석 스크립트
"""

import pandas as pd
import difflib

def analyze_text_loss():
    """원문 문자 손실 상세 분석"""
    print("🔍 원문 문자 손실 상세 분석")
    print("="*60)
    
    # 파일 읽기
    input_df = pd.read_excel("pa03.xlsx")
    output_df = pd.read_excel("output_pa_bge.xlsx")
    
    total_loss = 0
    loss_details = []
    
    for input_idx, input_row in input_df.iterrows():
        input_tgt = str(input_row.get('원문', ''))
        para_id = input_idx + 1
        output_rows = output_df[output_df['문단식별자'] == para_id]
        
        if len(output_rows) == 0:
            continue

        # 원문 결합
        output_tgt_combined = ' '.join(output_rows['원문'].astype(str))
        
        # 문자수 비교
        input_len = len(input_tgt)
        output_len = len(output_tgt_combined)
        loss = input_len - output_len
        
        if loss != 0:
            loss_details.append({
                'paragraph': para_id,
                'input_len': input_len,
                'output_len': output_len,
                'loss': loss,
                'input_text': input_tgt,
                'output_text': output_tgt_combined,
                'split_count': len(output_rows)
            })
            total_loss += loss
    
    print(f"📊 손실 발생 문단: {len(loss_details)}개")
    print(f"📊 총 손실 문자: {total_loss}개")
    
    # 상위 손실 문단들 분석
    loss_details.sort(key=lambda x: abs(x['loss']), reverse=True)
    
    print(f"\n🔍 손실이 큰 문단들:")
    for i, detail in enumerate(loss_details[:10]):  # 상위 10개
        print(f"\n{i+1}. 문단 {detail['paragraph']} (손실: {detail['loss']}문자, 분할: {detail['split_count']}개)")
        print(f"   입력 길이: {detail['input_len']}")
        print(f"   출력 길이: {detail['output_len']}")
        
        # 차이점 시각화
        input_text = detail['input_text']
        output_text = detail['output_text']
        
        if len(input_text) < 200:  # 짧은 텍스트만 전체 표시
            print(f"   입력: '{input_text}'")
            print(f"   출력: '{output_text}'")
            
            # diff 분석
            diff = list(difflib.unified_diff(
                input_text.splitlines(keepends=True),
                output_text.splitlines(keepends=True),
                fromfile='입력',
                tofile='출력',
                lineterm=''
            ))
            if diff:
                print(f"   차이점:")
                for line in diff[2:]:  # 헤더 제외
                    print(f"     {line.rstrip()}")
        else:
            print(f"   입력 시작: '{input_text[:100]}...'")
            print(f"   출력 시작: '{output_text[:100]}...'")
    
    # 손실 패턴 분석
    print(f"\n📈 손실 패턴 분석:")
    
    # 양수 손실 (실제 문자 손실)
    positive_losses = [d for d in loss_details if d['loss'] > 0]
    negative_losses = [d for d in loss_details if d['loss'] < 0]
    
    print(f"실제 손실 (양수): {len(positive_losses)}개 문단, 총 {sum(d['loss'] for d in positive_losses)}문자")
    print(f"문자 증가 (음수): {len(negative_losses)}개 문단, 총 {sum(d['loss'] for d in negative_losses)}문자")
    
    # 분할 횟수별 분석
    split_analysis = {}
    for detail in loss_details:
        split_count = detail['split_count']
        if split_count not in split_analysis:
            split_analysis[split_count] = {'count': 0, 'total_loss': 0}
        split_analysis[split_count]['count'] += 1
        split_analysis[split_count]['total_loss'] += detail['loss']
    
    print(f"\n📊 분할 횟수별 손실:")
    for split_count in sorted(split_analysis.keys()):
        data = split_analysis[split_count]
        avg_loss = data['total_loss'] / data['count']
        print(f"  {split_count}개 분할: {data['count']}개 문단, 평균 손실 {avg_loss:.1f}문자")

if __name__ == "__main__":
    analyze_text_loss()
