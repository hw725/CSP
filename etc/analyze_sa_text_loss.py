"""
SA 번역문 문자 손실 상세 분석 스크립트
"""

import pandas as pd
import difflib

def analyze_sa_text_loss():
    """SA 번역문 문자 손실 상세 분석"""
    print("🔍 SA 번역문 문자 손실 상세 분석")
    print("="*60)
    
    # 파일 읽기
    input_df = pd.read_excel("sa/input01.xlsx")
    output_df = pd.read_excel("sa/output01.xlsx")
    
    total_src_loss = 0
    total_tgt_loss = 0
    loss_details = []
    
    for input_idx, input_row in input_df.iterrows():
        input_src = str(input_row.get('원문', ''))
        input_tgt = str(input_row.get('번역문', ''))
        sent_id = input_idx + 1
        output_rows = output_df[output_df['문장식별자'] == sent_id]
        
        if len(output_rows) == 0:
            continue
            
        # 원문, 번역문 결합
        output_src_combined = ' '.join(output_rows['원문'].astype(str))
        output_tgt_combined = ' '.join(output_rows['번역문'].astype(str))
        
        # 문자수 비교
        input_src_len = len(input_src)
        output_src_len = len(output_src_combined)
        src_loss = input_src_len - output_src_len
        
        input_tgt_len = len(input_tgt)
        output_tgt_len = len(output_tgt_combined)
        tgt_loss = input_tgt_len - output_tgt_len
        
        if src_loss != 0 or tgt_loss != 0:
            loss_details.append({
                'sentence': sent_id,
                'input_src_len': input_src_len,
                'output_src_len': output_src_len,
                'src_loss': src_loss,
                'input_tgt_len': input_tgt_len,
                'output_tgt_len': output_tgt_len,
                'tgt_loss': tgt_loss,
                'input_src': input_src,
                'output_src': output_src_combined,
                'input_tgt': input_tgt,
                'output_tgt': output_tgt_combined,
                'split_count': len(output_rows)
            })
            total_src_loss += src_loss
            total_tgt_loss += tgt_loss
    
    print(f"📊 손실 발생 문장: {len(loss_details)}개")
    print(f"📊 총 원문 손실 문자: {total_src_loss}개")
    print(f"📊 총 번역문 손실 문자: {total_tgt_loss}개")
    
    # 상위 손실 문장들 분석
    loss_details.sort(key=lambda x: abs(x['src_loss']) + abs(x['tgt_loss']), reverse=True)
    
    print(f"\n🔍 손실이 큰 문장들:")
    for i, detail in enumerate(loss_details[:10]):  # 상위 10개
        print(f"\n{i+1}. 문장 {detail['sentence']} (원문 손실: {detail['src_loss']}, 번역문 손실: {detail['tgt_loss']}, 분할: {detail['split_count']}개)")
        print(f"   원문: {detail['input_src_len']} → {detail['output_src_len']}")
        print(f"   번역문: {detail['input_tgt_len']} → {detail['output_tgt_len']}")
        
        # 짧은 텍스트만 전체 표시
        if len(detail['input_src']) < 100:
            print(f"   입력 원문: '{detail['input_src']}'")
            print(f"   출력 원문: '{detail['output_src']}'")
        else:
            print(f"   입력 원문: '{detail['input_src'][:50]}...'")
            print(f"   출력 원문: '{detail['output_src'][:50]}...'")
            
        if len(detail['input_tgt']) < 100:
            print(f"   입력 번역문: '{detail['input_tgt']}'")
            print(f"   출력 번역문: '{detail['output_tgt']}'")
        else:
            print(f"   입력 번역문: '{detail['input_tgt'][:50]}...'")
            print(f"   출력 번역문: '{detail['output_tgt'][:50]}...'")
    
    # 손실 패턴 분석
    print(f"\n📈 손실 패턴 분석:")
    
    # 원문 손실
    src_positive_losses = [d for d in loss_details if d['src_loss'] > 0]
    src_negative_losses = [d for d in loss_details if d['src_loss'] < 0]
    
    print(f"원문 실제 손실: {len(src_positive_losses)}개 문장, 총 {sum(d['src_loss'] for d in src_positive_losses)}문자")
    print(f"원문 문자 증가: {len(src_negative_losses)}개 문장, 총 {sum(d['src_loss'] for d in src_negative_losses)}문자")
    
    # 번역문 손실
    tgt_positive_losses = [d for d in loss_details if d['tgt_loss'] > 0]
    tgt_negative_losses = [d for d in loss_details if d['tgt_loss'] < 0]
    
    print(f"번역문 실제 손실: {len(tgt_positive_losses)}개 문장, 총 {sum(d['tgt_loss'] for d in tgt_positive_losses)}문자")
    print(f"번역문 문자 증가: {len(tgt_negative_losses)}개 문장, 총 {sum(d['tgt_loss'] for d in tgt_negative_losses)}문자")
    
    # 분할 횟수별 분석
    split_analysis = {}
    for detail in loss_details:
        split_count = detail['split_count']
        if split_count not in split_analysis:
            split_analysis[split_count] = {'count': 0, 'total_src_loss': 0, 'total_tgt_loss': 0}
        split_analysis[split_count]['count'] += 1
        split_analysis[split_count]['total_src_loss'] += detail['src_loss']
        split_analysis[split_count]['total_tgt_loss'] += detail['tgt_loss']
    
    print(f"\n📊 분할 횟수별 손실:")
    for split_count in sorted(split_analysis.keys()):
        data = split_analysis[split_count]
        avg_src_loss = data['total_src_loss'] / data['count']
        avg_tgt_loss = data['total_tgt_loss'] / data['count']
        print(f"  {split_count}개 분할: {data['count']}개 문장, 평균 원문 손실 {avg_src_loss:.1f}문자, 평균 번역문 손실 {avg_tgt_loss:.1f}문자")

if __name__ == "__main__":
    analyze_sa_text_loss()
