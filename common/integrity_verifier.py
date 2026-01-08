"""
전역 무결성 검증 모듈 (SA/PA 공통)
- 입력(GT) 전체 텍스트 vs 출력(Pred) 전체 텍스트 비교
- 손실/추가된 문자 상세 분석
- 결과를 DataFrame으로 반환 (ExcelWriter에 시트 추가 가능)
"""

import pandas as pd
from typing import Dict, List, Tuple, Any
from difflib import SequenceMatcher


def analyze_text_integrity(
    gt_text: str,
    pred_text: str,
    text_name: str = "텍스트"
) -> Dict[str, Any]:
    """
    텍스트 무결성 분석 (손실/추가 문자 상세)
    
    Args:
        gt_text: 정답(입력) 전체 텍스트
        pred_text: 예측(출력) 전체 텍스트
        text_name: 텍스트 이름 (예: "원문", "번역문")
    
    Returns:
        {
            'text_name': str,
            'gt_len': int,
            'pred_len': int,
            'delta': int,  # gt_len - pred_len (양수=손실, 음수=추가)
            'losses': List[Dict],  # 손실된 문자 리스트
            'additions': List[Dict],  # 추가된 문자 리스트
            'only_space_loss': bool,  # 손실이 공백만인지
            'only_space_add': bool,  # 추가가 공백만인지
            'is_intact': bool,  # 완전 일치
        }
    """
    sm = SequenceMatcher(None, gt_text, pred_text)
    
    losses = []
    additions = []
    
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'delete':
            lost_text = gt_text[i1:i2]
            is_space = all(c.isspace() for c in lost_text)
            char_type = '공백' if is_space else '문자'
            losses.append({
                'position': i1,
                'lost_text': lost_text,
                'length': len(lost_text),
                'type': char_type,
                'context_before': gt_text[max(0, i1-30):i1],
                'context_after': gt_text[i2:i2+30]
            })
        elif tag == 'insert':
            added_text = pred_text[j1:j2]
            is_space = all(c.isspace() for c in added_text)
            char_type = '공백' if is_space else '문자'
            additions.append({
                'position': j1,
                'added_text': added_text,
                'length': len(added_text),
                'type': char_type,
                'context_before': pred_text[max(0, j1-30):j1],
                'context_after': pred_text[j2:j2+30]
            })
    
    only_space_loss = all(loss['type'] == '공백' for loss in losses) if losses else True
    only_space_add = all(add['type'] == '공백' for add in additions) if additions else True
    
    return {
        'text_name': text_name,
        'gt_len': len(gt_text),
        'pred_len': len(pred_text),
        'delta': len(gt_text) - len(pred_text),
        'losses': losses,
        'additions': additions,
        'only_space_loss': only_space_loss,
        'only_space_add': only_space_add,
        'is_intact': (len(gt_text) == len(pred_text)) and (gt_text == pred_text),
    }


def verify_global_integrity(
    input_df: pd.DataFrame,
    result_df: pd.DataFrame,
    source_col: str = '원문',
    target_col: str = '번역문',
    verbose: bool = True
) -> Tuple[bool, pd.DataFrame, Dict[str, Any]]:
    """
    전역 무결성 검증 및 손실 분석
    
    Args:
        input_df: 입력(GT) DataFrame
        result_df: 결과(Pred) DataFrame
        source_col: 원문 컬럼명
        target_col: 번역문 컬럼명
        verbose: 상세 출력 여부
    
    Returns:
        (passed: bool, losses_df: pd.DataFrame, analysis: Dict)
        - passed: 공백만 손실되었는지 여부
        - losses_df: 손실/추가 상세 정보 DataFrame
        - analysis: 분석 결과 딕셔너리
    """
    # 1. 입력 파일 전체 텍스트 복원
    gt_sources = "".join(str(row.get(source_col, '')) for _, row in input_df.iterrows())
    gt_targets = "".join(str(row.get(target_col, '')) for _, row in input_df.iterrows())
    
    # 2. 출력 파일 전체 텍스트 복원
    pred_sources = "".join(str(row.get(source_col, '')) for _, row in result_df.iterrows())
    pred_targets = "".join(str(row.get(target_col, '')) for _, row in result_df.iterrows())
    
    # 3. 분석
    src_analysis = analyze_text_integrity(gt_sources, pred_sources, "원문")
    tgt_analysis = analyze_text_integrity(gt_targets, pred_targets, "번역문")
    
    # 4. 손실 상세 정보 DataFrame 구성
    loss_records = []
    
    # 원문 손실
    for loss in src_analysis['losses']:
        loss_records.append({
            'Text Type': '원문',
            'Operation': '손실',
            'Position': loss['position'],
            'Length': loss['length'],
            'Type': loss['type'],
            'Lost/Added': loss['lost_text'],
            'Context Before': loss['context_before'],
            'Context After': loss['context_after']
        })
    
    # 번역문 손실
    for loss in tgt_analysis['losses']:
        loss_records.append({
            'Text Type': '번역문',
            'Operation': '손실',
            'Position': loss['position'],
            'Length': loss['length'],
            'Type': loss['type'],
            'Lost/Added': loss['lost_text'],
            'Context Before': loss['context_before'],
            'Context After': loss['context_after']
        })
    
    # 원문 추가
    for add in src_analysis['additions']:
        loss_records.append({
            'Text Type': '원문',
            'Operation': '추가',
            'Position': add['position'],
            'Length': add['length'],
            'Type': add['type'],
            'Lost/Added': add['added_text'],
            'Context Before': add['context_before'],
            'Context After': add['context_after']
        })
    
    # 번역문 추가
    for add in tgt_analysis['additions']:
        loss_records.append({
            'Text Type': '번역문',
            'Operation': '추가',
            'Position': add['position'],
            'Length': add['length'],
            'Type': add['type'],
            'Lost/Added': add['added_text'],
            'Context Before': add['context_before'],
            'Context After': add['context_after']
        })
    
    losses_df = pd.DataFrame(loss_records) if loss_records else pd.DataFrame()
    
    # 5. 종합 판정
    passed = src_analysis['only_space_loss'] and tgt_analysis['only_space_loss']
    
    # 6. 상세 출력
    if verbose:
        print("\n" + "="*80)
        print("🔍 전역 무결성 검증")
        print("="*80)
        
        print(f"\n📝 원문:")
        print(f"  입력(GT): {src_analysis['gt_len']:,}자")
        print(f"  출력(Pred): {src_analysis['pred_len']:,}자")
        print(f"  Δ: {src_analysis['delta']:+,}자")
        
        if src_analysis['delta'] != 0:
            if src_analysis['only_space_loss'] and src_analysis['delta'] > 0:
                print(f"  ✅ 손실: 공백만 ({len(src_analysis['losses'])}건)")
            elif src_analysis['only_space_add'] and src_analysis['delta'] < 0:
                print(f"  ✅ 추가: 공백만 ({len(src_analysis['additions'])}건)")
            else:
                print(f"  ❌ 손실/추가: 공백 외 문자 포함!")
                if src_analysis['losses']:
                    for loss in src_analysis['losses'][:3]:
                        print(f"    손실 [{loss['type']}] '{loss['lost_text']}' (위치: {loss['position']})")
                    if len(src_analysis['losses']) > 3:
                        print(f"    ... 및 {len(src_analysis['losses']) - 3}건 추가")
                if src_analysis['additions']:
                    for add in src_analysis['additions'][:3]:
                        print(f"    추가 [{add['type']}] '{add['added_text']}' (위치: {add['position']})")
                    if len(src_analysis['additions']) > 3:
                        print(f"    ... 및 {len(src_analysis['additions']) - 3}건 추가")
        
        print(f"\n📖 번역문:")
        print(f"  입력(GT): {tgt_analysis['gt_len']:,}자")
        print(f"  출력(Pred): {tgt_analysis['pred_len']:,}자")
        print(f"  Δ: {tgt_analysis['delta']:+,}자")
        
        if tgt_analysis['delta'] != 0:
            if tgt_analysis['only_space_loss'] and tgt_analysis['delta'] > 0:
                print(f"  ✅ 손실: 공백만 ({len(tgt_analysis['losses'])}건)")
            elif tgt_analysis['only_space_add'] and tgt_analysis['delta'] < 0:
                print(f"  ✅ 추가: 공백만 ({len(tgt_analysis['additions'])}건)")
            else:
                print(f"  ❌ 손실/추가: 공백 외 문자 포함!")
                if tgt_analysis['losses']:
                    for loss in tgt_analysis['losses'][:3]:
                        print(f"    손실 [{loss['type']}] '{loss['lost_text']}' (위치: {loss['position']})")
                    if len(tgt_analysis['losses']) > 3:
                        print(f"    ... 및 {len(tgt_analysis['losses']) - 3}건 추가")
                if tgt_analysis['additions']:
                    for add in tgt_analysis['additions'][:3]:
                        print(f"    추가 [{add['type']}] '{add['added_text']}' (위치: {add['position']})")
                    if len(tgt_analysis['additions']) > 3:
                        print(f"    ... 및 {len(tgt_analysis['additions']) - 3}건 추가")
        
        print("\n" + "="*80)
        if passed:
            print("✅ 무결성 검증 통과: 공백만 손실/추가됨")
        else:
            print("❌ 무결성 경고: 공백 외 문자도 손실됨 (코드 보강 필요)")
        print("="*80 + "\n")
    
    analysis = {
        'source': src_analysis,
        'target': tgt_analysis,
        'passed': passed,
        'total_losses': len(src_analysis['losses']) + len(tgt_analysis['losses']),
        'total_additions': len(src_analysis['additions']) + len(tgt_analysis['additions']),
    }
    
    return passed, losses_df, analysis
