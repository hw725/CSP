"""간소화된 병렬 처리 - 불필요한 코드 제거"""

import pandas as pd
from tqdm import tqdm
import logging
from pathlib import Path
import multiprocessing as mp
from typing import Dict, Any, List
import sys
import os

# 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

logger = logging.getLogger(__name__)

# 전역 변수 (각 프로세스에서 초기화)
worker_embed_func = None
worker_modules = {}

def init_worker(embedder_name='bge'):
    """워커 프로세스 초기화 - 간소화"""
    global worker_embed_func, worker_modules
    
    try:
        # 임베더 초기화
        if embedder_name == 'openai':
            from common.embedders.openai import compute_embeddings_with_cache
            worker_embed_func = compute_embeddings_with_cache
        else:  # bge 기본값
            from common.embedders.bge import get_embed_func
            worker_embed_func = get_embed_func()
        
        # 필요한 모듈들 import
        from sa_tokenizers.jieba_mecab import (
            split_src_meaning_units, 
            split_tgt_by_src_units_semantic
        )
        from punctuation import mask_brackets, restore_brackets
        
        worker_modules = {
            'split_src_meaning_units': split_src_meaning_units,
            'split_tgt_by_src_units_semantic': split_tgt_by_src_units_semantic,
            'mask_brackets': mask_brackets,
            'restore_brackets': restore_brackets
        }
        
    except Exception as e:
        print(f"워커 초기화 실패: {e}")
        worker_embed_func = None
        worker_modules = {}

def process_batch_sentences(sentence_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """배치 단위 문장 처리 - 간소화"""
    global worker_embed_func, worker_modules
    
    if worker_embed_func is None or not worker_modules:
        return []
    
    results = []
    
    for sentence_data in sentence_batch:
        try:
            sentence_id = sentence_data['sentence_id']
            src_text = sentence_data['src_text']
            tgt_text = sentence_data['tgt_text']
            
            # 모듈들 가져오기
            mask_brackets = worker_modules['mask_brackets']
            restore_brackets = worker_modules['restore_brackets']
            split_src_meaning_units = worker_modules['split_src_meaning_units']
            split_tgt_by_src_units_semantic = worker_modules['split_tgt_by_src_units_semantic']
            
            # 처리 파이프라인
            masked_src, src_masks = mask_brackets(src_text, 'source')
            masked_tgt, tgt_masks = mask_brackets(tgt_text, 'target')
            
            src_units = split_src_meaning_units(masked_src)
            tgt_units = split_tgt_by_src_units_semantic(
                src_units, masked_tgt, worker_embed_func, min_tokens=1
            )
            
            # 결과 생성
            for phrase_idx, (src_unit, tgt_unit) in enumerate(zip(src_units, tgt_units), 1):
                restored_src = restore_brackets(src_unit, src_masks)
                restored_tgt = restore_brackets(tgt_unit, tgt_masks)
                
                results.append({
                    '문장식별자': sentence_id,
                    '구식별자': phrase_idx,
                    '원문': restored_src,
                    '번역문': restored_tgt
                })
            
        except Exception as e:
            logger.error(f"문장 {sentence_data.get('sentence_id', '?')} 처리 오류: {e}")
            continue
    
    return results

def process_file(
    input_path: str, 
    output_path: str, 
    parallel: bool = False, 
    workers: int = 4, 
    embedder_name: str = 'bge'
) -> pd.DataFrame:
    """파일 처리 함수 - 간소화"""
    
    # 데이터 로드
    print(f"📂 파일 로드 중: {input_path}")
    
    try:
        df = pd.read_excel(input_path) if input_path.endswith('.xlsx') else pd.read_csv(input_path)
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        raise
    
    print(f"✅ 로드 완료: {len(df)}개 행")
    
    # 필수 컬럼 확인
    required_columns = ['문장식별자', '원문', '번역문']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        error_msg = f"필수 컬럼 누락: {missing_columns}"
        print(f"❌ {error_msg}")
        raise ValueError(error_msg)
    
    # 데이터 준비
    sentence_data_list = []
    for idx, row in df.iterrows():
        src_text = str(row['원문']).strip()
        tgt_text = str(row['번역문']).strip()
        
        if src_text and tgt_text:
            sentence_data_list.append({
                'sentence_id': row['문장식별자'],
                'src_text': src_text,
                'tgt_text': tgt_text
            })
    
    total_sentences = len(sentence_data_list)
    print(f"📊 처리할 문장: {total_sentences}개")
    
    if total_sentences == 0:
        raise ValueError("처리할 문장이 없습니다")
    
    # 처리 실행
    results = []
    
    if parallel and total_sentences > workers:
        print(f"🔄 병렬 처리 시작 ({workers}개 프로세스)")
        
        # 문장들을 배치로 분할
        batch_size_per_worker = max(1, total_sentences // workers)
        sentence_batches = []
        
        for i in range(0, total_sentences, batch_size_per_worker):
            batch = sentence_data_list[i:i + batch_size_per_worker]
            sentence_batches.append(batch)
        
        print(f"📋 배치 분할: {len(sentence_batches)}개 배치")
        print(f"🔧 {embedder_name.upper()} 모델 로딩 중...")
        
        # 프로세스 풀로 배치 처리
        with mp.Pool(processes=workers, initializer=init_worker, initargs=(embedder_name,)) as pool:
            try:
                print(f"✅ 모델 로딩 완료! 문장 처리를 시작합니다...")
                
                # 결과 수집
                with tqdm(total=total_sentences, desc="문장 처리", unit="문장") as pbar:
                    for i, batch in enumerate(sentence_batches):
                        try:
                            batch_results = pool.apply(process_batch_sentences, (batch,))
                            results.extend(batch_results)
                            
                            pbar.update(len(batch))
                            pbar.set_postfix({"구": len(results), "배치": f"{i+1}/{len(sentence_batches)}"})
                            
                        except Exception as e:
                            pbar.update(len(batch))
                            logger.error(f"배치 {i+1} 처리 오류: {e}")
                        
            except KeyboardInterrupt:
                print("사용자 중단")
                pool.terminate()
                pool.join()
                raise
                
    else:
        print(f"🔄 순차 처리 시작")
        print(f"🔧 {embedder_name.upper()} 모델 로딩 중...")
        
        # 순차 처리
        if embedder_name == 'openai':
            from common.embedders.openai import compute_embeddings_with_cache
            embed_func = compute_embeddings_with_cache
        else:  # bge 기본값
            from common.embedders.bge import get_embed_func
            embed_func = get_embed_func()
        
        print(f"✅ 모델 로딩 완료! 문장 처리를 시작합니다...")
        
        from sa_tokenizers.jieba_mecab import (
            split_src_meaning_units, 
            split_tgt_by_src_units_semantic
        )
        from punctuation import mask_brackets, restore_brackets
        
        for sentence_data in tqdm(sentence_data_list, desc="문장 처리"):
            try:
                sentence_id = sentence_data['sentence_id']
                src_text = sentence_data['src_text']
                tgt_text = sentence_data['tgt_text']
                
                # 처리 파이프라인
                masked_src, src_masks = mask_brackets(src_text, 'source')
                masked_tgt, tgt_masks = mask_brackets(tgt_text, 'target')
                
                src_units = split_src_meaning_units(masked_src)
                tgt_units = split_tgt_by_src_units_semantic(
                    src_units, masked_tgt, embed_func, min_tokens=1
                )
                
                # 결과 생성
                for phrase_idx, (src_unit, tgt_unit) in enumerate(zip(src_units, tgt_units), 1):
                    restored_src = restore_brackets(src_unit, src_masks)
                    restored_tgt = restore_brackets(tgt_unit, tgt_masks)
                    
                    results.append({
                        '문장식별자': sentence_id,
                        '구식별자': phrase_idx,
                        '원문': restored_src,
                        '번역문': restored_tgt
                    })
                
            except Exception as e:
                logger.error(f"문장 {sentence_data.get('sentence_id', '?')} 순차 처리 오류: {e}")
                continue
    
    # 결과 저장
    if not results:
        raise ValueError("처리된 결과가 없습니다")
    
    print(f"💾 결과 저장 중: {output_path}")
    result_df = pd.DataFrame(results)
    
    try:
        result_df.to_excel(output_path, index=False)
        
        print(f"🎉 처리 완료!")
        print(f"📊 결과: {total_sentences}개 문장 → {len(results)}개 구")
        print(f"📁 출력: {output_path}")
        
        return result_df
        
    except Exception as e:
        print(f"❌ 결과 저장 실패: {e}")
        raise