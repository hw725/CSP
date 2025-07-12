"""개선된 병렬 처리 - 작업 단위 분산"""
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from typing import Dict, Any, List
import math
import sys, os

CSP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if CSP_ROOT not in sys.path:
    sys.path.insert(0, CSP_ROOT)

logger = logging.getLogger(__name__)

# 전역 변수 선언을 함수 내부 사용보다 앞에 배치
worker_embed_func = None
worker_modules = None

def init_worker(embed_func, modules):
    """워커 프로세스 초기화"""
    global worker_embed_func, worker_modules
    worker_embed_func = embed_func
    worker_modules = modules

# 전역 변수 (각 프로세스에서 초기화)
worker_embed_func = None
worker_modules = {}
verbose_mode = False  # 기본적으로 비활성화

def set_verbose_mode(verbose=False):
    """전역 verbose 모드 설정"""
    global verbose_mode
    verbose_mode = verbose

def init_worker(device_id=None, embedder_name='bge', openai_model=None, openai_api_key=None):
    """워커 프로세스 초기화 - 한 번만 실행 (device_id로 GPU 분배)"""
    global worker_embed_func, worker_modules
    try:
        if verbose_mode:
            print(f"워커 {mp.current_process().pid}: 초기화 시작 (device_id={device_id})")
        
        # 임베더 로드 시도
        from core.embedder_factory import get_embedder
        worker_embed_func = get_embedder(
            embedder_name, 
            device_id=device_id, 
            openai_model=openai_model, 
            openai_api_key=openai_api_key
        )
        
        # 임베더 로드 실패 시에도 다른 모듈은 로드
        if worker_embed_func is None:
            print(f"워커 {mp.current_process().pid}: 임베더 로드 실패, 기본 처리로 진행")
        
        # 필요한 모듈들 로드
        from sa_tokenizers.jieba_mecab import split_src_meaning_units, split_tgt_by_src_units_semantic
        from punctuation import mask_brackets, restore_brackets
        
        worker_modules = {
            'split_src_meaning_units': split_src_meaning_units,
            'split_tgt_by_src_units_semantic': split_tgt_by_src_units_semantic,
            'mask_brackets': mask_brackets,
            'restore_brackets': restore_brackets
        }
        
        if verbose_mode:
            embed_status = "성공" if worker_embed_func else "실패(기본 처리)"
            print(f"워커 {mp.current_process().pid}: 초기화 완료 (임베더: {embed_status})")
            
    except Exception as e:
        print(f"워커 {mp.current_process().pid}: 초기화 실패: {e}")
        # 초기화가 실패해도 기본 모듈은 설정
        worker_modules = {}
        worker_embed_func = None
        raise

def process_batch_sentences(sentence_batch, device_id=None):
    """배치 단위 문장 처리 - 문장당 30초 이상 배정"""
    global worker_embed_func, worker_modules
    
    import gc
    import time
    
    results = []
    batch_size = len(sentence_batch)
    start_time = time.time()
    
    try:
        if verbose_mode:
            print(f"🔄 워커 {mp.current_process().pid}: {batch_size}개 문장 배치 시작 (문장당 30초+ 배정)")
        
        for idx, sentence_data in enumerate(sentence_batch):
            try:
                # 5문장 단위에서 더 자세한 진행률 표시
                if batch_size <= 5:
                    # 모든 문장 시작시 표시
                    print(f"🔄 워커 {mp.current_process().pid}: {idx+1}/{batch_size} 처리 중... ({((idx+1)/batch_size)*100:.0f}%) - 문장ID: {sentence_data.get('sentence_id', '?')}")
                else:
                    if (idx + 1) % 2 == 0:
                        print(f"🔄 워커 {mp.current_process().pid}: {idx+1}/{batch_size} 처리 중...")

                sentence_id = sentence_data['sentence_id']
                src_text = sentence_data['src_text']
                tgt_text = sentence_data['tgt_text']
                
                # 처리 파이프라인 (기존과 동일)
                if worker_modules:
                    mask_brackets = worker_modules['mask_brackets']
                    restore_brackets = worker_modules['restore_brackets']
                    split_src_meaning_units = worker_modules['split_src_meaning_units']
                    split_tgt_by_src_units_semantic = worker_modules['split_tgt_by_src_units_semantic']
                else:
                    from sa_tokenizers.jieba_mecab import split_src_meaning_units
                    from punctuation import mask_brackets, restore_brackets
                    split_tgt_by_src_units_semantic = None
                
                masked_src, src_masks = mask_brackets(src_text, 'source')
                masked_tgt, tgt_masks = mask_brackets(tgt_text, 'target')
                
                src_units = split_src_meaning_units(masked_src)
                
                if worker_embed_func and split_tgt_by_src_units_semantic:
                    tgt_units = split_tgt_by_src_units_semantic(
                        src_units, masked_tgt, worker_embed_func, min_tokens=1
                    )
                else:
                    # 기본 처리
                    tgt_words = masked_tgt.split()
                    words_per_unit = max(1, len(tgt_words) // len(src_units)) if src_units else 1
                    
                    tgt_units = []
                    for j in range(len(src_units)):
                        start_idx = j * words_per_unit
                        end_idx = min((j + 1) * words_per_unit, len(tgt_words))
                        unit = ' '.join(tgt_words[start_idx:end_idx])
                        if unit.strip():
                            tgt_units.append(unit)
                
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
                
                # 메모리 정리
                if (idx + 1) % 2 == 0:
                    gc.collect()
                
            except Exception as e:
                logger.error(f"문장 {sentence_data.get('sentence_id', '?')} 처리 오류: {e}")
                continue
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 처리 시간 기준을 문장당 30초로 조정
        expected_time = batch_size * 30  # 문장당 30초 예상
        if processing_time > expected_time:
            print(f"⚠️ 워커 {mp.current_process().pid}: 배치 처리 시간 주의 ({processing_time:.1f}초 > {expected_time}초 예상, 문장당 {processing_time/batch_size:.1f}초)")
        
        if verbose_mode or batch_size <= 5:
            print(f"✅ 워커 {mp.current_process().pid}: 배치 완료 → {len(results)}개 구 ({processing_time:.2f}초, 문장당 {processing_time/batch_size:.1f}초)")
        
    except Exception as e:
        print(f"❌ 워커 {mp.current_process().pid}: 배치 처리 실패: {e}")
        logger.error(f"배치 처리 실패: {e}")
    
    finally:
        gc.collect()
    
    return results

def process_file(input_path: str, output_path: str, parallel: bool = False, workers: int = 4, 
                batch_size: int = 20, device_ids=None, verbose: bool = False, 
                embedder_name: str = 'bge', openai_model: str = None, openai_api_key: str = None):
    """파일 처리 - 임베더 실패 시 중단"""
    
    # BGE 사용 시 강제 설정
    if embedder_name == 'bge':
        parallel = False
        print("🔧 BGE 임베더 사용으로 순차 처리 모드로 전환 (메모리 최적화)")
        
        # 메모리 정리
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # verbose 모드 설정
    set_verbose_mode(verbose)
    
    # 데이터 로드
    logger.info(f"파일 로드: {input_path}")
    print(f"📂 파일 로드 중: {input_path}")
    
    try:
        if input_path.endswith('.xlsx'):
            df = pd.read_excel(input_path)
        else:
            df = pd.read_csv(input_path)
    except Exception as e:
        logger.error(f"파일 로드 실패: {e}")
        raise
    
    logger.info(f"로드 완료: {len(df)}개 행")
    print(f"✅ 로드 완료: {len(df)}개 행")
    
    # 필수 컬럼 확인
    required_columns = ['문장식별자', '원문', '번역문']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"필수 컬럼 누락: {missing_columns}")
    
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
    logger.info(f"처리할 문장: {total_sentences}개")
    print(f"📊 처리할 문장: {total_sentences}개")
    
    if total_sentences == 0:
        raise ValueError("처리할 문장이 없습니다")
    
    # BGE 사용 시 메모리 최적화 안내
    if embedder_name == 'bge':
        print(f"🧠 BGE 메모리 최적화 모드:")
        print(f"   - 배치 크기: 3문장씩")
        print(f"   - 순차 처리 (병렬 처리 비활성화)")
        print(f"   - 메모리 절약 설정 적용")
        print(f"   - 예상 처리 시간: {total_sentences * 2:.0f}초 (문장당 2초)")
        print()
    
    # 전체 진행률 표시 초기화
    progress_bar = tqdm(
        total=total_sentences, 
        desc="🔄 문장 처리 (BGE 최적화)" if embedder_name == 'bge' else "🔄 문장 처리", 
        ncols=120,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    results = []
    
    try:
        # BGE는 항상 순차 처리
        print("🔄 순차 처리 시작 (BGE 메모리 최적화)")
        logger.info("순차 처리 시작 (BGE 최적화)")
        
        # 임베더 로드 시도
        embed_func = None
        try:
            if embedder_name == 'bge':
                from sa_embedders.bge import get_embed_func, clear_memory
                # 메모리 정리
                clear_memory()
                embed_func = get_embed_func()
                print("✅ BGE 임베더 로드 완료 (메모리 최적화 모드)")
            elif embedder_name == 'openai':
                from sa_embedders.openai import get_embed_func
                embed_func = get_embed_func(api_key=openai_api_key, model=openai_model)
                print("✅ OpenAI 임베더 로드 완료")
            else:
                embed_func = None
                print("🔄 임베더 없이 기본 텍스트 분할 사용")
        except Exception as e:
            logger.warning(f"임베더 로드 실패: {e}, 기본 처리로 진행")
            embed_func = None
        
        from sa_tokenizers.jieba_mecab import split_src_meaning_units, split_tgt_by_src_units_semantic
        from punctuation import mask_brackets, restore_brackets
        
        # 메모리 모니터링
        import psutil
        process = psutil.Process()
        
        for i, sentence_data in enumerate(sentence_data_list):
            try:
                sentence_id = sentence_data['sentence_id']
                src_text = sentence_data['src_text']
                tgt_text = sentence_data['tgt_text']
                
                # 10문장마다 메모리 정리 (BGE)
                if embedder_name == 'bge' and i % 10 == 0 and i > 0:
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # 메모리 사용량 출력
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    print(f"🧠 메모리 사용량: {memory_mb:.1f}MB (문장 {i+1}/{total_sentences})")
                
                # 처리 파이프라인
                masked_src, src_masks = mask_brackets(src_text, 'source')
                masked_tgt, tgt_masks = mask_brackets(tgt_text, 'target')
                
                src_units = split_src_meaning_units(masked_src)
                
                # 임베더 사용 여부에 따른 처리
                if embed_func:
                    tgt_units = split_tgt_by_src_units_semantic(
                        src_units, masked_tgt, embed_func, min_tokens=1
                    )
                else:
                    # 기본 처리 - 단순 분할
                    tgt_words = masked_tgt.split()
                    words_per_unit = max(1, len(tgt_words) // len(src_units)) if src_units else 1
                    
                    tgt_units = []
                    for j in range(len(src_units)):
                        start_idx = j * words_per_unit
                        end_idx = min((j + 1) * words_per_unit, len(tgt_words))
                        unit = ' '.join(tgt_words[start_idx:end_idx])
                        if unit.strip():
                            tgt_units.append(unit)
                
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
                
                # 진행률 업데이트
                progress_bar.update(1)
                if (i + 1) % 50 == 0 or (i + 1) == total_sentences:
                    progress_bar.set_description(f"🔄 BGE 최적화 처리 {i+1}/{total_sentences}")
                
            except Exception as e:
                logger.error(f"문장 {sentence_data.get('sentence_id', '?')} 순차 처리 오류: {e}")
                # 실패해도 진행률은 업데이트
                progress_bar.update(1)
                continue
    
    finally:
        # 진행률 바 완료
        progress_bar.close()
        
        # BGE 사용 시 최종 메모리 정리
        if embedder_name == 'bge':
            from sa_embedders.bge import clear_memory
            clear_memory()
            print("🧹 BGE 메모리 정리 완료")
    
    # 결과 검증 및 저장
    if not results:
        raise ValueError("처리된 결과가 없습니다")
    
    print(f"💾 결과 저장 중: {output_path}")
    result_df = pd.DataFrame(results)
    
    if output_path.endswith('.xlsx'):
        result_df.to_excel(output_path, index=False)
    else:
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"🎉 처리 완료!")
    print(f"📊 결과: {total_sentences}개 문장 → {len(results)}개 구")
    print(f"📁 출력: {output_path}")
    
    logger.info(f"처리 완료: {total_sentences}개 문장 → {len(results)}개 구")
    logger.info(f"출력: {output_path}")
    
    return result_df