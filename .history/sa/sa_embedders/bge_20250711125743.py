"""BGE 임베더 - 메모리 최적화 버전 (더미 모드 제거)"""

import logging
import numpy as np
import torch
import os
import gc
from typing import List, Optional, Callable
from tqdm import tqdm

logger = logging.getLogger(__name__)

# 전역 설정 - 메모리 절약 모드
DEFAULT_BATCH_SIZE = 3  # 배치 크기 더 축소
DEFAULT_EMBEDDING_MODEL = 'BAAI/bge-m3'
MAX_CACHE_SIZE = 500  # 캐시 크기 더 제한

class EmbeddingManager:
    """메모리 최적화된 임베딩 계산 클래스 - 더미 모드 제거"""
    
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device_id=None):
        self.model_name = model_name
        self.model = None
        self._cache = {}
        self._model_loaded = False
        self.process_id = os.getpid()
        self.device_id = device_id
        self._max_cache_size = MAX_CACHE_SIZE
    
    def _load_model(self):
        """메모리 효율적인 모델 로딩 - 실패 시 예외 발생"""
        if self._model_loaded and os.getpid() == self.process_id:
            return
            
        # 프로세스 변경 시 재초기화
        if os.getpid() != self.process_id:
            self.process_id = os.getpid()
            self._model_loaded = False
            self.model = None
            self._cache.clear()
        
        try:
            from FlagEmbedding import BGEM3FlagModel
            print(f"프로세스 {self.process_id}: BGE 모델 로딩 중... (메모리 최적화 모드)")
            
            # 메모리 최적화 설정
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:False'
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # 디바이스 설정
            if torch.cuda.is_available() and self.device_id is not None:
                device = f'cuda:{self.device_id}'
                # GPU 메모리 fraction 설정
                torch.cuda.set_per_process_memory_fraction(0.3, device=self.device_id)
            elif torch.cuda.is_available():
                device = 'cuda'
                torch.cuda.set_per_process_memory_fraction(0.3)
            else:
                device = 'cpu'
                print(f"⚠️ GPU를 사용할 수 없어 CPU 모드로 실행합니다. 처리 속도가 매우 느릴 수 있습니다.")
            
            # 모델 로딩 (FP16 사용으로 메모리 절약)
            self.model = BGEM3FlagModel(
                self.model_name,
                use_fp16=True if device != 'cpu' else False,
                device=device,
                normalize_embeddings=True
            )
            
            self._model_loaded = True
            print(f"프로세스 {self.process_id}: BGE 모델 로딩 완료 (device={device}, 메모리 절약 모드)")
            
        except ImportError as e:
            error_msg = f"FlagEmbedding 패키지가 설치되지 않았습니다: {e}"
            print(f"❌ {error_msg}")
            logger.error(error_msg)
            raise ImportError(error_msg + "\n설치 명령: pip install FlagEmbedding")
            
        except Exception as e:
            error_msg = f"BGE 모델 초기화 실패: {e}"
            print(f"❌ 프로세스 {self.process_id}: {error_msg}")
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _manage_cache(self):
        """캐시 크기 관리"""
        if len(self._cache) > self._max_cache_size:
            # LRU 방식으로 오래된 항목 제거
            keys_to_remove = list(self._cache.keys())[:-self._max_cache_size//2]
            for key in keys_to_remove:
                del self._cache[key]
            gc.collect()
    
    def compute_embeddings_with_cache(
        self,
        texts: List[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_batch_progress: bool = False
    ) -> np.ndarray:
        """메모리 최적화된 임베딩 계산 - 실패 시 예외 발생"""
        
        if not texts:
            return np.array([])
        
        # 모델 로딩 (실패 시 예외 발생)
        self._load_model()
        
        # 배치 크기를 더 작게 조정 (메모리 절약)
        batch_size = min(batch_size, 3)
        
        result_list: List[Optional[np.ndarray]] = [None] * len(texts)
        to_embed: List[str] = []
        indices_to_embed: List[int] = []

        # 캐시 확인
        for i, txt in enumerate(texts):
            cache_key = hash(txt) % (2**31)
            if cache_key in self._cache:
                result_list[i] = self._cache[cache_key]
            else:
                to_embed.append(txt)
                indices_to_embed.append(i)

        # 새 임베딩 계산 (실제 BGE 모델만 사용)
        if to_embed:
            embeddings = []
            
            progress_desc = f"BGE 임베딩 (프로세스 {self.process_id})"
            progress_iter = range(0, len(to_embed), batch_size)
            
            if show_batch_progress and len(to_embed) > batch_size:
                progress_iter = tqdm(progress_iter, desc=progress_desc, leave=False)
            
            for start in progress_iter:
                batch = to_embed[start:start + batch_size]
                
                try:
                    # 배치 처리 전 메모리 정리
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    # 임베딩 계산 (메모리 효율적)
                    with torch.no_grad():
                        output = self.model.encode(
                            batch,
                            return_dense=True,
                            return_sparse=False,
                            return_colbert_vecs=False,
                            batch_size=1,  # 내부 배치 크기도 1로 제한
                            max_length=512  # 최대 길이 제한
                        )
                        dense = output['dense_vecs']
                        
                        # GPU 텐서를 CPU로 즉시 이동
                        if isinstance(dense, torch.Tensor):
                            dense = dense.cpu().numpy()
                        elif isinstance(dense, list) and len(dense) > 0 and isinstance(dense[0], torch.Tensor):
                            dense = [emb.cpu().numpy() for emb in dense]
                        
                        embeddings.extend(dense)
                    
                    # 배치 처리 후 메모리 정리
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    error_msg = f"프로세스 {self.process_id}: 임베딩 계산 실패: {e}"
                    print(f"❌ {error_msg}")
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

            # 캐시 업데이트
            for i, (txt, emb) in enumerate(zip(to_embed, embeddings)):
                cache_key = hash(txt) % (2**31)
                self._cache[cache_key] = emb
                result_list[indices_to_embed[i]] = emb
            
            # 캐시 크기 관리
            self._manage_cache()

        return np.array(result_list)

# 전역 인스턴스
_embedding_manager = None

def get_embedding_manager(device_id=None) -> EmbeddingManager:
    """임베딩 매니저 싱글톤 반환"""
    global _embedding_manager
    if _embedding_manager is None or os.getpid() != getattr(_embedding_manager, 'process_id', -1):
        _embedding_manager = EmbeddingManager(device_id=device_id)
    return _embedding_manager

def compute_embeddings_with_cache(texts: List[str], **kwargs) -> np.ndarray:
    """하위 호환성 함수 - 실패 시 예외 발생"""
    allowed_keys = {'batch_size', 'show_batch_progress'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
    
    # 배치 크기 강제 제한
    if 'batch_size' in filtered_kwargs:
        filtered_kwargs['batch_size'] = min(filtered_kwargs['batch_size'], 3)
    else:
        filtered_kwargs['batch_size'] = 3
    
    manager = get_embedding_manager()
    return manager.compute_embeddings_with_cache(texts, **filtered_kwargs)

def get_embed_func(device_id=None) -> Callable:
    """임베딩 함수 반환 - 실패 시 예외 발생"""
    def embed_func(texts: List[str]) -> np.ndarray:
        return compute_embeddings_with_cache(texts, batch_size=3, show_batch_progress=False)
    return embed_func

# 메모리 정리 함수
def clear_memory():
    """메모리 정리"""
    global _embedding_manager
    if _embedding_manager:
        _embedding_manager._cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()