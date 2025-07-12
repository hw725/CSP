"""BGE 임베더 - 메모리 최적화 버전"""

import logging
import numpy as np
import torch
import os
import gc
from typing import List, Optional, Callable
from tqdm import tqdm

logger = logging.getLogger(__name__)

# 전역 설정 - 메모리 절약 모드
DEFAULT_BATCH_SIZE = 5  # 배치 크기 대폭 감소
DEFAULT_EMBEDDING_MODEL = 'BAAI/bge-m3'
MAX_CACHE_SIZE = 1000  # 캐시 크기 제한

class EmbeddingManager:
    """메모리 최적화된 임베딩 계산 클래스"""
    
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, fallback_to_dummy: bool = True, device_id=None):
        self.model_name = model_name
        self.model = None
        self._cache = {}
        self._fallback_to_dummy = fallback_to_dummy
        self._model_loaded = False
        self._use_dummy = False
        self.process_id = os.getpid()
        self.device_id = device_id
        self._max_cache_size = MAX_CACHE_SIZE
    
    def _load_model(self):
        """메모리 효율적인 모델 로딩"""
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
            
            # 모델 로딩 (FP16 사용으로 메모리 절약)
            self.model = BGEM3FlagModel(
                self.model_name,
                use_fp16=True,
                device=device,
                normalize_embeddings=True  # 정규화 활성화
            )
            
            self._model_loaded = True
            self._use_dummy = False
            print(f"프로세스 {self.process_id}: BGE 모델 로딩 완료 (device={device}, 메모리 절약 모드)")
            
        except Exception as e:
            print(f"프로세스 {self.process_id}: BGE 모델 로딩 실패: {e}")
            
            if self._fallback_to_dummy:
                print(f"프로세스 {self.process_id}: 더미 모드로 전환")
                self._use_dummy = True
                self._model_loaded = True
            else:
                raise RuntimeError(f"BGE 모델 초기화 실패: {e}")
    
    def _manage_cache(self):
        """캐시 크기 관리"""
        if len(self._cache) > self._max_cache_size:
            # LRU 방식으로 오래된 항목 제거
            keys_to_remove = list(self._cache.keys())[:-self._max_cache_size//2]
            for key in keys_to_remove:
                del self._cache[key]
            gc.collect()
    
    def _generate_dummy_embedding(self, text: str) -> np.ndarray:
        """더미 임베딩 생성"""
        seed = hash(text) % (2**31)
        np.random.seed(seed)
        dummy_emb = np.random.randn(1024).astype(np.float32)
        dummy_emb = dummy_emb / (np.linalg.norm(dummy_emb) + 1e-8)
        return dummy_emb
    
    def compute_embeddings_with_cache(
        self,
        texts: List[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_batch_progress: bool = False
    ) -> np.ndarray:
        """메모리 최적화된 임베딩 계산"""
        
        if not texts:
            return np.array([])
        
        # 모델 로딩
        self._load_model()
        
        # 배치 크기를 더 작게 조정 (메모리 절약)
        batch_size = min(batch_size, 3)
        
        result_list: List[Optional[np.ndarray]] = [None] * len(texts)
        to_embed: List[str] = []
        indices_to_embed: List[int] = []

        # 캐시 확인
        for i, txt in enumerate(texts):
            cache_key = hash(txt) % (2**31)  # 메모리 절약을 위해 해시 사용
            if cache_key in self._cache:
                result_list[i] = self._cache[cache_key]
            else:
                to_embed.append(txt)
                indices_to_embed.append(i)

        # 새 임베딩 계산
        if to_embed:
            if self._use_dummy:
                # 더미 임베딩 사용
                embeddings = [self._generate_dummy_embedding(text) for text in to_embed]
            else:
                # 실제 BGE 모델 사용 (메모리 최적화)
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
                        with torch.no_grad():  # 그래디언트 비활성화
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
                        print(f"프로세스 {self.process_id}: 임베딩 계산 실패: {e}")
                        # 실패한 배치는 더미로 대체
                        embeddings.extend([self._generate_dummy_embedding(text) for text in batch])

            # 캐시 업데이트
            for i, (txt, emb) in enumerate(zip(to_embed, embeddings)):
                cache_key = hash(txt) % (2**31)
                self._cache[cache_key] = emb
                result_list[indices_to_embed[i]] = emb

        return np.array(result_list)

# 전역 인스턴스
_embedding_manager = EmbeddingManager(fallback_to_dummy=True)

def compute_embeddings_with_cache(texts: List[str], **kwargs) -> np.ndarray:
    """하위 호환성 함수"""
    # model 등 불필요한 인자는 무시하고, 필요한 인자만 전달
    allowed_keys = {'batch_size', 'show_batch_progress'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
    return _embedding_manager.compute_embeddings_with_cache(texts, **filtered_kwargs)

def get_embed_func(device_id=None) -> Callable:
    """임베딩 함수 반환 (device_id 지정 가능)"""
    manager = EmbeddingManager(fallback_to_dummy=True, device_id=device_id)
    return manager.compute_embeddings_with_cache

def get_embedding_manager() -> EmbeddingManager:
    """임베딩 매니저 반환"""
    return _embedding_manager

# 프록시 클래스
class EmbeddingManagerProxy:
    def __getattr__(self, name):
        return getattr(_embedding_manager, name)

embedding_manager = EmbeddingManagerProxy()