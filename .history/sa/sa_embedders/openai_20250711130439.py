"""OpenAI 임베딩 API 기반 임베더"""

import os
import logging
import numpy as np
from typing import List, Optional, Union
import json
import hashlib
from pathlib import Path
import time
import random

logger = logging.getLogger(__name__)

# OpenAI 클라이언트 초기화
try:
    import openai
    from openai import OpenAI
    logger.debug("✅ OpenAI 라이브러리 로드 성공")
except ImportError:
    logger.error("❌ OpenAI 라이브러리가 설치되지 않았습니다. pip install openai")
    OPENAI_AVAILABLE = False
    OpenAI = None

# 캐시 설정
CACHE_DIR = Path("embeddings_cache_openai")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "openai_embeddings.json"

# 임베딩 캐시 (메모리)
_embedding_cache = {}

def _load_cache():
    """캐시 파일에서 임베딩 로드"""
    global _embedding_cache
    
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                _embedding_cache = {k: np.array(v) for k, v in cache_data.items()}
            logger.info(f"📂 OpenAI 캐시 로드: {len(_embedding_cache)}개 항목")
        except Exception as e:
            logger.warning(f"⚠️ 캐시 로드 실패: {e}")
            _embedding_cache = {}
    else:
        _embedding_cache = {}

def _save_cache():
    """임베딩을 캐시 파일에 저장"""
    global _embedding_cache
    
    try:
        cache_data = {k: v.tolist() for k, v in _embedding_cache.items()}
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False)
        logger.info(f"💾 OpenAI 캐시 저장: {len(_embedding_cache)}개 항목")
    except Exception as e:
        logger.warning(f"⚠️ 캐시 저장 실패: {e}")

def _get_cache_key(text: str) -> str:
    """텍스트에 대한 캐시 키 생성"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def compute_embeddings_batch(
    texts: List[str], 
    model: str = "text-embedding-3-large",
    max_batch_size: int = 100
) -> List[np.ndarray]:
    """OpenAI API로 배치 임베딩 생성"""
    
    if not texts:
        return []
    
    try:
        client = _get_openai_client()
        
        # 배치 크기 제한
        all_embeddings = []
        
        for i in range(0, len(texts), max_batch_size):
            batch_texts = texts[i:i + max_batch_size]
            
            logger.info(f"🔄 OpenAI API 호출: {len(batch_texts)}개 텍스트 (배치 {i//max_batch_size + 1})")
            
            response = client.embeddings.create(
                model=model,
                input=batch_texts,
                encoding_format="float",
                timeout=60  # 60초 제한
            )
            
            batch_embeddings = [np.array(item.embedding) for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            logger.info(f"✅ OpenAI 임베딩 생성: {len(batch_embeddings)}개 → 차원: {len(batch_embeddings[0])}")
        
        return all_embeddings
        
    except Exception as e:
        logger.error(f"❌ OpenAI 임베딩 생성 실패: {e}")
        raise

def compute_embeddings_with_cache(
    texts: Union[str, List[str]], 
    model: str = "text-embedding-3-large",
    use_cache: bool = True,
    api_key: Optional[str] = None
) -> Union[np.ndarray, List[np.ndarray]]:
    """캐시를 사용한 OpenAI 임베딩 생성"""
    
    # 초기화
    if not _embedding_cache and use_cache:
        _load_cache()
    
    # 단일 텍스트 처리
    if isinstance(texts, str):
        texts = [texts]
        return_single = True
    else:
        return_single = False
    
    # 캐시에서 찾기
    cached_embeddings = {}
    missing_texts = []
    missing_indices = []
    
    if use_cache:
        for i, text in enumerate(texts):
            cache_key = _get_cache_key(text)
            if cache_key in _embedding_cache:
                cached_embeddings[i] = _embedding_cache[cache_key]
            else:
                missing_texts.append(text)
                missing_indices.append(i)
    else:
        missing_texts = texts
        missing_indices = list(range(len(texts)))
    
    # 캐시 히트 로그
    if use_cache and cached_embeddings:
        logger.info(f"📂 캐시 히트: {len(cached_embeddings)}개, 누락: {len(missing_texts)}개")
    
    # 누락된 텍스트들 API 호출
    new_embeddings = {}
    if missing_texts:
        try:
            # OpenAI 클라이언트 설정
            if openai is None:
                raise ImportError("OpenAI 라이브러리가 설치되지 않았습니다")
            
            if api_key:
                openai.api_key = api_key
            else:
                # API 키 확인
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "OPENAI_API_KEY 환경변수가 설정되지 않았습니다.\n"
                        "Windows: set OPENAI_API_KEY=your-api-key\n"
                        "Linux/Mac: export OPENAI_API_KEY=your-api-key"
                    )
            
            # 배치 임베딩 생성
            batch_embeddings = compute_embeddings_batch(missing_texts, model)
            
            for i, (idx, embedding) in enumerate(zip(missing_indices, batch_embeddings)):
                new_embeddings[idx] = embedding
                
                # 캐시에 저장
                if use_cache:
                    cache_key = _get_cache_key(missing_texts[i])
                    _embedding_cache[cache_key] = embedding
            
            # 캐시 파일 저장
            if use_cache and new_embeddings:
                _save_cache()
                
        except Exception as e:
            logger.error(f"❌ OpenAI API 호출 실패: {e}")
            raise
    
    # 결과 조합
    all_embeddings = []
    for i in range(len(texts)):
        if i in cached_embeddings:
            all_embeddings.append(cached_embeddings[i])
        elif i in new_embeddings:
            all_embeddings.append(new_embeddings[i])
        else:
            raise ValueError(f"임베딩을 찾을 수 없습니다: {texts[i]}")
    
    logger.info(f"✅ OpenAI 임베딩 완료: {len(all_embeddings)}개")
    
    if return_single:
        return all_embeddings[0]
    else:
        return all_embeddings

def get_embedding_dimension(model: str = "text-embedding-3-small") -> int:
    """임베딩 차원 반환"""
    
    dimension_map = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536
    }
    
    return dimension_map.get(model, 1536)

def test_openai_connection(model: str = "text-embedding-3-small") -> bool:
    """OpenAI 연결 테스트"""
    
    try:
        logger.info("🔍 OpenAI API 연결 테스트...")
        
        test_embeddings = compute_embeddings_with_cache(
            ["테스트 문장입니다."], 
            model=model,
            use_cache=False
        )
        
        logger.info(f"✅ OpenAI 연결 성공! 차원: {len(test_embeddings[0])}")
        return True
        
    except Exception as e:
        logger.error(f"❌ OpenAI 연결 실패: {e}")
        return False

class OpenAIEmbedder:
    """OpenAI 임베더 - 429 에러 처리 포함"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-large", max_retries: int = 5):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not available")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.request_count = 0
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 최소 1초 간격
    
    def _wait_for_rate_limit(self):
        """요청 간격 제어"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _exponential_backoff(self, retry_count: int) -> float:
        """지수 백오프 계산"""
        base_delay = 1.0
        max_delay = 60.0
        delay = min(base_delay * (2 ** retry_count) + random.uniform(0, 1), max_delay)
        return delay
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """텍스트 임베딩 - 429 에러 재시도 로직"""
        if not texts:
            return np.array([])
        
        # 텍스트를 작은 배치로 분할 (OpenAI API 제한 고려)
        batch_size = 10  # 한 번에 최대 10개씩만 처리
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self._embed_batch_with_retry(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            # 배치 간 대기 (429 에러 방지)
            if i + batch_size < len(texts):
                time.sleep(0.5)
        
        return np.array(all_embeddings)
    
    def _embed_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        """배치 임베딩 - 재시도 로직 포함"""
        for retry in range(self.max_retries):
            try:
                # 요청 간격 제어
                self._wait_for_rate_limit()
                
                # OpenAI API 호출
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts
                )
                
                embeddings = [data.embedding for data in response.data]
                self.request_count += 1
                
                logger.info(f"OpenAI 임베딩 성공: {len(texts)}개 텍스트 (총 요청: {self.request_count})")
                return embeddings
                
            except openai.RateLimitError as e:
                # 429 에러 처리
                retry_after = getattr(e.response, 'headers', {}).get('Retry-After')
                if retry_after:
                    wait_time = int(retry_after)
                else:
                    wait_time = self._exponential_backoff(retry)
                
                logger.warning(f"OpenAI 429 에러 (시도 {retry+1}/{self.max_retries}): {wait_time:.1f}초 대기")
                print(f"⏳ OpenAI API 한도 초과, {wait_time:.1f}초 대기 중... (시도 {retry+1}/{self.max_retries})")
                
                time.sleep(wait_time)
                
                if retry == self.max_retries - 1:
                    logger.error("OpenAI API 재시도 한도 초과")
                    raise
                    
            except Exception as e:
                logger.error(f"OpenAI 임베딩 오류 (시도 {retry+1}/{self.max_retries}): {e}")
                
                if retry < self.max_retries - 1:
                    wait_time = self._exponential_backoff(retry)
                    time.sleep(wait_time)
                else:
                    raise

def get_embed_func(api_key: str = None, model: str = "text-embedding-3-large") -> Optional[callable]:
    """OpenAI 임베딩 함수 반환 - 실패 시 None 반환"""
    try:
        import openai
        from openai import OpenAI
    except ImportError as e:
        error_msg = "OpenAI 패키지가 설치되지 않았습니다"
        logger.error(error_msg)
        print(f"❌ {error_msg}: {e}")
        raise ImportError(error_msg + "\n설치 명령: pip install openai")
    
    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            error_msg = "OpenAI API 키가 설정되지 않았습니다"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            print("환경변수 설정: export OPENAI_API_KEY=your-api-key")
            raise ValueError(error_msg)
    
    try:
        embedder = OpenAIEmbedder(api_key=api_key, model=model, max_retries=5)
        
        def embed_func(texts: List[str]) -> np.ndarray:
            return embedder.embed(texts)
        
        logger.info(f"OpenAI 임베더 초기화 성공 (모델: {model})")
        print(f"✅ OpenAI 임베더 초기화 성공 (모델: {model})")
        return embed_func
        
    except Exception as e:
        error_msg = f"OpenAI 임베더 초기화 실패: {e}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)

def compute_embeddings_with_cache(texts: List[str], model: str = "text-embedding-3-large", api_key: str = None) -> np.ndarray:
    """OpenAI 임베딩 계산 - 캐시 및 429 에러 처리"""
    embedder = OpenAIEmbedder(api_key=api_key, model=model, max_retries=5)
    return embedder.embed(texts)

# 모듈 로드시 연결 테스트 (옵션)
if __name__ == "__main__":
    # 직접 실행시에만 테스트
    if test_openai_connection():
        print("🎉 OpenAI 임베더 정상 작동!")
    else:
        print("❌ OpenAI 임베더 연결 실패!")