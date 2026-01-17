"""Gemini Embedding Wrapper (REST API)

Google Gemini API를 사용하여 텍스트 임베딩을 생성합니다.
라이브러리 의존성을 피하기 위해 REST API를 직접 호출합니다.
"""
import os
import time
import json
import logging
import requests
from pathlib import Path
from typing import List, Dict, Union
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemini_embedder")

# 캐시 설정
CACHE_DIR = Path("embeddings_cache_gemini")
CACHE_FILE = CACHE_DIR / "gemini_embeddings.json"
CACHE_SAVE_INTERVAL = 50

# 전역 캐시
_embedding_cache: Dict[str, List[float]] = {}
_cache_dirty = False
_request_count = 0

def load_cache():
    """캐시 파일 로드"""
    global _embedding_cache
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                _embedding_cache = json.load(f)
            logger.info(f"Loaded {len(_embedding_cache)} embeddings from cache.")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            _embedding_cache = {}

def save_cache(force=False):
    """캐시 저장"""
    global _cache_dirty, _request_count
    if not _cache_dirty and not force:
        return
        
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_embedding_cache, f, ensure_ascii=False)
        _cache_dirty = False
        _request_count = 0
        if force:
            logger.info(f"Saved {len(_embedding_cache)} embeddings to cache.")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

def _call_gemini_embedding(text: str, model: str, api_key: str) -> List[float]:
    """Gemini API 호출하여 임베딩 생성 (단일 텍스트)"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={api_key}"
    
    payload = {
        "model": f"models/{model}",
        "content": {
            "parts": [{"text": text}]
        }
    }
    
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()
    
    return result['embedding']['values']

def _call_gemini_batch_embedding(texts: List[str], model: str, api_key: str) -> List[List[float]]:
    """Gemini API 호출하여 배치 임베딩 생성
    
    Note: batchEmbedContents는 분당 요청 제한이 엄격할 수 있으므로
    여기서는 단순화를 위해 개별 호출을 반복하거나, 지원되는 경우 배치 호출을 사용.
    Gemini Embed API는 batchEmbedContents 엔드포인트를 지원함.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents?key={api_key}"
    
    requests_payload = {
        "requests": [
            {
                "model": f"models/{model}",
                "content": {"parts": [{"text": t}]}
            }
            for t in texts
        ]
    }
    
    response = requests.post(url, json=requests_payload, timeout=60)
    response.raise_for_status()
    result = response.json()
    
    embeddings = []
    if 'embeddings' not in result:
        # Fallback or error
        raise ValueError(f"Invalid response from Gemini: {result}")
        
    for item in result['embeddings']:
        embeddings.append(item['values'])
        
    return embeddings

def compute_embeddings_batch(
    texts: List[str], 
    model: str = "text-embedding-004", 
    batch_size: int = 20  # Gemini 배치 제한 고려
) -> np.ndarray:
    """
    텍스트 리스트의 임베딩을 계산하여 numpy array로 반환.
    캐싱 기능 포함.
    """
    global _embedding_cache, _cache_dirty, _request_count
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    if not _embedding_cache:
        load_cache()
    
    results = []
    texts_to_process = []
    indices_to_process = []
    
    # 1. 캐시 확인
    for i, text in enumerate(texts):
        # 키 생성 (모델명 포함하여 구분 가능하도록)
        cache_key = f"{model}:{text}"
        if cache_key in _embedding_cache:
            results.append(_embedding_cache[cache_key])
        else:
            results.append(None)
            texts_to_process.append(text)
            indices_to_process.append(i)
    
    if not texts_to_process:
        return np.array(results, dtype=np.float32)
    
    # 2. API 호출 (배치 처리)
    logger.info(f"Computing Gemini embeddings for {len(texts_to_process)} texts...")
    
    for i in range(0, len(texts_to_process), batch_size):
        batch_texts = texts_to_process[i:i + batch_size]
        batch_indices = indices_to_process[i:i + batch_size]
        
        try:
            # API 호출
            embeddings = _call_gemini_batch_embedding(batch_texts, model, api_key)
            
            # 결과 저장
            for text, emb, idx in zip(batch_texts, embeddings, batch_indices):
                cache_key = f"{model}:{text}"
                _embedding_cache[cache_key] = emb
                results[idx] = emb
                _cache_dirty = True
            
            _request_count += 1
            if _request_count >= CACHE_SAVE_INTERVAL:
                save_cache()
                
            # Rate limit 방지 (심플하게)
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Gemini API Error details: {e}")
            # 배치가 실패하면 개별 시도? 여기서는 일단 스킵하거나 0으로 채움
            # 하지만 임베딩은 중요하므로 에러 발생시 중단하는게 나을 수 있음
            # fallback: wait and retry once
            time.sleep(5)
            try:
                embeddings = _call_gemini_batch_embedding(batch_texts, model, api_key)
                for text, emb, idx in zip(batch_texts, embeddings, batch_indices):
                    cache_key = f"{model}:{text}"
                    _embedding_cache[cache_key] = emb
                    results[idx] = emb
                    _cache_dirty = True
            except Exception as retry_e:
                logger.error(f"Retry failed: {retry_e}")
                # 0벡터로 채우기 (임시)
                dim = 768  # text-embedding-004 default
                for idx in batch_indices:
                    results[idx] = [0.0] * dim
    
    save_cache()
    return np.array(results, dtype=np.float32)

# 초기화 시 캐시 로드
load_cache()
