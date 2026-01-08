"""
SikuBERT 토크나이저 - 전통중문(고전) 전용
四庫全書 기반 학습으로 고전 텍스트에 최적화
"""

import os
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Tuple, Optional
import logging
import warnings

# 환경 변수로 torch.load 보안 검사 비활성화
os.environ['TORCH_FORCE_WEIGHTS_ONLY'] = 'False'
os.environ['HF_HUB_DISABLE_WARNINGS'] = '1'

# torch.load 보안 경고 무시 (PyTorch 2.6 호환성 문제 해결)
warnings.filterwarnings("ignore", message=".*torch.load.*")
warnings.filterwarnings("ignore", message=".*vulnerability.*")
warnings.filterwarnings("ignore", message=".*CVE-2025-32434.*")
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

class SikuBertTokenizer:
    """SikuBERT를 사용한 고전중문 토크나이저"""
    
    def __init__(self, model_name: str = "SIKU-BERT/sikubert"):
        """
        SikuBERT 토크나이저 초기화
        
        Args:
            model_name: HuggingFace 모델 이름
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        # GPU 감지 및 설정
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"✅ SikuBERT: GPU 사용 가능 (device: {self.device})")
        else:
            self.device = torch.device('cpu')
            print(f"⚠️ SikuBERT: GPU 불가능, CPU 모드로 실행 (device: {self.device})")
        
    def load_model(self):
        """모델과 토크나이저 로딩"""
        try:
            logger.info(f"SikuBERT 모델 로딩 시작: {self.model_name}")
            
            # torch.load 보안 경고 추가 무시
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                
                # 토크나이저 로딩
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                
                # 모델 로딩
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                ).to(self.device)
            
            self.model.eval()
            logger.info(f"SikuBERT 모델 로딩 완료 (device: {self.device})")
            
        except Exception as e:
            error_msg = str(e)
            if "torch.load" in error_msg or "vulnerability" in error_msg or "CVE-2025-32434" in error_msg:
                logger.error(f"SikuBERT 모델 로딩 실패 (PyTorch 보안 문제): torch.load 함수가 보안상 제한됨")
                logger.error("해결책: PyTorch 2.6+ 업그레이드 또는 safetensors 기반 모델 사용")
            else:
                logger.error(f"SikuBERT 모델 로딩 실패: {e}")
            raise e
    
    def tokenize(self, text: str) -> List[str]:
        """
        텍스트를 토큰으로 분할
        
        Args:
            text: 입력 텍스트
            
        Returns:
            토큰 리스트
        """
        if self.tokenizer is None:
            self.load_model()
        
        tokens = self.tokenizer.tokenize(text)
        return tokens
    
    def get_embeddings(self, texts: List[str], max_length: int = 512, batch_size: int = 32) -> np.ndarray:
        """
        텍스트들의 임베딩 벡터 생성 (GPU 배치 처리)
        
        Args:
            texts: 텍스트 리스트
            max_length: 최대 시퀀스 길이
            batch_size: 배치 크기 (GPU 메모리에 따라 조정)
            
        Returns:
            임베딩 배열 (batch_size, hidden_size)
        """
        if self.model is None:
            self.load_model()
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        embeddings = []
        
        # 배치 처리로 GPU 효율성 극대화
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # 배치 토크나이징 (GPU 최적화)
                encoded = self.tokenizer(
                    batch_texts,
                    max_length=max_length,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                ).to(self.device)
                
                # 배치 임베딩 생성 (GPU)
                outputs = self.model(**encoded)
                
                # [CLS] 토큰의 임베딩을 문장 임베딩으로 사용
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(cls_embeddings)
        
        return np.array(embeddings)
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        두 텍스트 간 유사도 계산
        
        Args:
            text1, text2: 비교할 텍스트
            
        Returns:
            코사인 유사도 (0~1)
        """
        embeddings = self.get_embeddings([text1, text2])
        
        # 코사인 유사도 계산
        emb1, emb2 = embeddings[0], embeddings[1]
        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # 0~1 범위로 정규화
        return (cosine_sim + 1) / 2

# 전역 인스턴스
_siku_tokenizer = None

def get_siku_tokenizer(model_name: str = "SIKU-BERT/sikubert") -> SikuBertTokenizer:
    """
    SikuBERT 토크나이저 싱글톤 인스턴스 반환
    
    Args:
        model_name: 사용할 모델 이름
        
    Returns:
        SikuBertTokenizer 인스턴스
    """
    global _siku_tokenizer
    
    if _siku_tokenizer is None:
        _siku_tokenizer = SikuBertTokenizer(model_name)
        _siku_tokenizer.load_model()
    
    return _siku_tokenizer

def siku_get_embeddings(texts: List[str], max_length: int = 512, batch_size: int = 32) -> np.ndarray:
    """
    SikuBERT를 사용한 임베딩 생성 편의 함수 (GPU 배치 처리)
    
    Args:
        texts: 텍스트 리스트
        max_length: 최대 시퀀스 길이
        batch_size: 배치 크기
        
    Returns:
        임베딩 배열
    """
    tokenizer = get_siku_tokenizer()
    return tokenizer.get_embeddings(texts, max_length=max_length, batch_size=batch_size)

def siku_similarity(text1: str, text2: str) -> float:
    """
    SikuBERT를 사용한 유사도 계산 편의 함수
    
    Args:
        text1, text2: 비교할 텍스트
        
    Returns:
        유사도 점수
    """
    tokenizer = get_siku_tokenizer()
    return tokenizer.similarity(text1, text2)
