"""
중국어 시대 감지 통합 모듈
SikuBERT 모델과 폴백 방식을 결합한 시대 분류기 (전근대 고전 전용)
"""

import logging
import pickle
import os
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class PeriodDetector:
    """중국어 시대 감지 통합 클래스"""
    
    def __init__(self, use_siku: bool = True, cache_file: str = "period_references.pkl"):
        """
        Args:
            use_siku: SikuBERT 모델 사용 여부 (전근대 고전 전용)
            cache_file: 참조 임베딩 캐시 파일명
        """
        self.use_siku = use_siku
        self.cache_file = cache_file
        self.period_references = None
        
        # 시대별 대표 텍스트
        self.reference_texts = {
            "shanggu": [
                "管子曰民者君之本也本固則國安矣",
                "論語曰學而時習之不亦說乎",
                "孟子曰仁義禮智非外鑠我也",
                "老子曰道可道非常道名可名非常名",
                "莊子曰天地與我並生而萬物與我為一"
            ],
            "zhonggu": [
                "大乘無量壽經佛說阿彌陀經",
                "般若波羅蜜多心經觀自在菩薩",
                "禪宗六祖慧能大師法寶壇經",
                "華嚴經如來藏性清淨光明",
                "法華經妙法蓮華經觀世音菩薩"
            ],
            "jindai": [
                "朱子語類理氣論說格物致知",
                "程朱理學性理大全誠意正心",
                "王陽明傳習錄知行合一致良知",
                "四書章句集注大學中庸論孟",
                "性理精義理學宗旨修身齊家"
            ],
            "xiandai": [
                "這是現代漢語的例子可以應該",
                "现在我们应该可能或者因为所以",
                "中國現代化建設改革開放發展",
                "科學技術現代社會經濟發展",
                "教育文化現代文明進步發展"
            ]
        }
        
        # 폴백용 키워드 패턴
        self.fallback_patterns = {
            "shanggu": {
                "keywords": ["之", "也", "者", "矣", "焉", "乎", "哉", "管子", "論語", "孟子"],
                "weight": 3.0
            },
            "zhonggu": {
                "keywords": ["佛", "菩薩", "般若", "涅槃", "禪", "法", "經", "如來"],
                "weight": 2.5
            },
            "jindai": {
                "keywords": ["朱子", "語類", "理氣", "程朱", "理學", "格物", "致知"],
                "weight": 2.0
            },
            "xiandai": {
                "keywords": ["這是", "現代", "应该", "可能", "或者", "因为", "所以", "例子"],
                "weight": 1.5
            }
        }
    
    def _get_siku_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """SikuBERT 모델로 임베딩 추출 (현재는 전근대 고전으로 가정)"""
        try:
            from .siku_tokenizer import siku_get_embeddings
            return siku_get_embeddings(texts)
        except Exception as e:
            logger.error(f"SikuBERT 임베딩 추출 실패: {e}")
            return [np.zeros(768) for _ in texts]
    
    def _build_period_references(self):
        """시대별 참조 임베딩 구축"""
        if not self.use_siku:
            return
            
        logger.info("SikuBERT 기반 시대별 참조 임베딩 구축 중... (전근대 고전 전용)")
        
        references = {}
        
        for period, texts in self.reference_texts.items():
            logger.info(f"{period} 시대 임베딩 추출 중...")
            
            embeddings = self._get_siku_embeddings(texts)
            
            # 평균 임베딩 계산
            if embeddings and len(embeddings) > 0:
                avg_embedding = np.mean(embeddings, axis=0)
                references[period] = avg_embedding
                logger.info(f"{period} 완료")
            else:
                logger.warning(f"{period} 임베딩 추출 실패")
        
        self.period_references = references
        
        # 캐시 저장
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(references, f)
            logger.info(f"참조 임베딩 캐시 저장: {self.cache_file}")
        except Exception as e:
            logger.warning(f"캐시 저장 실패: {e}")
    
    def _load_period_references(self) -> bool:
        """캐시된 참조 임베딩 로드"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.period_references = pickle.load(f)
                logger.info("캐시된 참조 임베딩 로드 완료")
                return True
            except Exception as e:
                logger.warning(f"캐시 로드 실패: {e}")
        return False
    
    def _detect_by_siku(self, text: str) -> str:
        """SikuBERT 모델 기반 시대 감지 (전근대 고전 전용)"""
        # 참조 임베딩 준비
        if self.period_references is None:
            if not self._load_period_references():
                self._build_period_references()
        
        if not self.period_references:
            raise Exception("참조 임베딩이 없음")
        
        # 입력 텍스트 임베딩 추출
        text_embeddings = self._get_siku_embeddings([text])
        if not text_embeddings or len(text_embeddings) == 0:
            raise Exception("텍스트 임베딩 추출 실패")
        
        text_embedding = text_embeddings[0]
        
        # 각 시대와의 유사도 계산
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = {}
        for period, ref_embedding in self.period_references.items():
            similarity = cosine_similarity(
                text_embedding.reshape(1, -1),
                ref_embedding.reshape(1, -1)
            )[0][0]
            similarities[period] = similarity
        
        # 최고 유사도 시대 선택
        detected_period = max(similarities, key=similarities.get)
        max_similarity = similarities[detected_period]
        
        logger.debug(f"SikuBERT 시대별 유사도: {similarities}")
        logger.info(f"SikuBERT 감지: {detected_period} (유사도: {max_similarity:.3f}, 전근대 고전 전용)")
        
        return detected_period
    
    def _detect_by_fallback(self, text: str) -> str:
        """폴백 키워드 기반 시대 감지"""
        scores = {}
        
        for period, pattern in self.fallback_patterns.items():
            score = 0.0
            for keyword in pattern["keywords"]:
                score += text.count(keyword) * pattern["weight"]
            scores[period] = score
        
        if max(scores.values()) > 0:
            detected_period = max(scores, key=scores.get)
        else:
            # 스마트 기본값
            if any(word in text for word in ["這是", "現代", "例子", "漢語"]):
                detected_period = "xiandai"
            elif any(word in text for word in ["朱子", "語類", "理氣", "論說"]):
                detected_period = "jindai"
            elif any(word in text for word in ["佛", "經", "無量", "壽經"]):
                detected_period = "zhonggu"
            else:
                detected_period = "shanggu"
        
        logger.debug(f"폴백 시대별 점수: {scores}")
        logger.info(f"폴백 감지: {detected_period}")
        
        return detected_period
    
    def detect(self, text: str) -> str:
        """시대 감지 (통합)"""
        if not text or not text.strip():
            return "shanggu"
        
        # SikuBERT 모델 우선 시도 (전근대 고전 전용)
        if self.use_siku:
            try:
                return self._detect_by_siku(text)
            except Exception as e:
                logger.warning(f"SikuBERT 모델 실패, 폴백 사용: {e}")
        
        # 폴백 방식 (전근대 고전으로 가정)
        return "shanggu"  # 전근대 고전으로 기본 설정

# 전역 인스턴스 (싱글톤)
_period_detector = None

def get_period_detector(use_siku: bool = True) -> PeriodDetector:
    """시대 감지기 싱글톤 인스턴스 반환 (전근대 고전 전용)"""
    global _period_detector
    
    if _period_detector is None:
        _period_detector = PeriodDetector(use_siku=use_siku)
    
    return _period_detector

def detect_chinese_period(text: str, use_koichi: bool = False) -> str:
    """중국어 시대 감지 편의 함수 (전근대 고전으로 가정)"""
    # 호환성을 위해 use_koichi 파라미터는 유지하지만 무시하고 전근대 고전으로 처리
    detector = get_period_detector(use_siku=True)
    return "shanggu"  # 전근대 고전으로 고정
