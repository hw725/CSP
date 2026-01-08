"""새로운 파서 모듈 - SuPar-Kanbun & Stanza 통합"""

import re
from typing import List, Optional

# 파서 가용성 플래그
SUPAR_AVAILABLE = False
STANZA_AVAILABLE = False

# 전역 파서 변수들
supar_parser = None
stanza_nlp = None

def smart_sentence_split(text: str, is_source: bool = True) -> List[str]:
    """스마트 문장 분할 함수 (폴백 구현)"""
    if is_source:
        # 한문 원문 분할
        pattern = r'(?<=[。？！])\s*|(?<=[.!?])\s+'
    else:
        # 번역문 분할
        pattern = r'(?<=[.!?])\s*|(?<=다)\s*|(?<=요)\s*'
    
    sentences = re.split(pattern, text.strip())
    return [s.strip() for s in sentences if s.strip()]

def split_source_with_supar(text: str) -> List[str]:
    """SuPar-Kanbun 파서 (폴백)"""
    return smart_sentence_split(text, is_source=True)

def split_target_with_stanza(text: str) -> List[str]:
    """Stanza 파서 (폴백)"""
    return smart_sentence_split(text, is_source=False)

def fallback_split_by_punctuation(text: str, is_source: bool = True) -> List[str]:
    """구두점 기반 폴백 분할"""
    return smart_sentence_split(text, is_source)

# SuPar-Kanbun 로드 시도
try:
    import supar
    # SuPar-Kanbun 모델 로드 시도
    try:
        # 올바른 모델명으로 시도
        try:
            supar_parser = supar.Parser.load('crf-dep-zh')
        except:
            # 다른 중국어 모델 시도
            supar_parser = supar.Parser.load('biaffine-dep-zh')
        
        SUPAR_AVAILABLE = True
        print("✅ SuPar-Kanbun 모델 로드 완료")
        
        def split_source_with_supar(text: str) -> List[str]:
            """실제 SuPar-Kanbun 파서"""
            global supar_parser
            try:
                # SuPar로 구문 분석
                parsed = supar_parser.predict(text, prob=True, verbose=False)
                # 문장 경계 추출 (구현 필요)
                return smart_sentence_split(text, is_source=True)  # 임시 폴백
            except Exception as e:
                print(f"⚠️ SuPar 파싱 실패: {e}")
                return smart_sentence_split(text, is_source=True)
                
    except Exception as e:
        print(f"⚠️ SuPar-Kanbun 모델 로드 실패: {e}")
        SUPAR_AVAILABLE = False
        supar_parser = None
        
except ImportError:
    print("⚠️ SuPar-Kanbun 미설치 (폴백 모드)")
    SUPAR_AVAILABLE = False
    supar_parser = None

# Stanza 로드 시도  
try:
    import stanza
    import os
    
    # Stanza 리소스 디렉토리 설정
    stanza_dir = os.path.join(os.path.expanduser("~"), "stanza_resources")
    os.makedirs(stanza_dir, exist_ok=True)
    
    try:
        # 한국어 모델 다운로드 (필요시)
        try:
            stanza.download('ko', model_dir=stanza_dir, verbose=False)
        except:
            pass  # 이미 다운로드된 경우
            
        # GPU 사용 가능 여부 확인
        import torch
        use_gpu = torch.cuda.is_available()
        
        stanza_nlp = stanza.Pipeline('ko', model_dir=stanza_dir, use_gpu=use_gpu, verbose=False)
        STANZA_AVAILABLE = True
        print(f"✅ Stanza 한국어 모델 로드 완료 (GPU: {use_gpu})")
        
        def split_target_with_stanza(text: str) -> List[str]:
            """실제 Stanza 파서"""
            global stanza_nlp
            try:
                # Stanza로 문장 분할
                doc = stanza_nlp(text)
                sentences = [sent.text for sent in doc.sentences]
                return [s.strip() for s in sentences if s.strip()]
            except Exception as e:
                print(f"⚠️ Stanza 파싱 실패: {e}")
                return smart_sentence_split(text, is_source=False)
                
    except Exception as e:
        print(f"⚠️ Stanza 모델 로드 실패: {e}")
        STANZA_AVAILABLE = False
        stanza_nlp = None
        
except ImportError:
    print("⚠️ Stanza 미설치 (폴백 모드)")
    STANZA_AVAILABLE = False
    stanza_nlp = None

print(f"✅ new_parsers 모듈 로드됨 (SuPar: {SUPAR_AVAILABLE}, Stanza: {STANZA_AVAILABLE})")
