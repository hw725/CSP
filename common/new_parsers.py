# 구문분석기 통합 로더 (Kanbun 무조건 우선 → Stanza 폴백)
# Kanbun(SuPar-Kanbun / suparkanbun + esupar)를 최우선으로 사용하도록 강화합니다.
# Stanza는 보조/폴백 용도로만 사용됩니다.
import logging
import os
import subprocess
import sys
logger = logging.getLogger(__name__)

# 환경 변수로 강제 설정 가능: CSP_KANBUN_FORCE=1 이면 Kanbun 강제 우선
KANBUN_FORCE = os.environ.get("CSP_KANBUN_FORCE", "1") not in ["0", "false", "False"]

# 로드 상태(모듈 설치 여부)
KANBUN_AVAILABLE = False
STANZA_MODULE_AVAILABLE = False

# Kanbun (suparkanbun / esupar) 우선 로드
try:
    import suparkanbun as skb  # PyPI: suparkanbun
    import esupar as es
    KANBUN_AVAILABLE = True
    logger.info("✅ Kanbun 파서 로드됨 (suparkanbun/esupar)")
except Exception as e:
    logger.warning(f"⚠️ Kanbun 파서 로드 실패: {e}")

# Stanza 로드 (폴백)
try:
    import stanza
    STANZA_MODULE_AVAILABLE = True
    logger.info("✅ Stanza 로드됨 (폴백용)")
except Exception as e:
    logger.warning(f"⚠️ Stanza 로드 실패: {e}")

def get_kanbun_parser():
    """Kanbun 파서 인스턴스를 반환 (필요 시 초기화)"""
    if not KANBUN_AVAILABLE:
        raise RuntimeError("Kanbun 파서가 설치되어 있지 않습니다 (suparkanbun/esupar 필요)")
    # suparkanbun은 문장 분할/의존 구문분석을 래핑 제공합니다.
    # 여기서는 모듈 로드 성공만으로 OK, 실제 사용은 호출부에서 수행.
    return {"module": "suparkanbun", "version": getattr(skb, "__version__", "unknown")}

def load_kanbun_pipeline(BERT: str | None = None, Danku: bool | None = None):
    """
    suparkanbun.load()을 사용해 고전 중국어 파이프라인 생성
    - BERT 옵션: roberta-classical-chinese-base-char | roberta-classical-chinese-large-char | guwenbert-base | guwenbert-large | sikubert | sikuroberta
    - Danku: True 시 자동 문장 경계(段句) 시도
    기본값:
      BERT = 환경변수 CSP_KANBUN_BERT 또는 'sikubert'
      Danku = 환경변수 CSP_KANBUN_DANKU ("1"/"true") 또는 False
    """
    if not KANBUN_AVAILABLE:
        raise RuntimeError("Kanbun 파서가 설치되어 있지 않습니다")
    # 환경변수로 기본값 제공
    if BERT is None:
        BERT = os.environ.get("CSP_KANBUN_BERT", "sikubert")
    if Danku is None:
        ev = os.environ.get("CSP_KANBUN_DANKU", "0")
        Danku = ev.lower() in ("1", "true", "yes")
    try:
        nlp = skb.load(BERT=BERT, Danku=Danku)
        logger.info(f"✅ suparkanbun.load() 성공 (BERT={BERT}, Danku={Danku})")
        return nlp
    except Exception as e:
        logger.error(f"❌ suparkanbun.load() 실패: {e}")
        raise

def get_stanza_pipeline(lang: str = "zh"):
    """Stanza 파이프라인 반환 (폴백)"""
    if not STANZA_MODULE_AVAILABLE:
        raise RuntimeError("Stanza가 설치되지 않았습니다")
    try:
        stanza.download(lang)
    except Exception:
        pass
    return stanza.Pipeline(lang=lang)

def select_parsers(prefer_kanbun: bool = True):
    """사용할 파서를 선택 (Kanbun 우선, Stanza 폴백)"""
    use_kanbun = KANBUN_FORCE or prefer_kanbun
    if use_kanbun and KANBUN_AVAILABLE:
        logger.info("🔀 파서 선택: Kanbun 우선 사용")
        return {"kanbun": True, "stanza": STANZA_MODULE_AVAILABLE}
    # Kanbun 불가 또는 비우선인 경우
    if STANZA_MODULE_AVAILABLE:
        logger.info("🔀 파서 선택: Stanza 폴백 사용")
        return {"kanbun": False, "stanza": True}
    # 둘 다 없으면 에러
    raise RuntimeError("사용 가능한 파서를 찾을 수 없습니다 (Kanbun/Stanza 모두 없음)")

def log_current_configuration():
    logger.info(f"⚙️ KANBUN_FORCE={KANBUN_FORCE}, KANBUN_AVAILABLE={KANBUN_AVAILABLE}, STANZA_MODULE_AVAILABLE={STANZA_MODULE_AVAILABLE}")
    if KANBUN_AVAILABLE:
        logger.info("🧠 Kanbun: suparkanbun/esupar 활성")
    if STANZA_MODULE_AVAILABLE:
        logger.info("🧠 Stanza: zh 파이프라인 폴백 활성")

"""새로운 파서 모듈 - SuPar-Kanbun & Stanza 통합"""

import re
from typing import List, Optional

# 파서 가용성 플래그(실제 파이프라인 준비 여부)
SUPAR_AVAILABLE = False
STANZA_AVAILABLE = False

# 전역 파서 변수들
supar_parser = None
stanza_nlp = None

def smart_sentence_split(text: str, is_source: bool = True) -> List[str]:
    """스마트 문장 분할 함수 (폴백 구현) - 인용 표지 병합 지원"""
    if is_source:
        # 한문 원문 분할
        pattern = r'(?<=[。？！])\s*|(?<=[.!?])\s+'
    else:
        # 번역문 분할
        pattern = r'(?<=[.!?])\s*|(?<=다)\s*|(?<=요)\s*'
    
    sentences = re.split(pattern, text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 인용 표지 병합 처리 (번역문 전용)
    if not is_source:
        sentences = merge_quotation_markers(sentences)
    
    return sentences

def merge_quotation_markers(sentences: List[str]) -> List[str]:
    """인용 표지를 이전 인용문과 병합 - 중첩 인용 지원
    
    인용 표지 구조: [인용조사] + [동사어간] + [어미]
    예: "고 하였다", "라고 말한다", "하고 명하셨다" 등
    중첩 예: [문장1] + [인용표지1] + [인용표지2] → 모두 병합
    """
    if len(sentences) <= 1:
        return sentences
    
    # 인용 조사 패턴 (빈번한 단독 '고' 포함)
    quotation_particles = r'(고|[이]?라?고|하고|며|면서)'
    
    # 발화 동사 어간 (인용 표지에 자주 쓰이는 동사들)
    speech_verbs = r'(하|말하|말씀하|명하|이르|대답하|답하|묻|문|여쭙|아뢰|전하|칭하|부르|외치)'
    
    # 존칭+시제 통합 패턴 (축약형 포함)
    # "시었다" → "셨다", "시었" → "셨", "시어" → "셔" 등
    honorific_tense = r'(?:셨|ㅆ|시었|시어|시는|시ㄴ|시ㄹ|시|었|았|였|는|ㄴ|ㄹ|을)?'
    
    # 종결 어미
    endings = r'(다|ㄴ다|는다|습니다|ㅂ니다|까|ㄹ까|을까|느냐|ㄴ가|는가|라|거라|소|오|어라|아라|니|으니)'
    
    # 따옴표(닫는 따옴표)와 종결 부호 (선택적)
    closing_quote = r'["”’]?'
    punctuation = r'[\.。?!,，]?'

    # 마커 1개 조각: (선행 닫는 따옴표) + 인용조사 + 동사(시제/존칭) + 종결어미 + (부호/닫는 따옴표)
    marker_chunk = (
        closing_quote +
        r'\s*' +
        quotation_particles +
        r'\s+' +
        speech_verbs +
        honorific_tense +
        endings +
        r'\s*' +
        punctuation +
        r'\s*' +
        closing_quote +
        r'\s*'
    )

    # 전체 패턴: 마커 조각이 1회 이상 연쇄된 문장 전체
    quotation_marker_pattern = r'^\s*(?:' + marker_chunk + r')+$'
    
    # 반복 병합 (중첩 인용 처리)
    changed = True
    while changed:
        changed = False
        merged = []
        i = 0
        
        while i < len(sentences):
            current = sentences[i]
            
            # 연속된 인용 표지들을 모두 병합
            accumulated_markers = []
            j = i + 1
            while j < len(sentences):
                next_sent = sentences[j]
                if re.match(quotation_marker_pattern, next_sent, re.IGNORECASE):
                    accumulated_markers.append(next_sent)
                    j += 1
                    changed = True
                else:
                    break
            
            # 병합
            if accumulated_markers:
                merged_text = current + ' ' + ' '.join(accumulated_markers)
                merged.append(merged_text)
                i = j  # 병합된 만큼 건너뛰기
            else:
                merged.append(current)
                i += 1
        
        sentences = merged
    
    return sentences

def split_source_with_supar(text: str) -> List[str]:
    """SuPar-Kanbun 파서 (폴백)"""
    return smart_sentence_split(text, is_source=True)

def split_target_with_stanza(text: str) -> List[str]:
    """Stanza 파서 (폴백)"""
    return smart_sentence_split(text, is_source=False)

def fallback_split_by_punctuation(text: str, is_source: bool = True) -> List[str]:
    """구두점 기반 폴백 분할"""
    return smart_sentence_split(text, is_source)

"""SuPar-Kanbun(suparkanbun/esupar) & Stanza는 무겁기 때문에 import 시점에 자동 초기화하지 않고,
실제 사용 시점에 lazy-init 합니다.

- SUPAR_AVAILABLE / STANZA_AVAILABLE: '파이프라인이 실제로 준비됨'을 의미
- KANBUN_AVAILABLE / STANZA_MODULE_AVAILABLE: '패키지 import 가능'을 의미
"""

def _get_stanza_model_dir() -> str:
    # Docker에서는 /opt/stanza_resources를 권장(빌드 타임에 미리 다운로드 가능)
    env_dir = os.environ.get("CSP_STANZA_DIR")
    if env_dir:
        return env_dir
    if os.path.isdir("/opt/stanza_resources"):
        return "/opt/stanza_resources"
    return os.path.join(os.path.expanduser("~"), "stanza_resources")


def ensure_kanbun_pipeline(BERT: str | None = None, Danku: bool | None = None):
    """suparkanbun 파이프라인을 1회만 초기화하고 반환합니다."""
    global supar_parser, SUPAR_AVAILABLE
    if supar_parser is not None:
        return supar_parser

    if not KANBUN_AVAILABLE:
        # 요구사항: 미설치면 즉시 설치 시도
        try:
            logger.warning("⚠️ Kanbun 파서 미설치: suparkanbun/esupar 설치를 시도합니다...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--no-cache-dir", "suparkanbun", "esupar"]
            )
            import importlib
            importlib.invalidate_caches()
            globals()["skb"] = importlib.import_module("suparkanbun")
            globals()["es"] = importlib.import_module("esupar")
            globals()["KANBUN_AVAILABLE"] = True
            logger.info("✅ Kanbun 파서 패키지 설치/임포트 성공")
        except Exception as e:
            raise RuntimeError(
                "Kanbun 파서가 설치되어 있지 않습니다 (suparkanbun/esupar 필요). "
                f"자동 설치도 실패했습니다: {e}"
            )

    supar_parser = load_kanbun_pipeline(BERT=BERT, Danku=Danku)
    SUPAR_AVAILABLE = True
    logger.info("✅ Kanbun(suparkanbun) 파이프라인 초기화 완료")
    return supar_parser


def ensure_stanza_pipeline(lang: str = "ko"):
    """Stanza 파이프라인을 1회만 초기화하고 반환합니다."""
    global stanza_nlp, STANZA_AVAILABLE
    if stanza_nlp is not None:
        return stanza_nlp

    if not STANZA_MODULE_AVAILABLE:
        # 요구사항: 미설치면 즉시 설치 시도
        try:
            logger.warning("⚠️ Stanza 미설치: stanza 설치를 시도합니다...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "stanza"])
            import importlib
            importlib.invalidate_caches()
            globals()["stanza"] = importlib.import_module("stanza")
            globals()["STANZA_MODULE_AVAILABLE"] = True
            logger.info("✅ Stanza 패키지 설치/임포트 성공")
        except Exception as e:
            raise RuntimeError(f"Stanza가 설치되지 않았습니다. 자동 설치도 실패했습니다: {e}")

    model_dir = _get_stanza_model_dir()
    os.makedirs(model_dir, exist_ok=True)

    # 리소스가 없으면 Pipeline 생성이 실패할 수 있으므로, 여기서는 다운로드를 강제하지 않습니다.
    # (Docker 빌드 시 stanza.download로 미리 받아두는 것을 권장)
    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except Exception:
        use_gpu = False

    def _try_init() -> "stanza.Pipeline":
        return stanza.Pipeline(lang, model_dir=model_dir, use_gpu=use_gpu, verbose=False)

    try:
        stanza_nlp = _try_init()
        STANZA_AVAILABLE = True
        logger.info(f"✅ Stanza 파이프라인 초기화 완료 (lang={lang}, GPU={use_gpu}, dir={model_dir})")
        return stanza_nlp
    except Exception as e:
        # 리소스가 없으면 다운로드 후 1회 재시도
        try:
            logger.warning(f"⚠️ Stanza 파이프라인 초기화 실패: {e} → 리소스 다운로드 후 재시도")
            stanza.download(lang, model_dir=model_dir, verbose=False)
            stanza_nlp = _try_init()
            STANZA_AVAILABLE = True
            logger.info(f"✅ Stanza 파이프라인 초기화 완료(재시도) (lang={lang}, GPU={use_gpu}, dir={model_dir})")
            return stanza_nlp
        except Exception as e2:
            STANZA_AVAILABLE = False
            stanza_nlp = None
            raise RuntimeError(f"Stanza 파이프라인 초기화 실패(다운로드/재시도 포함): {e2}")


def split_source_with_supar(text: str) -> List[str]:
    """suparkanbun 파이프라인을 사용한 한문 문장 분할(실패 시 휴리스틱 폴백)."""
    try:
        nlp = ensure_kanbun_pipeline()
        if hasattr(nlp, 'Danku') and getattr(nlp, 'Danku'):
            doc = nlp(text)
            sentences: List[str] = []
            if hasattr(doc, 'sentences') and doc.sentences:
                for s in doc.sentences:
                    if hasattr(s, 'values') and s.values:
                        sent_text = ''.join([getattr(tok, 'form', '') for tok in s.values])
                        if sent_text.strip():
                            sentences.append(sent_text.strip())
                    elif hasattr(s, 'text'):
                        t = getattr(s, 'text')
                        if isinstance(t, str) and t.strip():
                            sentences.append(t.strip())
            if sentences:
                return sentences
        return smart_sentence_split(text, is_source=True)
    except Exception as e:
        logger.warning(f"⚠️ suparkanbun 분할 실패(휴리스틱 폴백): {e}")
        return smart_sentence_split(text, is_source=True)


def split_target_with_stanza(text: str) -> List[str]:
    """Stanza 기반 한국어 문장 분할(실패 시 휴리스틱 폴백) + 인용 표지 병합."""
    try:
        nlp = ensure_stanza_pipeline(lang="ko")
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sentences]
        sentences = [s.strip() for s in sentences if s.strip()]
        return merge_quotation_markers(sentences)
    except Exception as e:
        logger.warning(f"⚠️ Stanza 문장 분할 실패(휴리스틱 폴백): {e}")
        return smart_sentence_split(text, is_source=False)


logger.info(
    "✅ new_parsers import OK "
    f"(Kanbun module={KANBUN_AVAILABLE}, Stanza module={STANZA_MODULE_AVAILABLE}, "
    f"pipelines created(lazy init): supar={SUPAR_AVAILABLE}, stanza={STANZA_AVAILABLE})"
)

# === Korean clause boundary detection (comma) ===
def get_korean_clause_boundary_commas(text: str, mode: str = 'soft') -> list[int]:
    """텍스트 내에서 절/구 경계로 볼만한 콤마(,)의 문자 오프셋 목록을 반환합니다.

    mode:
      - 'soft': 단순 나열(와/과/및/그리고/또 등) 패턴으로 보이는 콤마는 제외
      - 'strict': 모든 콤마를 경계 후보로 간주
      - 기타: 'soft'와 동일 처리
    """
    if not text:
        return []

    # 모든 콤마 위치 수집
    comma_positions = [i for i, ch in enumerate(text) if ch == ',']
    if not comma_positions:
        return []

    if mode == 'strict':
        return comma_positions

    # soft 모드: 나열 패턴 제외 (간단 휴리스틱)
    # - 연결어: 와/과/및/그리고/또/혹은/내지
    # - 수사적 나열: 첫째/둘째/셋째 ... 직후 콤마
    conjunctions = ["와", "과", "및", "그리고", "또", "혹은", "내지"]
    ordinal_markers = ["첫째", "둘째", "셋째", "넷째", "다섯째"]

    def looks_like_enumeration(left_ctx: str, right_ctx: str) -> bool:
        # 연결어가 바로 인접하거나 근접(<= 5자)한 경우
        for cj in conjunctions:
            if left_ctx.endswith(cj) or cj in left_ctx[-5:]:
                return True
            if right_ctx.startswith(cj) or cj in right_ctx[:5]:
                return True
        # 수사 마커 바로 앞
        for od in ordinal_markers:
            if left_ctx.endswith(od):
                return True
        # 간단 명사열 추정: 조사(은/는/이/가/을/를/와/과) 바로 앞/뒤 콤마
        for josa in ["은","는","이","가","을","를","와","과"]:
            if left_ctx.endswith(josa) or right_ctx.startswith(josa):
                return True
        return False

    boundaries = []
    for pos in comma_positions:
        left_ctx = text[max(0, pos-12):pos]
        right_ctx = text[pos+1:pos+1+12]
        if not looks_like_enumeration(left_ctx, right_ctx):
            boundaries.append(pos)
    return boundaries

# ===== 고급: Stanza 기반 한국어 절/구 경계 추출 =====
_KO_CLAUSE_CACHE: dict[tuple[str, str], list[int]] = {}

def get_korean_clause_boundaries_stanza(text: str, mode: str = 'default') -> list[int]:
    """Stanza 한국어 파서를 사용해 절/구 경계 후보의 문자 오프셋 목록을 반환합니다.

    반환되는 위치는 경계 직전 문자 위치(end_char - 1)로 정규화합니다.

    mode는 현재 동작에 큰 영향이 없고, 향후 확장을 대비한 자리입니다.
    """
    key = (text, mode)
    cached = _KO_CLAUSE_CACHE.get(key)
    if cached is not None:
        return cached

    if not text or not STANZA_AVAILABLE or stanza_nlp is None:
        # 폴백: 콤마 기반만 사용
        res = get_korean_clause_boundary_commas(text, mode='soft')
        _KO_CLAUSE_CACHE[key] = res
        return res

    try:
        doc = stanza_nlp(text)
        offsets: list[int] = []
        for sent in doc.sentences:
            for token in sent.tokens:
                # token.words[0]로 UD 속성 접근
                w = token.words[0] if token.words else None
                upos = w.upos if w else None
                deprel = w.deprel if w else None
                txt = token.text
                # 1) 구두점 기반(쉼표/세미콜론)
                if upos == 'PUNCT' and txt in {',', ';', '、', '，', '；'}:
                    if token.end_char is not None:
                        offsets.append(token.end_char - 1)
                    continue
                # 2) 절/접속 표지 (SCONJ/CCONJ) 흔적
                if upos in {'SCONJ', 'CCONJ'} or (deprel in {'mark', 'cc', 'conj', 'advcl', 'parataxis'}):
                    if token.end_char is not None:
                        offsets.append(token.end_char - 1)
        # 정렬·중복 제거
        res = sorted(set(offsets))
        _KO_CLAUSE_CACHE[key] = res
        return res
    except Exception:
        # 실패 시 폴백
        res = get_korean_clause_boundary_commas(text, mode='soft')
        _KO_CLAUSE_CACHE[key] = res
        return res

# 강도 포함: offset -> strength 계수(1.0~1.6 권장)
_KO_CLAUSE_STRENGTH_CACHE: dict[str, dict[int, float]] = {}

def get_korean_clause_offsets_with_strength(text: str) -> dict[int, float]:
    """Stanza 기반으로 절/구 경계 오프셋과 강도를 반환합니다.
    - PUNCT(쉼표/세미콜론 등): 1.0
    - CCONJ/cc: 1.1~1.2
    - SCONJ/mark/advcl/parataxis: 1.3~1.5
    미가용/실패 시, 콤마 휴리스틱으로 1.0 강도로 대체합니다.
    """
    if not text:
        return {}
    cached = _KO_CLAUSE_STRENGTH_CACHE.get(text)
    if cached is not None:
        return cached
    if not STANZA_AVAILABLE or stanza_nlp is None:
        # 휴리스틱 콤마만 1.0 강도로
        offs = get_korean_clause_boundary_commas(text, mode='soft')
        res = {o: 1.0 for o in offs}
        _KO_CLAUSE_STRENGTH_CACHE[text] = res
        return res
    try:
        doc = stanza_nlp(text)
        factors: dict[int, float] = {}
        for sent in doc.sentences:
            for token in sent.tokens:
                w = token.words[0] if token.words else None
                upos = w.upos if w else None
                deprel = w.deprel if w else None
                t = token.text
                end = token.end_char if token.end_char is not None else None
                if end is None:
                    continue
                # 기본 강도
                strength = None
                if upos == 'PUNCT' and t in {',',';','、','，','；'}:
                    strength = 1.0
                elif upos == 'CCONJ' or deprel in {'cc', 'conj'}:
                    strength = 1.15
                elif upos == 'SCONJ' or deprel in {'mark', 'advcl', 'parataxis'}:
                    strength = 1.35
                # 한국어 구어체 접속미사 추정(간단): '는데','ㄴ데','지만','으나','더니' 등 토큰 텍스트로 보정
                if strength is None and t and any(s in t for s in ['는데','ㄴ데','지만','으나','더니','으면서','면서']):
                    strength = 1.3
                if strength is not None:
                    factors[end - 1] = max(factors.get(end - 1, 1.0), strength)
        res = factors
        _KO_CLAUSE_STRENGTH_CACHE[text] = res
        return res
    except Exception:
        offs = get_korean_clause_boundary_commas(text, mode='soft')
        res = {o: 1.0 for o in offs}
        _KO_CLAUSE_STRENGTH_CACHE[text] = res
        return res

# ===== 중국어(SuPar 또는 단순 구두점) 기반 원문 단위 경계 =====
def get_chinese_unit_boundary_indices(source_units: list[str]) -> set[int]:
    """원문 단위 사이 경계(단위 index j: 단위 j-1과 j 사이)를 반환합니다.

    현재는 단위가 구두점(。！？；、，,.;:)로 끝날 때 경계로 간주합니다.
    SuPar 사용 시 보강 가능 (향후 확장 포인트).
    """
    if not source_units:
        return set()
    punct_ends = set(list('。！？；、，,.;:'))
    boundaries: set[int] = set()
    # j는 1..len(units)-1 사이 인덱스 경계
    for j in range(1, len(source_units)):
        prev = source_units[j-1]
        if prev and prev[-1] in punct_ends:
            boundaries.add(j)
    return boundaries

def get_chinese_unit_boundary_indices_supar(source_text: str, source_units: list[str]) -> set[int]:
    """SuPar-Kanbun 기반(가능하면)으로 원문 단위 경계를 추정합니다.
    실패/미설치 시 get_chinese_unit_boundary_indices로 폴백합니다.
    현재 구현은 안전 폴백 중심이며, SuPar 토큰/문장 경계를 문자열에 맞춰 매핑하는 고급 매칭은 향후 확장 포인트입니다.
    """
    try:
        if not source_units:
            return set()
        if not SUPAR_AVAILABLE or supar_parser is None:
            return get_chinese_unit_boundary_indices(source_units)
        # 간단 접근: 이전 단위가 문장 구두점으로 끝나거나, SuPar 문장 경계가 단위 사이에 있을 때 경계로 표기
        text = source_text or ''.join(source_units)
        parsed = supar_parser.predict(text, prob=False, verbose=False)
        # parsed로부터 문장 텍스트 목록 복원 시도(실패 가능성 고려)
        sentences = []
        try:
            for s in getattr(parsed, 'sentences', []) or []:
                if hasattr(s, 'values') and s.values:
                    sent_text = ''.join([tok.form for tok in s.values if hasattr(tok, 'form')])
                    if sent_text:
                        sentences.append(sent_text)
        except Exception:
            sentences = []
        # 단위 경계의 문자 오프셋 계산(공백 제거 연결)
        unit_offsets = []
        offset = 0
        for u in source_units:
            offset += len(u)
            unit_offsets.append(offset)  # 경계 직후 오프셋
        # SuPar 문장 누적 길이로 경계 후보 생성
        supar_offsets = set()
        if sentences:
            acc = 0
            for s in sentences:
                acc += len(s)
                supar_offsets.add(acc)
        # 경계 판단: 단위 경계 오프셋이 SuPar 경계와 근접(±1)하거나, 이전 단위가 구두점으로 끝나면 경계
        punct_ends = set(list('。！？；、，,.;:'))
        boundaries: set[int] = set()
        for j, off in enumerate(unit_offsets, start=1):
            if j >= len(source_units):
                break
            prev = source_units[j-1]
            if prev and prev[-1] in punct_ends:
                boundaries.add(j)
                continue
            if off in supar_offsets or (off-1) in supar_offsets or (off+1) in supar_offsets:
                boundaries.add(j)
        return boundaries
    except Exception:
        return get_chinese_unit_boundary_indices(source_units)
