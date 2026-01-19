"""PA 문장 분할기 - SuPar-Kanbun & Stanza 기반 (spaCy 대체)"""
from typing import Any, Dict, List, Tuple
try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False
import os
import json
import hashlib
import numpy as np
from pathlib import Path
import re
import sys

try:
    from common.llm_boundary_refiner import refine_boundaries_with_llm
except Exception:
    refine_boundaries_with_llm = None

# BGE Embedder import (의미 기반 경계 감지용)
# NOTE: /workspace/common 을 sys.path 최상단에 넣으면, HF의 `tokenizers` 패키지보다
# 로컬 `common/tokenizers`가 우선되어 transformers 내부 import가 깨질 수 있다.
# (Docker에서 PretrainedConfig circular import로 표면화)
try:
    from common.embedders import get_embedding_manager
    BGE_AVAILABLE = True
except Exception:
    BGE_AVAILABLE = False

# OpenAI wrapper 클래스 (SA와 동일)
class OpenAIWrapper:
    """OpenAI API 래퍼 - SA 시스템과 동일한 방식"""
    
    def __init__(self, api_key=None, model="text-embedding-3-large"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.cache_dir = Path("embeddings_cache_openai")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "openai_embeddings.json"
        self._embedding_cache = {}
        self._load_cache()
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다")
    
    def _load_cache(self):
        """캐시 파일에서 임베딩 로드"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    self._embedding_cache = {k: np.array(v) for k, v in cache_data.items()}
                print(f"📂 OpenAI 캐시 로드: {len(self._embedding_cache)}개 항목")
            except Exception as e:
                print(f"⚠️ 캐시 로드 실패: {e}")
                self._embedding_cache = {}
    
    def _save_cache(self):
        """임베딩을 캐시 파일에 저장"""
        try:
            cache_data = {k: v.tolist() for k, v in self._embedding_cache.items()}
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False)
            print(f"💾 OpenAI 캐시 저장: {len(self._embedding_cache)}개 항목")
        except Exception as e:
            print(f"⚠️ 캐시 저장 실패: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """텍스트에 대한 캐시 키 생성"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def compute_embeddings_with_cache(self, texts, use_cache=True):
        """캐시를 사용한 OpenAI 임베딩 생성"""
        
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
                cache_key = self._get_cache_key(text)
                if cache_key in self._embedding_cache:
                    cached_embeddings[i] = self._embedding_cache[cache_key]
                else:
                    missing_texts.append(text)
                    missing_indices.append(i)
        else:
            missing_texts = texts
            missing_indices = list(range(len(texts)))
        
        # 캐시 히트 로그
        if use_cache and cached_embeddings:
            print(f"📂 캐시 히트: {len(cached_embeddings)}개, 누락: {len(missing_texts)}개")
        
        # 누락된 텍스트들 API 호출
        new_embeddings = {}
        if missing_texts:
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                
                print(f"🔄 OpenAI API 호출: {len(missing_texts)}개 텍스트")
                
                response = client.embeddings.create(
                    model=self.model,
                    input=missing_texts,
                    encoding_format="float"
                )
                
                batch_embeddings = [np.array(item.embedding) for item in response.data]
                
                for i, (idx, embedding) in enumerate(zip(missing_indices, batch_embeddings)):
                    new_embeddings[idx] = embedding
                    
                    # 캐시에 저장
                    if use_cache:
                        cache_key = self._get_cache_key(missing_texts[i])
                        self._embedding_cache[cache_key] = embedding
                
                # 캐시 파일 저장
                if use_cache and new_embeddings:
                    self._save_cache()
                    
                print(f"✅ OpenAI 임베딩 생성: {len(batch_embeddings)}개 → 차원: {len(batch_embeddings[0])}")
                
            except Exception as e:
                print(f"❌ OpenAI API 호출 실패: {e}")
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
        
        if return_single:
            return all_embeddings[0]
        else:
            return all_embeddings

# SuPar-Kanbun과 Stanza 사용 (spaCy 대체)
import re
import regex
try:
    import sys
    import os
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(current_dir)  # CSP 디렉토리
    sys.path.insert(0, project_root)
    
    from common.new_parsers import (
        smart_sentence_split,
        split_source_with_supar, 
        split_target_with_stanza,
        fallback_split_by_punctuation,
        SUPAR_AVAILABLE,
        STANZA_AVAILABLE,
        KANBUN_AVAILABLE,
        STANZA_MODULE_AVAILABLE
    )
    print(
        "✅ PA: 새 파서 모듈 임포트 완료 "
        f"(Kanbun module: {KANBUN_AVAILABLE}, Stanza module: {STANZA_MODULE_AVAILABLE}, "
        f"pipelines created(lazy init): supar={SUPAR_AVAILABLE}, stanza={STANZA_AVAILABLE})"
    )
except Exception as e:
    print(f"⚠️ PA: 새 파서 로드 실패, 폴백 모드: {e}")
    SUPAR_AVAILABLE = False
    STANZA_AVAILABLE = False
    
    def smart_sentence_split(text: str, is_source: bool = True) -> List[str]:
        """폴백: 정규식 기반 분할"""
        pattern = r'(?<=[。？！○])\s*|(?<=[.!?])\s+'
        sentences = re.split(pattern, text.strip())
        return [s.strip() for s in sentences if s.strip()]

# 하이브리드 토크나이저 추가
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    # 중국어: SikuBERT (기존 유지)
    from common.tokenizers import (
        get_siku_tokenizer,
        siku_get_embeddings,
        siku_similarity
    )
    
    # 🆕 한국어: 하이브리드 토크나이저 (RoBERTa-Hanja + Kiwipiepy)
    from common.tokenizers import (
        get_hybrid_korean_tokenizer,
        hybrid_tokenize_korean,
        get_roberta_hanja_tokenizer,
        get_kiwi_tokenizer
    )
    
    # 토크나이저 초기화
    siku_tokenizer = get_siku_tokenizer()
    
    # 🆕 하이브리드 한국어 토크나이저 초기화
    hybrid_korean_tokenizer = get_hybrid_korean_tokenizer()
    
    # 로그는 p2s/main.py에서만 출력한다(여기서 중복 출력 금지).
    
except Exception as e:
    siku_tokenizer = None
    hybrid_korean_tokenizer = None
    print(f"⚠️ PA: 하이브리드 토크나이저 초기화 실패: {e}")

def analyze_script_segments(text: str) -> List[Tuple[str, str]]:
    """텍스트를 스크립트별로 분석"""
    segments = []
    current_segment = ""
    current_script = None
    
    for char in text:
        if regex.match(r'\p{Han}', char):  # 한자
            script = 'han'
        elif regex.match(r'\p{Hangul}', char):  # 한글
            script = 'hangul'
        else:  # 기타 (공백, 구두점 등)
            script = 'other'
        
        if current_script != script:
            if current_segment:
                segments.append((current_segment, current_script))
            current_segment = char
            current_script = script
        else:
            current_segment += char
    
    if current_segment:
        segments.append((current_segment, current_script))
    
    return segments

def preprocess_with_hybrid_tokenization(text: str) -> str:
    """
    ⚠️ 이 함수는 spaCy 전용 전처리입니다. 
    원문 분할에는 절대 사용하지 마세요! 어절 경계만 사용!
    - 중국어: SikuBERT (기존 유지)
    - 한국어: RoBERTa-Hanja (한자) + Kiwipiepy (한글)
    """
    # 원문 분할에는 사용 금지! 번역문 spaCy 전처리 전용!
    if not hybrid_korean_tokenizer or not siku_tokenizer:
        return preprocess_with_siku_tokenization_fallback(text)
    
    try:
        # 🆕 하이브리드 한국어 토큰화 (번역문 분석 전용!)
        if regex.search(r'\p{Hangul}', text):  # 한글이 포함된 경우
            korean_result = hybrid_korean_tokenizer.tokenize_korean_text(text, text_type="translation")
            
            # 하이브리드 결과를 spaCy가 이해할 수 있는 형태로 변환
            processed_tokens = []
            for segment in korean_result['segments']:
                if segment['type'] == 'hanja':
                    # 한자 부분: RoBERTa 결과 사용
                    processed_tokens.extend(segment['tokens'])
                elif segment['type'] == 'hangul':
                    # 한글 부분: Kiwi 결과 사용
                    processed_tokens.extend(segment['tokens'])
            
            if processed_tokens:
                return ' '.join(processed_tokens)
        
        # 중국어 처리 (기존 SikuBERT 유지, 번역문 분석 전용!)
        if regex.search(r'\p{Han}', text) and not regex.search(r'\p{Hangul}', text):
            segments = analyze_script_segments(text)
            processed_parts = []
            
            for segment_text, script_type in segments:
                if script_type == 'han':  # 한자 부분은 SikuBERT로 토크나이징
                    tokens = siku_tokenizer.tokenize_chinese(segment_text)
                    processed_parts.append(' '.join(tokens))
                else:
                    processed_parts.append(segment_text)
            
            return ' '.join(processed_parts)
        
        # 폴백: 원본 텍스트 반환
        return text
        
    except Exception as e:
        print(f"⚠️ 하이브리드 토크나이징 실패: {e}")
        return preprocess_with_siku_tokenization_fallback(text)

def preprocess_with_siku_tokenization_fallback(text: str) -> str:
    """SikuBERT 폴백: 한자 부분만 사전 토크나이징"""
    if not siku_tokenizer:
        return text
    
    segments = analyze_script_segments(text)
    processed_segments = []
    
    han_segments = []
    han_indices = []
    
    # 한자 부분만 추출
    for i, (segment, script) in enumerate(segments):
        if script == 'han' and len(segment.strip()) > 0:
            han_segments.append(segment)
            han_indices.append(i)
    
    # 배치로 SikuBERT 처리
    if han_segments:
        try:
            tokenized_han = []
            for han_text in han_segments:
                tokens = siku_tokenizer.tokenize_chinese(han_text)
                tokenized_han.append(' '.join(tokens))
            han_dict = dict(zip(han_indices, tokenized_han))
        except Exception as e:
            print(f"⚠️ SikuBERT 토크나이징 실패: {e}")
            han_dict = {}
    else:
        han_dict = {}
    
    # 결과 재조립
    for i, (segment, script) in enumerate(segments):
        if i in han_dict:
            # SikuBERT로 토크나이징된 한자 부분
            processed_segments.append(' '.join(han_dict[i]))
        else:
            # 원본 유지
            processed_segments.append(segment)
    
    return ''.join(processed_segments)

# 기존 함수명과의 호환성
def preprocess_with_siku_tokenization(text: str) -> str:
    """호환성을 위한 래퍼 - 하이브리드 토크나이저 우선 사용"""
    return preprocess_with_hybrid_tokenization(text)

def split_target_sentences_advanced(text: str, max_length: int = 200, splitter: str = "punctuation", use_siku_preprocessing: bool = True) -> List[str]:
    """
    번역문 분할 - 종결 구두점 우선, 구두점 없으면 의미 기반 경계 감지
    
    ⭐️ 개선: 의미 기반 경계 감지로 구두점 없는 텍스트도 분할 가능
    ⭐️ 인용 표지 병합 지원
    """
    # 1) 구두점 기반 분할 시도
    strong_end_pattern = r'(?<=[。！？.!?])\s+'
    sentences = re.split(strong_end_pattern, text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 구두점이 없어서 분할 실패한 경우 의미 기반 경계 감지 (PA: 문장 단위)
    if len(sentences) == 1 and len(text) > 100 and BGE_AVAILABLE:
        try:
            offsets = detect_semantic_boundaries(text, window_size=80, threshold=0.75, min_segment_length=30)
            if len(offsets) > 1:
                sentences = [text[start:end].strip() for start, end in offsets if start < end]
                sentences = [s for s in sentences if s]
                print(f"✅ 의미 기반 경계 감지: {len(sentences)}개 문장")
        except Exception as e:
            print(f"⚠️ 의미 기반 경계 감지 실패: {e}")
    
    # 2) SikuBERT 전처리 (필요 시)
    if use_siku_preprocessing and contains_chinese(text):
        text = preprocess_with_siku_tokenization(text)

    # 3) 종결 구두점 기준 1차 분할 (강한 종결만)
    strong_end_pattern = r'(?<=[。！？.!?])\s+'  # 중국어/영문 종결부호
    sentences = re.split(strong_end_pattern, text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 인용 표지 병합
    sentences = merge_quotation_markers_in_list(sentences)

    # 🗣️ 화자+발화 병합: 구두점으로 잘린 "화자(발화동사)" + "발화내용"을 결합
    # - 예: "공자가 말했다." + "…" → 하나로 병합
    # - 번역문 분할은 rule-based가 정답이라는 전제 하에, 과도한 병합을 피하려고 매우 보수적으로 적용한다.
    def merge_speaker_utterance_pairs(segs: List[str]) -> List[str]:
        if len(segs) <= 1:
            return segs

        # 화자/발화 도입부로 끝나는 짧은 세그먼트 패턴
        # (구두점으로 문장 분할이 발생한 케이스만 합치기 위함)
        speaker_end = re.compile(
            r"(?:"  # 끝이 이런 형태면 다음 세그먼트가 발화 내용일 가능성이 높음
            r"(?:말(?:했|하였)다|말(?:하|했)다|말하였다|이르(?:렀)?다|이르되|대답하였다|답하였다|묻(?:었)?다|문(?:었)?다)"  # 한국어 발화동사
            r"|(?:曰|云|言曰|問曰|答曰)"  # 한문 발화 표지
            r")"  # 동사/표지
            r"\s*[.。!?]?$"  # 종결부호(있을 수도/없을 수도)
        )

        opening_quote = re.compile(r"^[\s\"'“”‘’「『《〈【\(\[]")

        merged: List[str] = []
        i = 0
        while i < len(segs):
            cur = segs[i].strip()
            if not cur:
                i += 1
                continue

            if i < len(segs) - 1:
                nxt = segs[i + 1].strip()
                if nxt and len(cur) <= 60 and speaker_end.search(cur):
                    # 다음 세그가 따옴표/괄호 시작이거나, 비교적 긴 발화내용이면 병합
                    if opening_quote.match(nxt) or len(nxt) >= 20:
                        merged.append((cur.rstrip() + " " + nxt.lstrip()).strip())
                        i += 2
                        continue

            merged.append(cur)
            i += 1

        return merged

    sentences = merge_speaker_utterance_pairs(sentences)

    # 📌 제목형 접두 분리: gold가 '...차자(箚子)' / '...장(狀)' 등을 단독 문장으로 두는 케이스 대응
    # - 예: "...차자(箚子) 충성...이다." -> "...차자(箚子)" + "충성...이다."
    def split_title_like_prefixes(segs: List[str]) -> List[str]:
        if not segs:
            return segs

        # 너무 공격적인 분할을 피하기 위해 괄호 포함 표지만 대상으로 한다.
        title_pat = re.compile(
            r"^(?P<prefix>.{0,120}?(?:차자\(箚子\)|장\(狀\)|표\(表\)|서\(書\)|론\(論\)|기\(記\)|설\(說\)))\s+(?P<rest>.+)$"
        )

        out: List[str] = []
        for s in segs:
            s = s.strip()
            if not s:
                continue
            m = title_pat.match(s)
            if not m:
                out.append(s)
                continue

            prefix = m.group("prefix").strip()
            rest = m.group("rest").strip()

            # prefix가 너무 짧거나, rest가 사실상 없으면 분할하지 않는다.
            if len(prefix) < 12 or len(rest) < 8:
                out.append(s)
                continue

            out.append(prefix)
            out.append(rest)

        return out

    sentences = split_title_like_prefixes(sentences)

    # 📌 〈...〉 열거 블록 접두 분리: 서술부 + 열거부를 gold가 분리하는 케이스 대응
    # - 예: "...등을 더하여 〈길례...〉 ...이다." -> "...등을 더하여" + "〈길례...〉 ...이다."
    def split_angle_bracket_list_prefixes(segs: List[str]) -> List[str]:
        if not segs:
            return segs

        # 너무 공격적인 분할을 피하기 위해 트리거(더하여/더해/추가하여) + 〈 로만 동작시킨다.
        trigger_end = re.compile(r"(?:더하여|더해|추가하여|보태어)\s*$")

        out: List[str] = []
        for s in segs:
            s = s.strip()
            if not s:
                continue

            idx = s.find("〈")
            if idx <= 0:
                out.append(s)
                continue

            left = s[:idx].rstrip()
            right = s[idx:].lstrip()
            if len(left) < 20 or len(right) < 30:
                out.append(s)
                continue

            # left가 트리거로 끝나지 않으면 분리하지 않는다.
            if not trigger_end.search(left):
                out.append(s)
                continue

            out.append(left)
            out.append(right)

        return out

    sentences = split_angle_bracket_list_prefixes(sentences)

    # 🗣️ 인용 연속 병합: 닫는 따옴표 뒤 "라/고/라고 ..."가 이어지는 경우, 종결부호 분할을 되돌린다.
    # - 예: "...못 된다. ”" + "라 했을 것입니다." -> 하나로 병합
    # - 예: "...없다. ”" + "라 하니, ..." / "고 하니, ..." / "라고 하니, ..." -> 하나로 병합
    def merge_quote_ra_continuations(segs: List[str]) -> List[str]:
        if len(segs) <= 1:
            return segs

        closers = set([
            '"', "'",
            "\u201D", "\u2019",  # ” ’
            "\u300D", "\u300F",  # 」 』
            "\u300B", "\u3009",  # 》 〉
        ])

        # 일부 전처리/토크나이징 경로에서 따옴표가 제거될 수 있어,
        # 닫는따옴표가 아니라 종결부호로 끝난 경우도 병합 트리거로 허용한다.
        end_punct = set([".", "!", "?", "。", "！", "？"])

        # 다음 세그가 '라/고/라고 ...'로 시작하면, 앞 세그와 합쳐지는 경우가 많다.
        # - 닫는 따옴표가 먼저 나오고 이어지는 경우가 많아서 leading closers를 허용
        quote_cont_start = re.compile(
            r"^[\s\"\'\u201D\u2019\u300D\u300F\u300B\u3009]*"
            r"(?:라|고|라고)\s*"
            r"(?:하고|하니|하되|하였으|하였을|하였|한다고|한다|했다|했을|했으|했는|했|하)"
            r"(?=\s|[\.,，。!?]|$)"
        )

        # 폴백: 패턴이 약간 변형되더라도(예: 조사/공백) '라/고/라고'로 시작하면 병합
        quote_cont_fallback = re.compile(
            r"^[\s\"\'\u201D\u2019\u300D\u300F\u300B\u3009]*(?:라|고|라고)\b"
        )

        merged: List[str] = []
        i = 0
        while i < len(segs):
            cur = segs[i].strip()
            if not cur:
                i += 1
                continue
            if i < len(segs) - 1:
                nxt = segs[i + 1].strip()
                if nxt and (quote_cont_start.match(nxt) or quote_cont_fallback.match(nxt)):
                    # 앞 세그가 닫는 따옴표로 끝나거나, 닫는 따옴표가 문장 끝에 붙어있으면 병합
                    cur_tail = cur.rstrip()
                    if cur_tail and (cur_tail[-1] in closers or cur_tail[-1] in end_punct):
                        merged.append((cur_tail + nxt).strip())
                        i += 2
                        continue
            merged.append(cur)
            i += 1

        return merged

    sentences = merge_quote_ra_continuations(sentences)

    # 4) 너무 긴 문장만 예외적으로 콤마에서 분할 (괄호/브래킷 내부 제외)
    # - 과거 "첫 콤마 1회" 분할은 (1) '뿐 아니라,' 같은 고정 연결 표현에서 과분할,
    #   (2) 분할 지점이 gold와 다르게 선택되어 tgt_exact 불일치를 만드는 사례가 있었다.
    # - 따라서 후보 콤마(, / ，) 중 "균형"이 좋은 지점을 고르고, 필요시 재귀적으로 추가 분할한다.
    def split_long_by_comma_outside_brackets(s: str, limit: int) -> List[str]:
        s = s.strip()
        if len(s) <= limit:
            return [s]

        # 분할 금지 패턴(연결 표현): "...뿐 아니라, ..." 같은 구문은 한 문장으로 취급되는 경우가 많다.
        # 공백/개행이 끼어도 매칭되도록 whitespace 제거 버전으로 검사.
        def _norm_ws(x: str) -> str:
            return re.sub(r"\s+", "", x)

        def _strip_trailing_commas(x: str) -> str:
            return re.sub(r"[，,]+\s*$", "", x)

        def _is_forbidden_left(left_with_comma: str) -> bool:
            base = _strip_trailing_commas(left_with_comma)
            nl = _norm_ws(base)
            return (
                nl.endswith("뿐아니라")
                or nl.endswith("뿐만아니라")
                or nl.endswith("아니라")
            )

        # 괄호/브래킷 내부 콤마는 무시하며 후보 수집
        level = 0
        comma_positions: list[int] = []
        for i, ch in enumerate(s):
            if ch in "([{":
                level += 1
            elif ch in ")]}" and level > 0:
                level -= 1
            elif level == 0 and ch in {",", "，"}:
                comma_positions.append(i)

        if not comma_positions:
            return [s]

        def _pick_best(require_min_frac: bool) -> tuple[float, int, str, str] | None:
            best: tuple[float, int, str, str] | None = None

            # 콤마 뒤가 "새 문장/새 절"처럼 시작하는 경우를 선호한다.
            discourse_start = re.compile(
                r"^(?:또한|그러나|하지만|그런데|이에|따라서|그러므로|그렇다면|즉|즉시|예컨대|혹자|대저|만약|지금|이렇게|그렇게)\b"
            )
            proper_noun_paren = re.compile(r"^[가-힣]{2,10}\s*\(")
            # 콤마 뒤가 '새 주어/새 절'처럼 시작하는 경우를 강하게 우대
            # - 예: 전사공(錢思公)이 / 그가 / 이는 / 이것이 / 이러한 / 어떤 사람이 ...
            right_subject_start = re.compile(
                r"^(?:"
                r"[가-힣]{2,12}\s*\([^\)]{1,20}\)\s*(?:은|는|이|가)\b"
                r"|[가-힣]{2,12}\s*(?:은|는|이|가)\b"
                r"|(?:그는|그가|그것이|이는|이것은)\b"
                r"|(?:어떤\s+사람|어떤\s+자|한\s+사람|사람이|사람은)\b"
                r")"
            )
            left_clause_end = re.compile(
                r"(?:하였고|했고|하며|하면서|하니|하였으니|했으니|하였지만|했지만|라\s*하니|라\s*했다|라고\s*하였다)$"
            )

            def _bonus(left_base: str, right: str) -> float:
                b = 0.0
                r0 = right.lstrip()
                lb = left_base.rstrip()
                if discourse_start.match(r0):
                    b += 25.0
                if proper_noun_paren.match(r0):
                    b += 18.0
                if right_subject_start.match(r0):
                    b += 22.0
                if left_clause_end.search(lb):
                    b += 10.0
                return b

            for pos in comma_positions:
                # gold/기존 데이터와의 정합을 위해 콤마 자체는 좌측에 포함시킨다.
                left = s[: pos + 1].strip()
                right = s[pos + 1 :].strip()
                if not left or not right:
                    continue
                left_base = _strip_trailing_commas(left)
                # 특정 표현은 gold에서 붙여 쓰는 경향이 강하므로 과분할 방지
                # - 예: "...만드니, 이것이 정관례이다." 는 한 문장으로 유지되는 경우가 많다.
                r0 = right.lstrip()
                if r0.startswith("이것이 정관례"):
                    continue
                if r0.startswith("이것이") and re.sub(r"\s+", "", left_base).endswith("만드니"):
                    continue
                if re.search(r"[。！？.!?]$", left_base):
                    continue
                if _is_forbidden_left(left):
                    continue

                # 너무 짧은 조각을 피한다 (과분할 방지)
                if require_min_frac:
                    if len(left_base) < limit * 0.35 or len(right) < limit * 0.35:
                        continue

                # 균형 점수: 분할 후 가장 긴 조각 길이를 최소화
                # - 기본은 가장 긴 조각을 최소화
                # - 단, 콤마 뒤가 새 절처럼 시작하는 지점은 약간 우대(=score 감소)
                score = max(len(left_base), len(right)) - _bonus(left_base, right)
                if best is None or score < best[0]:
                    best = (score, pos, left, right)
            return best

        # 1차(엄격): 기존과 동일한 제약
        best = _pick_best(require_min_frac=True)
        # 2차(완화): 제약 때문에 split 자체가 불가능한 케이스를 방지
        if best is None:
            best = _pick_best(require_min_frac=False)
        if best is None:
            return [s]

        _, _, left, right = best
        # 재귀적으로 추가 분할 (각 조각이 limit 이하가 되거나 더 이상 분할 불가할 때까지)
        return split_long_by_comma_outside_brackets(left, limit) + split_long_by_comma_outside_brackets(right, limit)

    adjusted = []
    for s in sentences:
        parts = split_long_by_comma_outside_brackets(s, max_length)
        adjusted.extend(parts)

    # 5) 닫는 따옴표 단독 문장 방지: ", ', ”, ’, 」, 』, 〉, 》 등 단독 토큰을 앞 문장에 붙인다
    def merge_lonely_closers(segs: List[str]) -> List[str]:
        if not segs:
            return segs
        closers_chars = set([
            ")", "]", "}", "\"", "'",  # ASCII closers including quote/double-quote
            "\u201D", "\u2019",            # ” ’
            "\u3009", "\u300B", "\u300D", "\u300F",  # 〉 》 」 』
            "\u3011", "\u3015",              # 】 〕
            "\uFF07", "\uFF3D", "\uFF5D"   # FULLWIDTH ' ] }
        ])
        # 닫는 따옴표 세트 (문장 시작 검사용)
        closing_quotes = set([
            "\"", "'",
            "\u201D", "\u2019",            # ” ’
            "\u300D", "\u300F",            # 」 』
            "\u300B", "\u3009",            # 》 〉
            "\u3011", "\u3015",            # 】 〕
        ])

        # 따옴표 닫힘 뒤에 붙는 '후행 서술'은 같은 문장으로 취급해야 함.
        # 예) "...". ”는격이니, ... / "...". ”하였습니다.
        # - 실제 데이터에서 종결부호(., 。) 뒤 공백으로 분할되면서 ‘닫는 따옴표+후행구문’이 분리되는 현상이 발생
        tail_after_quote_prefix = re.compile(
            r"^(?:"
            # (1) 인용조사 + 발화동사 (라고/고/라 ... 하니/했다/말했다)류
            r"(?:고|라|[이]?라?고|하고|며|면서)\s*(?:하|말(?:하|했)|말씀(?:하|했)|명(?:하|했)|이르(?:렀)?|대답(?:하|했)|답(?:하|했)|묻|문|물|여쭙|아뢰|전(?:하|했)|칭(?:하|했)|부르|외치)"
            r"(?:셨|ㅆ|시었|시어|시는|시ㄴ|시ㄹ|시|었|았|였|는|ㄴ|ㄹ|을)?"
            r"(?:다|ㄴ다|는다|습니다|ㅂ니다|까|ㄹ까|을까|느냐|ㄴ가|는가|라|거라|소|오|어라|아라|니|으니)"
            r"|"
            # (2) 인용조사 없이 바로 발화동사로 이어지는 케이스: ”하였다/”하였습니다
            r"(?:하|말(?:하|했)|말씀(?:하|했)|명(?:하|했)|이르(?:렀)?|대답(?:하|했)|답(?:하|했)|묻|문|물|여쭙|아뢰|전(?:하|했)|칭(?:하|했)|부르|외치)"
            r"(?:셨|ㅆ|시었|시어|시는|시ㄴ|시ㄹ|시|었|았|였|는|ㄴ|ㄹ|을)?"
            r"(?:다|ㄴ다|는다|습니다|ㅂ니다|까|ㄹ까|을까|느냐|ㄴ가|는가|라|거라|소|오|어라|아라|니|으니)"
            r"|"
            # (3) 격/정의/설명류 후행구문: ”는격이니, ”라는 뜻이다 등
            r"(?:는|란|라는|이라는|이라|이며|이고|이요|이다|이었다|이니|이므로|라서|라며|며|면서|하고|이라고|라고)"
            r")"
        )
        
        merged: List[str] = []
        for seg in segs:
            s = seg.strip()
            if not s:
                continue
            
            # Case 1: 전체가 닫는 기호로만 구성된 경우
            if all(ch in closers_chars for ch in s):
                if merged:
                    merged[-1] = merged[-1].rstrip() + s
                else:
                    merged.append(s)
                continue
            
            # Case 2: 문장이 닫는 따옴표로 시작하는 경우
            if s[0] in closing_quotes and merged:
                # 앞부분의 연속된 닫는 따옴표들 추출
                i = 0
                while i < len(s) and s[i] in closing_quotes:
                    i += 1
                leading = s[:i]
                remaining = s[i:].lstrip()

                # 이전 문장에 닫는 따옴표 붙이기
                # - 종결부호 뒤 공백으로 분할된 케이스를 고려해, quote 앞에 한 칸을 복원
                prev = merged[-1].rstrip()
                merged[-1] = (prev + " " + leading).rstrip()

                # 남은 텍스트가 '후행 서술'이면 문장 경계를 만들지 않고 통째로 병합
                if remaining:
                    if tail_after_quote_prefix.match(remaining):
                        merged[-1] = (merged[-1] + remaining).strip()
                    else:
                        merged.append(remaining)
            else:
                merged.append(s)
        
        return merged

    adjusted = merge_lonely_closers(adjusted)

    # merge_lonely_closers 단계에서 생성될 수 있는
    # "… ”" + "라/고/라고 …" 형태의 경계를 다시 흡수
    adjusted = merge_quote_ra_continuations(adjusted)

    # 최종 반환 전 인용 표지 병합 (여러 단계 처리 후 재병합)
    adjusted = merge_quotation_markers_in_list(adjusted)

    # 위 단계에서 추가한 화자+발화 병합을 최종 결과에도 한 번 더 적용
    adjusted = merge_speaker_utterance_pairs(adjusted)

    # 🔧 선두 구두점 이동: 다음 문장/절 선두로 넘어간 구두점을 이전 문장 끝으로 되돌린다.
    # - mismatch subtype: before_sentence_punct / before_clause_punct 케이스 완화
    # - 예: "… 있었소|.내가 …" → "… 있었소." + "내가 …"
    def shift_leading_punct_to_prev(segs: List[str]) -> List[str]:
        if len(segs) <= 1:
            return segs

        # 종결/절 구두점 + 생략부호 변종 + 닫는 괄호/따옴표 변종(일부는 merge_lonely_closers가 처리하나, 여기서는 포괄)
        lead_punct = set(".!?。！？…") | set(",;:，；：、") | set("…⋯‥︙")
        lead_punct |= set(")）]】}」』〉》”’\"'" ) | set(["］", "｝", "〕"])

        out: List[str] = []
        for seg in segs:
            s = str(seg).strip()
            if not s:
                continue

            if out:
                s0 = s.lstrip()
                if s0 and s0[0] in lead_punct:
                    i = 0
                    while i < len(s0) and s0[i] in lead_punct:
                        i += 1
                    leading = s0[:i]
                    remaining = s0[i:].lstrip()

                    out[-1] = out[-1].rstrip() + leading
                    if remaining:
                        out.append(remaining)
                    continue

            out.append(s)

        return out

    adjusted = shift_leading_punct_to_prev(adjusted)

    if refine_boundaries_with_llm:
        try:
            adjusted = refine_boundaries_with_llm(text, adjusted, task="pa")
        except Exception:
            pass
    
    return adjusted

def split_with_new_parsers(text: str, is_target: bool = True, use_siku_preprocessing: bool = True) -> List[str]:
    """새로운 파서들(SuPar-Kanbun/Stanza)을 사용한 문장 분할 - 인용 표지 병합 지원"""
    # SikuBERT 전처리 적용
    if use_siku_preprocessing and contains_chinese(text):
        text = preprocess_with_siku_tokenization(text)
    
    try:
        # is_target이 True면 번역문(Stanza), False면 원문(SuPar-Kanbun)
        sentences = smart_sentence_split(text, is_source=not is_target)
        
        # 번역문인 경우 인용 표지 병합 처리
        if is_target and sentences:
            sentences = merge_quotation_markers_in_list(sentences)
            if refine_boundaries_with_llm:
                try:
                    sentences = refine_boundaries_with_llm(text, sentences, task="pa")
                except Exception:
                    pass
        
        return sentences if sentences else [text]
    except Exception as e:
        print(f"⚠️ 새 파서 분할 실패, 폴백: {e}")
        result = split_with_smart_punctuation_rules(text)
        # 폴백에서도 인용 표지 병합
        if is_target:
            result = merge_quotation_markers_in_list(result)
        return result

def merge_quotation_markers_in_list(sentences: List[str]) -> List[str]:
    """인용 표지를 이전 인용문과 병합 (PA 전용 헬퍼) - 중첩 인용 지원
    
    인용 표지 구조: [인용조사] + [동사어간] + [어미]
    예: "고 하였다", "라고 말한다", "하고 명하셨다", "며 여쭙는다" 등
    중첩 예: [문장1] + [인용표지1] + [인용표지2] → 모두 병합
    """
    if len(sentences) <= 1:
        return sentences
    
    # 인용 조사 패턴 (빈번한 단독 '고' 포함)
    quotation_particles = r'(고|[이]?라?고|하고|며|면서)'
    
    # 발화 동사 어간 (인용 표지에 자주 쓰이는 동사들)
    # - 축약형("말했다"=말했-, "답했다"=답했-)까지 포괄하도록 보강
    # - 너무 광범위한 매칭을 피하기 위해 발화 관련 어근만 허용
    speech_verbs = r'(?:'
    speech_verbs += r'하'
    speech_verbs += r'|말(?:하|했)'
    speech_verbs += r'|말씀(?:하|했)'
    speech_verbs += r'|명(?:하|했)'
    speech_verbs += r'|이르(?:렀)?'
    speech_verbs += r'|대답(?:하|했)'
    speech_verbs += r'|답(?:하|했)'
    speech_verbs += r'|묻|문|물'
    speech_verbs += r'|여쭙|아뢰'
    speech_verbs += r'|전(?:하|했)'
    speech_verbs += r'|칭(?:하|했)'
    speech_verbs += r'|부르|외치'
    speech_verbs += r')'
    
    # 존칭+시제 통합 패턴 (축약형 포함)
    # "시었다" → "셨다", "시었" → "셨", "시어" → "셔" 등
    honorific_tense = r'(?:셨|ㅆ|시었|시어|시는|시ㄴ|시ㄹ|시|었|았|였|는|ㄴ|ㄹ|을)?'
    
    # 종결 어미
    endings = r'(다|ㄴ다|는다|습니다|ㅂ니다|까|ㄹ까|을까|느냐|ㄴ가|는가|라|거라|소|오|어라|아라|니|으니)'
    
    # 따옴표(닫는 따옴표)와 종결 부호 (선택적)
    # - 실제 데이터에서 자주 나오는 동아시아 따옴표/괄호 닫힘 기호까지 포함
    closing_quote = r'["\'”’」』》〉】〕\)\]\}]?'
    punctuation = r'[\.。?!,，]?'  # 마침표/물음표/쉼표 허용

    # 마커 1개 조각: (선행 닫는 따옴표) + 인용조사 + 동사(시제/존칭) + 종결어미 + (부호/닫는 따옴표)
    marker_chunk = (
        closing_quote +               # 문장 시작에 붙을 수 있는 닫는 따옴표
        r'\s*' +
        quotation_particles +
        r'\s*' +
        speech_verbs +
        honorific_tense +
        endings +
        r'\s*' +
        punctuation +
        r'\s*' +
        closing_quote +               # 끝쪽 닫는 따옴표 허용
        r'\s*'
    )

    # 전체 패턴: 마커 조각이 1회 이상 연쇄된 문장 전체
    quotation_marker_pattern = r'^\s*(?:' + marker_chunk + r')+$'

    # 비종결(관형/명사절) 인용 구문도 병합
    # 예: "...". '고 하는 것입니다.' / "...". '라고 하는 바는 ...'
    # - '고 하다'가 종결(다/습니다) 대신 관형형(는/던)으로 끝나는 형태
    relative_quote_tail_pattern = (
        r'^\s*' + closing_quote + r'\s*'
        r'(?:고|[이]?라?고|하고|며|면서)\s*'
        r'하(?:는|던)\b'
    )
    
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
                if re.match(quotation_marker_pattern, next_sent, re.IGNORECASE) or re.match(relative_quote_tail_pattern, next_sent, re.IGNORECASE):
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

    # 🎯 인용 표지가 문장 맨 앞에 붙은 경우도 앞 문장으로 당겨 붙이기
    marker_prefix_regex = re.compile(r'^\s*(?:' + marker_chunk + r'|' + relative_quote_tail_pattern + r')', re.IGNORECASE)
    merged_prefix = []
    for idx, seg in enumerate(sentences):
        if idx > 0 and seg and marker_prefix_regex.match(seg):
            # 바로 앞 문장에 현재 전체 구문을 붙인다 (내용 보존)
            merged_prefix[-1] = (merged_prefix[-1].rstrip() + ' ' + seg.lstrip()).strip()
        else:
            merged_prefix.append(seg)

    # 🧷 닫는 따옴표/괄호 + (라/고/라고 ...) 연속 병합 (new parser 경로 보호)
    # - 예: "...없다. ”" + "고 대답한다." / "라고 하니," 등
    # - Stanza/파서가 종결부호를 보고 공백 없이도 경계를 넣는 경우가 있어 후처리로 흡수한다.
    closers_tail = set(['"', "'", '\u201D', '\u2019', '\u300D', '\u300F', '\u300B', '\u3009', '】', '〕', ')', '）', ']', '】', '}', '」', '』', '〉', '》'])
    end_punct_tail = set(['.', '!', '?', '。', '！', '？'])
    quote_cont_fallback = re.compile(
        r"^[\s\"\'\u201D\u2019\u300D\u300F\u300B\u3009]*"
        r"(?:라|고|라고)\b"
    )

    merged2: List[str] = []
    i = 0
    while i < len(merged_prefix):
        cur = merged_prefix[i].strip()
        if not cur:
            i += 1
            continue
        if i < len(merged_prefix) - 1:
            nxt = merged_prefix[i + 1].strip()
            if nxt and quote_cont_fallback.match(nxt):
                tail = cur.rstrip()
                if tail and (tail[-1] in closers_tail or tail[-1] in end_punct_tail):
                    merged2.append((tail + nxt).strip())
                    i += 2
                    continue
        merged2.append(cur)
        i += 1

    return merged2

def split_with_smart_punctuation_rules(text: str) -> List[str]:
    """중국고전 문장 분할 패턴 강화 + 의미 기반 경계 감지"""
    
    # 1) 구두점 패턴 시도
    classical_chinese_patterns = [
        r'(?<=[。？！])',  # 기본 문장 부호
        r'(?<=하고)\s*',   # '하고' 뒤 분할
        r'(?<=하여)\s*',   # '하여' 뒤 분할  
        r'(?<=하니)\s*',   # '하니' 뒤 분할
    ]
    
    for pattern in classical_chinese_patterns:
        if re.search(pattern, text):
            segments = re.split(pattern, text)
            segments = [seg.strip() for seg in segments if seg.strip()]
            if len(segments) > 1:
                return segments
    
    # 2) 구두점이 없으면 의미 기반 경계 감지 (PA: 문장 단위)
    if len(text) > 100 and BGE_AVAILABLE:
        try:
            offsets = detect_semantic_boundaries(text, window_size=80, threshold=0.75, min_segment_length=30)
            if len(offsets) > 1:
                sentences = [text[start:end].strip() for start, end in offsets if start < end]
                sentences = [s for s in sentences if s]
                if sentences:
                    print(f"✅ 의미 기반 경계 감지: {len(sentences)}개 문장")
                    return sentences
        except Exception as e:
            print(f"⚠️ 의미 기반 경계 감지 실패: {e}")
    
    # 🆕 중국고전 특화 분할 패턴 (문단식별자 2 문제 해결)
    classical_chinese_patterns = [
        r'(?<=[。？！])',  # 기본 문장 부호
        r'(?<=하고)\s*',   # '하고' 뒤 분할
        r'(?<=하여)\s*',   # '하여' 뒤 분할  
        r'(?<=하니)\s*',   # '하니' 뒤 분할
        r'(?<=이요)\s*',   # '이요' 뒤 분할 (고전 어미)
        r'(?<=이며)\s*',   # '이며' 뒤 분할
        r'(?<=라)\s*',     # '라' 뒤 분할 (고전 어미)
        r'(?<=것을)\s*',   # '것을' 뒤 분할
        r'(?<=것이)\s*',   # '것이' 뒤 분할
        r'(?<=한다)\s*\.', # '한다.' 뒤 분할
    ]
    
    # 첫 번째: 고전 한문 패턴 적용
    for pattern in classical_chinese_patterns:
        if re.search(pattern, text):
            segments = re.split(pattern, text)
            # 빈 문자열 제거하고 앞뒤 공백 정리
            segments = [seg.strip() for seg in segments if seg.strip()]
            if len(segments) > 1:
                return segments
    
    # 폴백: 기존 패턴
    pattern = r'(?<=[。？！○])|(?<=[.!?]\s)'
    segments = re.split(pattern, text)
    return [seg.strip() for seg in segments if seg.strip()]

def apply_legacy_rules(sentences: List[str], max_length: int = 150) -> List[str]:
    length_adjusted = []
    for sent in sentences:
        if len(sent) > max_length:
            length_adjusted.extend(split_long_sentence_semantically(sent, max_length))
        else:
            length_adjusted.append(sent)
    return merge_low_chinese_segments(length_adjusted)

def split_long_sentence_semantically(sentence: str, max_length: int) -> List[str]:
    if len(sentence) <= max_length:
        return [sentence]
    parts = []
    remaining = sentence
    while len(remaining) > max_length:
        split_pos = find_semantic_split_near_position(remaining, max_length)
        if split_pos > 0:
            parts.append(remaining[:split_pos])
            remaining = remaining[split_pos:]
        else:
            break
    if remaining:
        parts.append(remaining)
    return parts

def find_semantic_split_near_position(text: str, target_pos: int) -> int:
    start = max(0, target_pos - 20)
    end = min(len(text), target_pos + 20)
    search_text = text[start:end]
    split_patterns = [
        (r'[。！？○]', 1),
        (r'[.!?]\s', 2),
        (r'[：]', 1),
        (r'[:]\s', 2),
        (r'[，]\s*(?=.{10,})', 1),
        (r'[,]\s+(?=.{10,})', 2),
        (r'\s+', 1),
    ]
    for pattern, offset in split_patterns:
        matches = list(re.finditer(pattern, search_text))
        if matches:
            return start + matches[0].end()
    return target_pos

def detect_semantic_boundaries(text: str, window_size: int = 50, threshold: float = 0.75, min_segment_length: int = 20) -> List[Tuple[int, int]]:
    """
    의미 기반 경계 감지 (BGE 임베딩 + 코사인 유사도)
    
    슬라이딩 윈도우로 텍스트를 스캔하면서 의미 변화가 큰 지점을 경계로 판단
    
    Args:
        text: 분할할 텍스트
        window_size: 슬라이딩 윈도우 크기 (문자 단위)
                    - PA(문장 단위): 80-100 권장
                    - SA(구 단위): 20-30 권장
        threshold: 코사인 유사도 임계값 (낮을수록 경계로 판단)
        min_segment_length: 최소 세그먼트 길이 (이보다 짧으면 병합)
    
    Returns:
        [(0, 45), (45, 92), ...] 형태의 오프셋 쌍
    """
    if not BGE_AVAILABLE or len(text) < window_size * 2:
        return [(0, len(text))]
    
    try:
        embedder = get_embedding_manager()
        
        # 1. 슬라이딩 윈도우로 텍스트 세그먼트 추출
        segments = []
        segment_offsets = []
        step_size = window_size // 2  # 50% 오버랩
        
        for start in range(0, len(text) - window_size + 1, step_size):
            end = start + window_size
            segment = text[start:end].strip()
            if segment:
                segments.append(segment)
                segment_offsets.append((start, end))
        
        # 마지막 부분 처리
        if segment_offsets and segment_offsets[-1][1] < len(text):
            last_segment = text[segment_offsets[-1][1]:].strip()
            if last_segment:
                segments.append(last_segment)
                segment_offsets.append((segment_offsets[-1][1], len(text)))
        
        if len(segments) < 2:
            return [(0, len(text))]
        
        # 2. BGE 임베딩 계산
        embeddings = embedder.compute_embeddings_with_cache(
            segments, 
            batch_size=32,
            use_multi_vector=False  # Dense vector만 사용 (1024차원)
        )
        
        if embeddings is None or len(embeddings) == 0:
            return [(0, len(text))]
        
        # 3. 인접 세그먼트 간 코사인 유사도 계산
        similarities = []
        for i in range(len(embeddings) - 1):
            emb1 = embeddings[i]
            emb2 = embeddings[i + 1]
            
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 > 0 and norm2 > 0:
                cosine_sim = np.dot(emb1, emb2) / (norm1 * norm2)
            else:
                cosine_sim = 1.0
            
            similarities.append(cosine_sim)
        
        # 4. 유사도가 threshold 이하인 지점을 경계로 판단
        boundary_candidates = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                boundary_pos = (segment_offsets[i][1] + segment_offsets[i + 1][0]) // 2
                boundary_candidates.append(boundary_pos)
        
        if not boundary_candidates:
            return [(0, len(text))]
        
        # 5. 경계로 문장 분할
        boundaries = []
        start = 0
        for pos in sorted(set(boundary_candidates)):
            if pos > start:
                boundaries.append((start, pos))
                start = pos
        
        if start < len(text):
            boundaries.append((start, len(text)))
        
        # 너무 짧은 세그먼트 병합
        merged_boundaries = []
        for start, end in boundaries:
            if len(merged_boundaries) > 0 and end - start < min_segment_length:
                prev_start, _ = merged_boundaries[-1]
                merged_boundaries[-1] = (prev_start, end)
            else:
                merged_boundaries.append((start, end))
        
        return merged_boundaries if merged_boundaries else [(0, len(text))]
        
    except Exception as e:
        print(f"⚠️ 의미 기반 경계 감지 실패: {e}")
        return [(0, len(text))]

def merge_low_chinese_segments(sentences: List[str]) -> List[str]:
    if not sentences:
        return []
    merged, buffer = [], ''
    for sent in sentences:
        han_count = len(regex.findall(r'\p{Han}', sent))
        if han_count <= 3:
            buffer += sent
        else:
            if buffer:
                merged.append(buffer)
                buffer = ''
            merged.append(sent)
    if buffer:
        if merged:
            merged[-1] += buffer
        else:
            merged.append(buffer)
    return [s for s in merged if s]

def contains_chinese(text: str) -> bool:
    chinese_count = len(regex.findall(r'\p{Han}', text))
    return chinese_count > len(text) * 0.3

def split_source_by_whitespace_and_align(
    source: str,
    target_count: int,
    target_sentences: List[str] = None,
    embedder_name: str = "bge",
    embedder_func=None,
    enable_src_marker_whitespace_dp_bonus: bool = False,
    max_workers: int = 4,
    batch_size: int = 100,
    debug_meta_out: Dict[str, Any] | None = None,
) -> List[str]:
    """
    원문(한문) 분할: 어절 경계에서만 분할, 어절 내부 절대 분할 금지!
    🎯 원본 텍스트 형태 절대 보존: augmentation은 임베딩 계산용만 사용
    
    Args:
        source: 원문 텍스트
        target_count: 분할할 개수
        target_sentences: 번역문 문장들 (의미적 매칭용)
        embedder_name: 사용할 임베더 이름 ("bge" 또는 "openai")
        embedder_func: 외부에서 전달된 임베더 함수 (선택적)
    """
    def augment_source_hanja(text: str) -> str:
        """원문 괄호 속 한자 노출 (임베딩용만)"""
        import regex as re
        han_regex = re.compile(r"\p{Han}")
        def repl(m: re.Match) -> str:
            inner = m.group(1)
            hanja = ''.join(ch for ch in inner if han_regex.match(ch))
            return f" {hanja} " if hanja else " "
        text_aug = re.sub(r"\(([^)]*)\)", repl, text)
        text_aug = re.sub(r"\[([^\]]*)\]", repl, text_aug)
        return ' '.join(text_aug.split())
    
    def strip_bracket_segments(text: str):
        """[-…] 또는 [-(…)] 구간을 정렬에서 완전히 제외하고, 복원용 위치/내용을 기록한다.

        - 앞뒤 공백은 제거하지 않고 남겨 위치/간격을 그대로 보존한다.
        - working 텍스트를 순차적으로 구성하면서 현재 working 오프셋 기준으로 삽입 지점을 기록한다.

        Returns:
            working (str): 괄호 구간이 제거된 텍스트
            insertions (List[Tuple[int, str]]): (working_index, bracket_text)
        """
        import re as _re
        pattern = _re.compile(r"\[-\([^)]*\)\]|\[-[^\]]*\]")
        insertions: List[Tuple[int, str]] = []
        parts: List[str] = []
        last = 0
        curr_len = 0
        for m in pattern.finditer(text):
            # 본문 부분 추가
            parts.append(text[last:m.start()])
            curr_len += (m.start() - last)
            # 괄호 블록은 제거, 복원용으로 현재 working 오프셋에 삽입
            bracket_text = m.group(0)
            insertions.append((curr_len, bracket_text))
            last = m.end()
        parts.append(text[last:])
        working = ''.join(parts)
        return working, insertions

    def restore_brackets_in_chunks(chunks: List[str], insertions: List[Tuple[int, str]]) -> List[str]:
        """작게 분할된 working 텍스트 조각들에 기록된 괄호 블록을 원위치에 삽입한다.

        insertions는 working 텍스트의 전역 오프셋 기준이며, 각 조각의 누적 길이를 이용해
        적절한 조각과 로컬 오프셋을 계산하여 삽입한다.
        """
        if not insertions:
            return chunks

        # 각 chunk의 전역 누적 오프셋 경계 계산
        cumulative = [0]
        for ch in chunks:
            cumulative.append(cumulative[-1] + len(ch))

        # 문자열 삽입이 빈번하므로 리스트로 변환 후 조합
        chunk_buffers = [list(ch) for ch in chunks]

        for pos, content in insertions:
            # pos가 어느 chunk에 속하는지 탐색 (cumulative[i] <= pos <= cumulative[i+1])
            idx = 0
            while idx < len(chunks) and not (cumulative[idx] <= pos <= cumulative[idx+1]):
                idx += 1
            if idx >= len(chunks):
                # 마지막 범위를 넘어가면 최종 조각 뒤에 붙임
                chunk_buffers[-1].extend(list(content))
                # 경계 갱신
                cumulative[-1] += len(content)
                continue

            # 경계에 정확히 일치하면 앞 조각에 붙여 이전 어절과 결합
            if pos == cumulative[idx] and idx > 0:
                idx -= 1
            local_pos = pos - cumulative[idx]
            # local_pos는 0~len(chunk) 범위. 그 위치에 content 삽입
            # 리스트 삽입 비용을 줄이기 위해 앞/뒤로 분할 병합
            buf = chunk_buffers[idx]
            left = buf[:local_pos]
            right = buf[local_pos:]
            chunk_buffers[idx] = left + list(content) + right

            # 이후 조각들의 경계 보정
            delta = len(content)
            for j in range(idx+1, len(cumulative)):
                cumulative[j] += delta

        return [''.join(b) for b in chunk_buffers]
    if not source.strip():
        return [''] * target_count

    # 0. 비대응 구간 제거 후(정렬에서 제외) 분할 처리, 최종에 복원
    working_source, bracket_insertions = strip_bracket_segments(source)

    # 🎯 원본 어절 보존 (최종 출력용) - 어절 경계의 원문 오프셋을 수집하여 슬라이스로 재조립
    # 공백을 기준으로 어절(span)들을 추출하되, 최종 결과 조립은 원문 슬라이스로 수행
    word_spans = []  # List[Tuple[start, end]] for each non-whitespace token
    for m in re.finditer(r"\S+", working_source):
        word_spans.append((m.start(), m.end()))
    words_original = [working_source[s:e] for (s, e) in word_spans]

    def slice_segment_by_word_index(i_start: int, i_end: int) -> str:
        """어절 인덱스 구간 [i_start, i_end) 에 해당하는 원문 슬라이스를 그대로 반환
        - 경계 사이의 공백/개행 포함 원문을 보존
        - 인덱스가 유효하지 않으면 빈 문자열 반환
        """
        if i_start >= i_end or i_start < 0 or i_end > len(word_spans):
            return ''
        start_char = word_spans[i_start][0]
        end_char = word_spans[i_end - 1][1]
        return working_source[start_char:end_char]
    
    # 🎯 Augmented 어절 생성 (임베딩 계산용만)
    # working_source 기준으로 augmentation (원본 텍스트는 최종 슬라이스로만 사용)
    source_augmented = augment_source_hanja(working_source)
    words_aug = source_augmented.split()
    
    # 1. 어절 단위로 분할 (공백 기준, 어절 내부 절대 분할 금지!)
    if not words_original:
        return [''] * target_count

    # 어절이 target_count보다 적으면 균등 분배 (원본 사용!)
    if len(words_original) <= target_count:
        result = []
        for i in range(target_count):
            if i < len(words_original):
                # 단일 어절 슬라이스 (원문 보존)
                result.append(slice_segment_by_word_index(i, i+1))
            else:
                result.append('')
        return restore_brackets_in_chunks(result, bracket_insertions)
    
    # 1-1. PA 원문에 의미 기반 경계 감지 적용 (어절 경계만 지키면서 문장 단위로)
    # ⚠️ 번역문 문장들이 주어진 경우(target_sentences), 목표는 "정확히 target_count개"이므로
    # 개수가 가변적인 의미 경계 탐지 결과를 그대로 반환하면 빈 세그먼트 패딩이 발생할 수 있다.
    # 따라서 target_sentences가 없을 때만 보조적으로 사용한다.
    if (not target_sentences) and len(words_original) > 10 and BGE_AVAILABLE:  # 어절 10개 이상이면 의미 분석
        try:
            # 어절을 공백으로 재조합
            reconstructed_text = ' '.join(words_original)
            
            # 의미 기반 경계 감지 (PA: 문장 단위, window_size=80)
            offsets = detect_semantic_boundaries(
                reconstructed_text, 
                window_size=80, 
                threshold=0.75, 
                min_segment_length=30
            )
            
            if len(offsets) > 1 and len(offsets) <= target_count:
                # 오프셋을 어절 경계로 변환
                result = []
                for start, end in offsets:
                    # 재구성 텍스트의 오프셋 → 어절 인덱스 매핑
                    # reconstructed_text = ' '.join(words_original)
                    # 각 어절의 시작 오프셋을 누적하여 인덱스를 찾는다
                    recon_starts = []
                    pos = 0
                    for w in words_original:
                        recon_starts.append(pos)
                        pos += len(w) + 1  # 공백 포함 누적
                    recon_ends = [s + len(w) for s, w in zip(recon_starts, words_original)]

                    # 시작/끝 인덱스 추정
                    word_start = 0
                    for idx, s in enumerate(recon_starts):
                        if s >= start:
                            word_start = idx
                            break
                    else:
                        word_start = len(words_original)

                    word_end = len(words_original)
                    for idx, e in enumerate(recon_ends):
                        if e >= end:
                            word_end = idx + 1
                            break

                    if word_start < word_end:
                        segment = slice_segment_by_word_index(word_start, word_end)
                        result.append(segment)
                
                # 이 경로는 target_sentences가 없을 때만 사용하지만,
                # 그래도 빈/부분 결과를 그대로 반환하지 않도록 최소한의 가드
                if len(result) > 0:
                    print(f"✅ PA 원문 의미 기반 분할: {len(result)}개 어절 그룹")
                    return restore_brackets_in_chunks(result, bracket_insertions)
        except Exception as e:
            print(f"⚠️ PA 원문 의미 기반 분할 실패: {e}")
    
    # 2. 임베딩 기반 의미적 분할 (어절 경계에서만!)
    if target_sentences and len(target_sentences) > 0:
        try:
            def _as_vector(x):
                # embedder가 list/np.ndarray 등 무엇을 반환하든 1D numpy로 정리
                if x is None:
                    return None
                arr = np.asarray(x)
                if arr.ndim == 1:
                    return arr
                # 예상 밖 형태면 평탄화
                return arr.reshape(-1)

            def _cosine(a, b) -> float:
                a = _as_vector(a)
                b = _as_vector(b)
                if a is None or b is None:
                    return 0.0
                an = float(np.linalg.norm(a))
                bn = float(np.linalg.norm(b))
                if an <= 1e-8 or bn <= 1e-8:
                    return 0.0
                return float(np.dot(a, b) / (an * bn))

            _ending_bonus_strong = [
                '이어늘', '하니', '하매', '하자', '하더니', '이니', '이라', '하되', '하노라'
            ]
            _ending_bonus_weak = ['니', '라', '다', '고', '며']

            def _marker_end_bonus(last_word: str) -> float:
                """현토(한글 marker) 패턴 기반의 아주 약한 경계 보너스.

                - 어절 끝에 한글이 1~2자 붙고, 그 직전 문자가 CJK(한자)면 marker로 간주
                  예: “也에”, “之者가” 등
                - DP 경계 선택 tie-break 수준으로만 사용
                """

                if not enable_src_marker_whitespace_dp_bonus:
                    return 0.0

                w = (last_word or '').strip()
                if not w:
                    return 0.0

                # trailing punctuation/quotes/brackets 제거
                w = w.rstrip("\"'”’」』》〉】〕)\]\}.,，。?!;:·、")
                if not w:
                    return 0.0

                def is_hangul(ch: str) -> bool:
                    o = ord(ch)
                    return 0xAC00 <= o <= 0xD7A3

                def is_cjk(ch: str) -> bool:
                    o = ord(ch)
                    return (
                        0x4E00 <= o <= 0x9FFF
                        or 0x3400 <= o <= 0x4DBF
                        or 0xF900 <= o <= 0xFAFF
                    )

                end = len(w)
                start = end - 1
                if start < 0 or not is_hangul(w[start]):
                    return 0.0

                while start >= 0 and is_hangul(w[start]):
                    start -= 1
                start += 1

                mlen = end - start
                if mlen <= 0:
                    return 0.0

                prev = start - 1
                if prev < 0 or not is_cjk(w[prev]):
                    return 0.0

                # 3자 이상은 현토라 보기 어려워 제외
                if mlen >= 3:
                    return 0.010
                if mlen == 2:
                    return 0.008
                return 0.003

            def _boundary_end_bonus(last_word: str) -> float:
                w = (last_word or '').strip()
                if not w:
                    return 0.0
                # 구두점 종결
                if re.search(r"[。？！.!?]$", w):
                    return 0.12
                for suf in _ending_bonus_strong:
                    if w.endswith(suf):
                        return 0.18
                for suf in _ending_bonus_weak:
                    if w.endswith(suf):
                        return 0.06
                return 0.0

            def _build_candidate_cuts(words: List[str]) -> List[int]:
                # 어절 인덱스(0..W) 중 후보 컷 생성
                W = len(words)
                cuts = {0, W}

                for i, w in enumerate(words, start=1):
                    if _boundary_end_bonus(w) > 0:
                        cuts.add(i)
                    # marker 단서도 후보 컷으로 추가(ON일 때만)
                    if enable_src_marker_whitespace_dp_bonus and _marker_end_bonus(w) > 0:
                        cuts.add(i)

                # 너무 후보가 적으면 주기적으로 추가 (의미 DP가 완전히 무력화되는 것 방지)
                step = 2 if W >= 30 else 1
                for i in range(0, W + 1, step):
                    cuts.add(i)

                cuts = sorted(cuts)
                # 여전히 부족하면 전체 컷 허용
                if len(cuts) < target_count + 1:
                    cuts = list(range(0, W + 1))
                return cuts

            # 임베더 준비(실패 시 하이브리드로 폴백). 이 블록 실패는 분할 전체 실패가 아니다.
            try:
                if debug_meta_out is not None:
                    debug_meta_out.clear()
                    debug_meta_out["enabled_marker_bonus"] = bool(enable_src_marker_whitespace_dp_bonus)
                # 외부에서 전달된 임베더 함수가 있으면 우선 사용
                if embedder_func:
                    embed_func = embedder_func
                    target_embeddings = embed_func(target_sentences)
                    print(f"✅ 외부 임베더 함수 사용 ({embedder_name})")
                    if debug_meta_out is not None:
                        debug_meta_out["embedder_used"] = f"external({embedder_name})"

                # OpenAI 임베더 사용
                elif embedder_name == "openai":
                    # 외부에서 전달된 임베더 함수를 사용해야 함
                    if not embedder_func:
                        print("⚠️ OpenAI 임베더 함수가 전달되지 않았습니다. BGE로 폴백합니다.")
                        embedder_name = "bge"  # BGE로 폴백
                    else:
                        embed_func = embedder_func
                        target_embeddings = embed_func(target_sentences)
                        print("✅ OpenAI 임베딩 사용 (외부 함수)")
                        if debug_meta_out is not None:
                            debug_meta_out["embedder_used"] = "openai_external"

                # BGE 임베더 사용 (기본값 또는 OpenAI 실패시 폴백)
                if embedder_name == "bge":
                    from common.embedders.bge import get_embedding_manager

                    embedder = get_embedding_manager()
                    # PA에서도 SA와 동일한 안정적 멀티벡터 설정을 공유하므로, 용도를 명확히 표기
                    print("✅ BGE-M3 Multi-Vector 임베딩 사용 (PA 분할용, 작은 배치)")

                    # SA 방식: 작은 배치 크기 + Multi-vector로 안전하게 처리
                    target_embeddings = embedder.compute_embeddings_with_cache(
                        target_sentences,
                        batch_size=4,
                        use_multi_vector=True
                    )

                    def bge_embed_source(texts):
                        # PA에서도 SA처럼 작은 배치 + Multi-vector로 처리
                        return embedder.compute_embeddings_with_cache(
                            texts,
                            batch_size=4,
                            use_multi_vector=True
                        )

                    embed_func = bge_embed_source
                    if debug_meta_out is not None:
                        debug_meta_out["embedder_used"] = "bge_m3_multivector"

            except Exception as e:
                print(f"⚠️ 임베딩 실패, 하이브리드로 폴백: {e}")
                from common.tokenizers import siku_get_embeddings
                from common.tokenizers import get_roberta_hanja_tokenizer

                print("✅ 하이브리드 임베딩 폴백 (원문:SikuBERT + 번역문:RoBERTa)")

                roberta_tokenizer = get_roberta_hanja_tokenizer()

                def get_roberta_embeddings(texts, batch_size=32):
                    embeddings = []
                    for text in texts:
                        emb = roberta_tokenizer.get_embeddings(text)
                        embeddings.append(emb.cpu().numpy())
                    return np.array(embeddings)

                target_embeddings = get_roberta_embeddings(target_sentences, batch_size=32)

                def hybrid_embed_source(texts):
                    return siku_get_embeddings(texts, batch_size=32)

                embed_func = hybrid_embed_source
                if debug_meta_out is not None:
                    debug_meta_out["embedder_used"] = "hybrid_fallback"
                    debug_meta_out["embedder_fallback_error"] = str(e)

            # === 실제 DP 분할 (BGE 성공 경로에서도 수행) ===
            N = target_count
            W = len(words_original)
            if W <= 0:
                return restore_brackets_in_chunks([''] * target_count, bracket_insertions)

            cuts = _build_candidate_cuts(words_original)
            if debug_meta_out is not None:
                try:
                    debug_meta_out["words_count"] = int(len(words_original))
                    debug_meta_out["cuts_count"] = int(len(cuts))
                    debug_meta_out["boundary_hint_words"] = int(sum(1 for w in words_original if _boundary_end_bonus(w) > 0))
                    if enable_src_marker_whitespace_dp_bonus:
                        debug_meta_out["marker_hint_words"] = int(sum(1 for w in words_original if _marker_end_bonus(w) > 0))
                    else:
                        debug_meta_out["marker_hint_words"] = 0
                except Exception:
                    pass

            # span 임베딩 캐시: 후보 컷 쌍만 계산 (과도한 O(W^2) 방지)
            max_words_per_span = max(5, int((W / max(1, N)) * 3))
            span_keys = []
            span_texts = []

            for a_pos, a in enumerate(cuts[:-1]):
                for b in cuts[a_pos + 1:]:
                    if b <= a:
                        continue
                    if (b - a) > max_words_per_span:
                        continue
                    seg_raw = slice_segment_by_word_index(a, b)
                    seg_aug = augment_source_hanja(seg_raw)
                    seg_aug = ' '.join(seg_aug.split())
                    if not seg_aug:
                        continue
                    span_keys.append((a, b))
                    span_texts.append(seg_aug)

            span_cache = {}
            if span_texts:
                try:
                    span_embs = embed_func(span_texts)
                    for key, emb in zip(span_keys, span_embs):
                        span_cache[key] = _as_vector(emb)
                except Exception:
                    span_cache = {}

            dp = np.full((N + 1, len(cuts)), -np.inf)
            back = np.full((N + 1, len(cuts)), -1, dtype=int)
            dp[0, 0] = 0.0

            target_vecs = [_as_vector(e) for e in target_embeddings]

            for i in range(1, N + 1):
                tgt = target_vecs[i - 1]
                for j_pos in range(1, len(cuts)):
                    j = cuts[j_pos]
                    # 남은 구간에 최소 1어절씩 배분 가능해야 함
                    remaining_segments = N - i
                    remaining_words = W - j
                    if remaining_words < remaining_segments:
                        continue

                    for k_pos in range(0, j_pos):
                        k = cuts[k_pos]
                        if (j - k) <= 0:
                            continue
                        if dp[i - 1, k_pos] == -np.inf:
                            continue
                        if (j - k) > max_words_per_span:
                            continue

                        span_vec = span_cache.get((k, j))
                        if span_vec is None:
                            # 캐시 미존재 시 즉시 임베딩(드물게)
                            try:
                                seg_raw = slice_segment_by_word_index(k, j)
                                seg_aug = augment_source_hanja(seg_raw)
                                seg_aug = ' '.join(seg_aug.split())
                                if not seg_aug:
                                    continue
                                span_vec = _as_vector(embed_func([seg_aug])[0])
                                span_cache[(k, j)] = span_vec
                            except Exception:
                                continue

                        sim = _cosine(span_vec, tgt)
                        bonus = _boundary_end_bonus(words_original[j - 1]) if j - 1 < len(words_original) else 0.0
                        if enable_src_marker_whitespace_dp_bonus and (j - 1) < len(words_original):
                            bonus += _marker_end_bonus(words_original[j - 1])
                        score = dp[i - 1, k_pos] + sim + bonus
                        if score > dp[i, j_pos]:
                            dp[i, j_pos] = score
                            back[i, j_pos] = k_pos

            # 마지막 컷은 W여야 하는데 cuts에 포함되어 있음
            end_pos = cuts.index(W) if W in cuts else (len(cuts) - 1)
            if dp[N, end_pos] == -np.inf:
                raise Exception("DP failed to find valid split")

            # 역추적
            cut_positions = [end_pos]
            curr_pos = end_pos
            for i in range(N, 0, -1):
                prev_pos = back[i, curr_pos]
                if prev_pos < 0:
                    raise Exception("DP backtrack failed")
                cut_positions.append(prev_pos)
                curr_pos = prev_pos
            cut_positions.reverse()

            word_cuts = [cuts[p] for p in cut_positions]
            if len(word_cuts) != N + 1:
                raise Exception("Invalid cut reconstruction")

            result = []
            for i in range(N):
                result.append(slice_segment_by_word_index(word_cuts[i], word_cuts[i + 1]))
            while len(result) < target_count:
                result.append('')

            # 선택된 경계에서 실제로 어떤 보너스/유사도가 더해졌는지 요약(=기여도)
            if debug_meta_out is not None:
                try:
                    boundary_bonus_sum = 0.0
                    marker_bonus_sum = 0.0
                    boundary_bonus_hits = 0
                    marker_bonus_hits = 0
                    sim_sum = 0.0
                    sim_cnt = 0

                    for i in range(N):
                        j = word_cuts[i + 1]
                        if (j - 1) < len(words_original) and (j - 1) >= 0:
                            w_last = words_original[j - 1]
                            b = float(_boundary_end_bonus(w_last))
                            if b > 0:
                                boundary_bonus_hits += 1
                                boundary_bonus_sum += b
                            if enable_src_marker_whitespace_dp_bonus:
                                m = float(_marker_end_bonus(w_last))
                                if m > 0:
                                    marker_bonus_hits += 1
                                    marker_bonus_sum += m

                        # 유사도(선택된 span)
                        k = word_cuts[i]
                        j = word_cuts[i + 1]
                        tgt = target_vecs[i] if i < len(target_vecs) else None
                        if tgt is not None:
                            span_vec = span_cache.get((k, j))
                            if span_vec is not None:
                                sim = float(_cosine(span_vec, tgt))
                                sim_sum += sim
                                sim_cnt += 1

                    debug_meta_out["path_boundary_bonus_hits"] = int(boundary_bonus_hits)
                    debug_meta_out["path_boundary_bonus_sum"] = float(boundary_bonus_sum)
                    debug_meta_out["path_marker_bonus_hits"] = int(marker_bonus_hits)
                    debug_meta_out["path_marker_bonus_sum"] = float(marker_bonus_sum)
                    debug_meta_out["path_sim_mean"] = (float(sim_sum / sim_cnt) if sim_cnt else None)
                except Exception:
                    pass

            return restore_brackets_in_chunks(result[:target_count], bracket_insertions)
            
        except Exception as e:
            # 로깅으로 변경하여 진행바 간섭 방지
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"임베딩 기반 분할 실패, 균등 분배로 폴백: {e}")

    # 3. 폴백: 어절 기준 균등 분배 (🎯 원본 슬라이스 보존!)
    chunk_size = len(words_original) // target_count
    remainder = len(words_original) % target_count
    
    result = []
    start = 0
    for i in range(target_count):
        current_size = chunk_size + (1 if i < remainder else 0)
        end = start + current_size
        result.append(slice_segment_by_word_index(start, end))
        start = end
    
    return restore_brackets_in_chunks(result, bracket_insertions)