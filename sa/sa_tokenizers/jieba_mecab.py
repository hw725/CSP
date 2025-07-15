"""SA: 공백 기준 분할 + 분석 도구 활용 (최적화 버전)"""

import logging
import pandas as pd
from typing import List, Callable, Dict

# 🔧 필수 import 추가
try:
    import numpy as np
except ImportError:
    # 🔧 verbose 모드에서만 출력 (logger가 아직 설정되기 전이므로 나중에 처리)
    np = None

try:
    import jieba
    jieba.setLogLevel(logging.WARNING)
except ImportError:
    jieba = None

try:
    import MeCab
except ImportError:
    MeCab = None

logger = logging.getLogger(__name__)

# 기본 설정값
DEFAULT_MIN_TOKENS = 1
DEFAULT_MAX_TOKENS = 50

# MeCab 초기화 (안전하게)
mecab = None
try:
    if MeCab:
        mecabrc_path = 'C:/Users/junto/Downloads/head-repo/CSP/.venv/Lib/site-packages/mecab_ko_dic/dicdir/mecabrc'
        dicdir_path = 'C:/Users/junto/Downloads/head-repo/CSP/.venv/Lib/site-packages/mecab_ko_dic/dicdir'
        userdic_path = 'C:/Users/junto/Downloads/head-repo/CSP/.venv/Lib/site-packages/mecab_ko_dic/dicdir/user.dic'
        mecab = MeCab.Tagger(f'-r {mecabrc_path} -d {dicdir_path} -u {userdic_path}')
        # 🔧 verbose 모드에서만 출력
        if logger.isEnabledFor(logging.DEBUG):
            print("✅ MeCab 초기화 성공")
        logger.info("✅ MeCab 초기화 성공")
except Exception as e:
    if logger.isEnabledFor(logging.DEBUG):
        print(f"⚠️ MeCab 초기화 실패: {e}")
    logger.warning(f"⚠️ MeCab 초기화 실패: {e}")
    mecab = None

def split_src_meaning_units(text: str, **kwargs) -> List[str]:
    """SA 원문 분할: 무조건 공백 단위 (분석은 내부적으로만 활용)"""
    
    if not text or not text.strip():
        return []
    
    # 🎯 SA 핵심: 무조건 공백 단위로 분할
    words = text.split()
    
    # 📊 내부 분석 (분할에는 영향 안 줌, 로깅용) - 안전하게
    try:
        if jieba and logger.isEnabledFor(logging.DEBUG):
            jieba_analysis = list(jieba.cut(text))
            logger.debug(f"jieba 분석 (참고용): {jieba_analysis}")
    except:
        pass
    
    try:
        if mecab and logger.isEnabledFor(logging.DEBUG):
            mecab_analysis = _analyze_with_mecab(text)
            logger.debug(f"MeCab 분석 (참고용): {mecab_analysis[:5]}...")  # 처음 5개만
    except:
        pass
    
    logger.debug(f"SA 원문 공백 분할: {len(words)}개 어절 - {words}")
    return words

def split_tgt_meaning_units_sequential(
    src_text: str, 
    tgt_text: str, 
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    embed_func: Callable = None,
    **kwargs
) -> List[str]:
    """SA 번역문 분할: 원문 공백 단위의 의미에 맞춰 지능적으로 분할"""
    
    if not tgt_text or not tgt_text.strip():
        return []
    
    # 1. 원문 공백 단위 추출
    src_units = split_src_meaning_units(src_text)
    target_count = len(src_units)
    
    logger.debug(f"원문 {target_count}개 어절의 의미에 맞춰 번역문 분할")
    logger.debug(f"원문 단위들: {src_units}")
    
    # 2. 전각 콜론 예외 처리 (하드 경계)
    if '：' in tgt_text and target_count >= 2:
        colon_result = _handle_colon_split(tgt_text, target_count)
        if colon_result:
            logger.debug(f"전각 콜론 분할 적용: {colon_result}")
            return colon_result
    
    # 3. 번역문을 원문 의미 단위에 맞춰 지능적 분할
    if target_count == 1:
        return [tgt_text]
    
    # 4. 🎯 의미 기반 번역문 분할 (핵심 로직)
    try:
        tgt_units = _split_tgt_by_src_meanings(
            src_units, 
            tgt_text, 
            embed_func
        )
    except Exception as e:
        logger.warning(f"의미 기반 분할 실패, 균등 분할 적용: {e}")
        # 폴백: 균등 분할
        tgt_words = tgt_text.split()
        tgt_units = _distribute_words_evenly(tgt_words, target_count)
    
    logger.debug(f"의미 기반 번역문 분할 완료: {tgt_units}")
    return tgt_units

def process_single_row(row: pd.Series, row_id: str = None, **kwargs) -> List[Dict]:
    """SA 단일 행 처리: 무결성 보장 + 분석 도구 활용"""
    
    try:
        src_text = str(row.get('원문', ''))
        tgt_text = str(row.get('번역문', ''))
        
        if not src_text.strip() or not tgt_text.strip():
            return []
        
        # 🔧 임베더 함수 가져오기 (완전히 안전한 방식)
        embed_func = None
        embedder_name = kwargs.get('embedder_name', 'bge')
        
        if embedder_name and embedder_name != 'none':
            try:
                # 🔧 안전한 임베더 로드 시도
                if embedder_name == 'bge':
                    try:
                        from common.embedders.bge import get_embed_func
                        embed_func = get_embed_func()
                        logger.debug(f"BGE 임베더 로드 성공")
                    except:
                        try:
                            from common.embedders import get_embed_func
                            embed_func = get_embed_func()
                            logger.debug(f"일반 임베더 로드 성공")
                        except:
                            logger.warning(f"BGE 임베더 로드 실패")
                
                elif embedder_name == 'openai':
                    try:
                        from common.embedders.openai import get_embedder
                        embed_func = get_embedder()
                        logger.debug(f"OpenAI 임베더 로드 성공")
                    except:
                        try:
                            from common.embedders import get_embedder
                            embed_func = get_embedder('openai')
                            logger.debug(f"OpenAI 임베더 로드 성공 (대안)")
                        except:
                            logger.warning(f"OpenAI 임베더 로드 실패")
                
            except Exception as e:
                logger.warning(f"임베더 로드 실패: {e}")
        
        # 🎯 SA 핵심 처리
        # 1. 원문 = 무조건 공백 단위
        src_units = split_src_meaning_units(src_text)
        
        # 2. 번역문 = 원문에 맞춰 의미적 분할 (분석 도구 활용)
        tgt_units = split_tgt_meaning_units_sequential(
            src_text, 
            tgt_text, 
            embed_func=embed_func,
            **kwargs
        )
        
        # 3. 개수 일치 보장
        max_units = max(len(src_units), len(tgt_units))
        
        while len(src_units) < max_units:
            src_units.append('')
        while len(tgt_units) < max_units:
            tgt_units.append('')
        
        # 4. 결과 생성 - 구식별자 포함한 형식으로 출력
        results = []
        
        # row_id에서 문장식별자 추출 (안전하게)
        try:
            if row_id and '_' in row_id:
                # file_14bfb2de_chunk_0_row_1 -> 1 추출 시도
                parts = row_id.split('_')
                # 마지막 부분이 숫자인지 확인
                if parts[-1].isdigit():
                    sentence_id = int(parts[-1]) + 1  # 0-based를 1-based로 변환
                else:
                    sentence_id = getattr(row, 'name', 0) + 1
            else:
                sentence_id = getattr(row, 'name', 0) + 1
        except (ValueError, AttributeError):
            sentence_id = getattr(row, 'name', 0) + 1
        
        for i in range(max_units):
            src_unit = src_units[i]
            tgt_unit = tgt_units[i]
            
            if not src_unit.strip() and not tgt_unit.strip():
                continue
            
            result = {
                '문장식별자': sentence_id,  # 🔧 안전한 정수 추출
                '구식별자': i + 1,  # 🔧 구식별자 컬럼 추가
                '원문': src_unit,
                '번역문': tgt_unit
            }
            results.append(result)
        
        logger.debug(f"SA 행 처리 완료: {len(results)}개 단위")
        return results
        
    except Exception as e:
        logger.error(f"SA 행 처리 실패: {e}")
        
        # 오류 시에도 같은 형식으로 반환 (안전하게)
        try:
            if row_id and '_' in row_id:
                parts = row_id.split('_')
                if parts[-1].isdigit():
                    sentence_id = int(parts[-1]) + 1
                else:
                    sentence_id = getattr(row, 'name', 0) + 1
            else:
                sentence_id = getattr(row, 'name', 0) + 1
        except (ValueError, AttributeError):
            sentence_id = getattr(row, 'name', 0) + 1
        
        return [{
            '문장식별자': sentence_id,  # 🔧 안전한 정수 추출
            '구식별자': 1,  # 🔧 구식별자 컬럼 추가
            '원문': str(row.get('원문', '')),
            '번역문': str(row.get('번역문', '')),
        }]

def tokenize_text(text):
    """텍스트 토큰화"""
    if not text or not text.strip():
        return []
    return text.split()

# 🔄 호환성 함수들 (안전한 리다이렉트)
def split_tgt_by_src_units_semantic(
    src_units: List[str], 
    tgt_text: str, 
    embed_func: Callable = None, 
    **kwargs
) -> List[str]:
    """원문 단위에 따른 번역문 분할 - 의미 기반 (호환성 함수)"""
    
    logger.debug(f"호환성 함수 split_tgt_by_src_units_semantic 호출됨")
    
    try:
        # 새 함수로 리다이렉트
        src_text = ' '.join(src_units) if src_units else ''
        return split_tgt_meaning_units_sequential(
            src_text, 
            tgt_text, 
            embed_func=embed_func,
            **kwargs
        )
    except Exception as e:
        logger.error(f"호환성 함수 오류: {e}")
        # 폴백: 균등 분할
        if not tgt_text:
            return [''] * len(src_units)
        words = tgt_text.split()
        return _distribute_words_evenly(words, len(src_units))

def split_tgt_by_src_units(src_units: List[str], tgt_text: str, **kwargs) -> List[str]:
    """호환성 함수 - 이전 버전과의 호환성"""
    logger.debug("호환성 함수 split_tgt_by_src_units 호출됨")
    return split_tgt_by_src_units_semantic(src_units, tgt_text, **kwargs)

def split_tgt_meaning_units(
    tgt_text: str, 
    src_text: str = '', 
    **kwargs
) -> List[str]:
    """번역문 의미 단위 분할 (호환성 함수)"""
    logger.debug("호환성 함수 split_tgt_meaning_units 호출됨")
    return split_tgt_meaning_units_sequential(src_text, tgt_text, **kwargs)

# ===== 내부 함수들 (안전하게 구현) =====

def _split_tgt_by_src_meanings(
    src_units: List[str], 
    tgt_text: str, 
    embed_func: Callable = None
) -> List[str]:
    """원문 단위의 의미에 맞춰 번역문을 지능적으로 분할 - 통합 분석"""
    
    target_count = len(src_units)
    tgt_text = tgt_text.strip()
    tgt_words = tgt_text.split()
    
    if len(tgt_words) <= target_count:
        # 번역문 어절이 적으면 패딩
        return tgt_words + [''] * (target_count - len(tgt_words))
    
    logger.debug(f"통합 분석으로 {len(tgt_words)}개 어절을 {target_count}개 의미 단위로 분할")
    
    try:
        # 🎯 모든 도구를 동시에 사용한 종합 분석
        analysis_results = _comprehensive_analysis(
            src_units, tgt_words, embed_func
        )
        
        # 📊 분석 결과를 종합하여 최적 경계 결정
        optimal_boundaries = _determine_optimal_boundaries(
            analysis_results, target_count - 1, len(tgt_words)
        )
        
        # ✂️ 결정된 경계로 번역문 분할
        final_units = _split_by_boundaries(tgt_words, optimal_boundaries)
        
        # 🔧 목표 개수에 맞춰 조정
        if len(final_units) != target_count:
            final_units = _adjust_to_target_count(final_units, target_count)
        
        logger.debug(f"통합 분석 분할 결과: {final_units}")
        return final_units
        
    except Exception as e:
        logger.warning(f"고급 분할 실패, 균등 분할 적용: {e}")
        return _distribute_words_evenly(tgt_words, target_count)

def _comprehensive_analysis(
    src_units: List[str], 
    tgt_words: List[str], 
    embed_func: Callable = None
) -> Dict:
    """임베더 + jieba + MeCab 통합 분석 (안전 버전)"""
    
    analysis = {
        'semantic_scores': [],      # 임베더 의미 유사도
        'grammar_boundaries': [],   # MeCab 문법 경계
        'jieba_boundaries': [],     # jieba 의미 단위 경계
        'combined_scores': []       # 종합 점수
    }
    
    # 1️⃣ 임베더 의미 분석 (안전하게)
    if embed_func and np:
        try:
            analysis['semantic_scores'] = _calculate_semantic_alignment(
                src_units, tgt_words, embed_func
            )
            logger.debug(f"임베더 분석 완료: {len(analysis['semantic_scores'])}개 점수")
        except Exception as e:
            logger.warning(f"임베더 분석 실패: {e}")
            analysis['semantic_scores'] = [0.5] * max(0, len(tgt_words) - 1)
    else:
        analysis['semantic_scores'] = [0.5] * max(0, len(tgt_words) - 1)
    
    # 2️⃣ MeCab 문법 분석 (안전하게)
    if mecab:
        try:
            analysis['grammar_boundaries'] = _analyze_grammar_boundaries(tgt_words)
            logger.debug(f"MeCab 분석 완료: {len(analysis['grammar_boundaries'])}개 경계")
        except Exception as e:
            logger.warning(f"MeCab 분석 실패: {e}")
            analysis['grammar_boundaries'] = [0.3] * max(0, len(tgt_words) - 1)
    else:
        analysis['grammar_boundaries'] = [0.3] * max(0, len(tgt_words) - 1)
    
    # 3️⃣ jieba 의미 단위 분석 (안전하게)
    if jieba:
        try:
            analysis['jieba_boundaries'] = _analyze_jieba_boundaries(tgt_words)
            logger.debug(f"jieba 분석 완료: {len(analysis['jieba_boundaries'])}개 경계")
        except Exception as e:
            logger.warning(f"jieba 분석 실패: {e}")
            analysis['jieba_boundaries'] = [0.4] * max(0, len(tgt_words) - 1)
    else:
        analysis['jieba_boundaries'] = [0.4] * max(0, len(tgt_words) - 1)
    
    # 4️⃣ 종합 점수 계산 (안전하게)
    try:
        analysis['combined_scores'] = _calculate_combined_scores(
            analysis['semantic_scores'],
            analysis['grammar_boundaries'],
            analysis['jieba_boundaries']
        )
    except Exception as e:
        logger.warning(f"종합 점수 계산 실패: {e}")
        analysis['combined_scores'] = [0.4] * max(0, len(tgt_words) - 1)
    
    logger.debug(f"통합 분석 완료: 종합 점수 {len(analysis['combined_scores'])}개")
    return analysis

def _calculate_semantic_alignment(
    src_units: List[str], 
    tgt_words: List[str], 
    embed_func: Callable
) -> List[float]:
    """임베더로 각 번역문 어절이 어느 원문 단위와 가장 유사한지 분석 (안전 버전)"""
    
    if not np:
        return [0.5] * max(0, len(tgt_words) - 1)
    
    try:
        # 원문 단위들의 임베딩
        src_embeddings = []
        for src_unit in src_units:
            try:
                emb = embed_func([src_unit])
                src_embeddings.append(emb[0] if emb and len(emb) > 0 else np.zeros(768))
            except:
                src_embeddings.append(np.zeros(768))
        
        # 번역문 어절들의 임베딩과 원문과의 유사도
        word_alignments = []
        
        for word in tgt_words:
            try:
                word_emb = embed_func([word])
                word_emb = word_emb[0] if word_emb and len(word_emb) > 0 else np.zeros(768)
                
                # 각 원문 단위와의 유사도 계산
                similarities = []
                for src_emb in src_embeddings:
                    sim = np.dot(word_emb, src_emb) / (
                        np.linalg.norm(word_emb) * np.linalg.norm(src_emb) + 1e-8
                    )
                    similarities.append(sim)
                
                # 가장 유사한 원문 단위의 인덱스와 유사도
                best_match_idx = np.argmax(similarities)
                best_similarity = similarities[best_match_idx]
                
                word_alignments.append({
                    'word': word,
                    'best_src_idx': best_match_idx,
                    'similarity': best_similarity,
                    'all_similarities': similarities
                })
                
            except:
                word_alignments.append({
                    'word': word,
                    'best_src_idx': 0,
                    'similarity': 0.5,
                    'all_similarities': [0.5] * len(src_units)
                })
        
        # 경계 점수 계산: 인접한 어절들이 서로 다른 원문 단위에 매칭되면 높은 점수
        boundary_scores = []
        
        for i in range(len(tgt_words) - 1):
            curr_alignment = word_alignments[i]
            next_alignment = word_alignments[i + 1]
            
            if curr_alignment['best_src_idx'] != next_alignment['best_src_idx']:
                # 서로 다른 원문 단위에 매칭되면 경계 가능성 높음
                score = (curr_alignment['similarity'] + next_alignment['similarity']) / 2
                boundary_scores.append(max(0.1, min(0.9, score)))
            else:
                # 같은 원문 단위에 매칭되면 경계 가능성 낮음
                boundary_scores.append(0.1)
        
        return boundary_scores
        
    except Exception as e:
        logger.warning(f"의미 정렬 분석 실패: {e}")
        return [0.5] * max(0, len(tgt_words) - 1)

def _analyze_grammar_boundaries(tgt_words: List[str]) -> List[float]:
    """MeCab으로 문법적 경계 강도 분석 (안전 버전)"""
    
    boundary_scores = []
    
    for i in range(len(tgt_words) - 1):
        try:
            curr_word = tgt_words[i]
            next_word = tgt_words[i + 1]
            
            # 현재 어절의 마지막 형태소 분석
            curr_result = mecab.parse(curr_word)
            curr_pos = None
            
            for line in curr_result.split('\n'):
                if line and line != 'EOS':
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        curr_pos = parts[1].split(',')[0]
                        break
            
            # 다음 어절의 첫 형태소 분석
            next_result = mecab.parse(next_word)
            next_pos = None
            
            for line in next_result.split('\n'):
                if line and line != 'EOS':
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        next_pos = parts[1].split(',')[0]
                        break
            
            # 경계 강도 계산
            score = _calculate_grammar_boundary_strength(curr_pos, next_pos, curr_word)
            boundary_scores.append(score)
            
        except:
            boundary_scores.append(0.3)  # 기본값
    
    return boundary_scores

def _calculate_grammar_boundary_strength(curr_pos: str, next_pos: str, curr_word: str) -> float:
    """문법적 경계 강도 계산 (안전 버전)"""
    
    try:
        # 강한 경계 (절 경계, 문장 경계)
        if curr_pos in ['EF', 'EC']:  # 종결어미, 연결어미
            return 0.9
        
        if curr_pos in ['JX', 'JC']:  # 보조사, 접속조사
            return 0.8
        
        # 중간 강도 경계 (품사 변화)
        if curr_pos and next_pos and curr_pos != next_pos:
            # 명사 -> 동사, 동사 -> 명사 등의 품사 변화
            if (curr_pos.startswith('N') and next_pos.startswith('V')) or \
               (curr_pos.startswith('V') and next_pos.startswith('N')):
                return 0.6
            
            # 기타 품사 변화
            return 0.4
        
        # 약한 경계 또는 경계 없음
        return 0.2
        
    except:
        return 0.3

def _analyze_jieba_boundaries(tgt_words: List[str]) -> List[float]:
    """jieba로 의미 단위 경계 분석 (안전 버전)"""
    
    try:
        # 전체 텍스트를 jieba로 분할
        full_text = ' '.join(tgt_words)
        jieba_units = list(jieba.cut(full_text))
        jieba_units = [unit.strip() for unit in jieba_units if unit.strip()]
        
        # jieba 분할 결과를 어절 단위로 매핑
        boundary_scores = [0.4] * max(0, len(tgt_words) - 1)
        
        # 간단한 휴리스틱: jieba 단위 개수 기반 점수 조정
        jieba_ratio = len(jieba_units) / len(tgt_words) if tgt_words else 1
        
        if jieba_ratio > 1.5:  # jieba가 더 세분화했음
            for i in range(len(boundary_scores)):
                boundary_scores[i] = min(0.8, boundary_scores[i] + 0.2)
        elif jieba_ratio < 0.7:  # jieba가 덜 세분화했음
            for i in range(len(boundary_scores)):
                boundary_scores[i] = max(0.1, boundary_scores[i] - 0.2)
        
        return boundary_scores
        
    except Exception as e:
        logger.warning(f"jieba 경계 분석 실패: {e}")
        return [0.4] * max(0, len(tgt_words) - 1)

def _calculate_combined_scores(
    semantic_scores: List[float],
    grammar_scores: List[float], 
    jieba_scores: List[float]
) -> List[float]:
    """세 분석 결과를 가중 평균으로 결합 (안전 버전)"""
    
    try:
        # 가중치 설정
        semantic_weight = 0.5    # 임베더 의미 분석 50%
        grammar_weight = 0.3     # MeCab 문법 분석 30%
        jieba_weight = 0.2       # jieba 의미 단위 20%
        
        combined = []
        max_len = max(len(semantic_scores), len(grammar_scores), len(jieba_scores))
        
        for i in range(max_len):
            semantic = semantic_scores[i] if i < len(semantic_scores) else 0.5
            grammar = grammar_scores[i] if i < len(grammar_scores) else 0.3
            jieba = jieba_scores[i] if i < len(jieba_scores) else 0.4
            
            combined_score = (
                semantic * semantic_weight + 
                grammar * grammar_weight + 
                jieba * jieba_weight
            )
            
            # 점수 범위 제한
            combined_score = max(0.0, min(1.0, combined_score))
            combined.append(combined_score)
        
        return combined
        
    except Exception as e:
        logger.warning(f"종합 점수 계산 실패: {e}")
        return [0.4] * max(0, len(semantic_scores))

def _determine_optimal_boundaries(
    analysis_results: Dict, 
    needed_boundaries: int, 
    total_words: int
) -> List[int]:
    """통합 분석 결과로 최적 경계 위치 결정 (안전 버전)"""
    
    try:
        combined_scores = analysis_results.get('combined_scores', [])
        
        if needed_boundaries <= 0:
            return []
        
        if not combined_scores or len(combined_scores) < needed_boundaries:
            # 점수가 부족하면 균등 분할
            if total_words <= needed_boundaries + 1:
                return list(range(1, min(needed_boundaries + 1, total_words)))
            step = total_words // (needed_boundaries + 1)
            return [i * step for i in range(1, needed_boundaries + 1) if i * step < total_words]
        
        # 점수가 높은 위치들을 경계로 선택
        scored_positions = [(score, i + 1) for i, score in enumerate(combined_scores) if i + 1 < total_words]
        scored_positions.sort(reverse=True)  # 높은 점수 순으로 정렬
        
        # 상위 점수의 위치들 선택 (너무 가까운 경계 제외)
        selected_boundaries = []
        min_distance = max(1, total_words // (needed_boundaries + 2))  # 최소 거리
        
        for score, position in scored_positions:
            if len(selected_boundaries) >= needed_boundaries:
                break
            
            # 기존 선택된 경계들과 최소 거리 확인
            too_close = False
            for existing_pos in selected_boundaries:
                if abs(position - existing_pos) < min_distance:
                    too_close = True
                    break
            
            if not too_close and 0 < position < total_words:
                selected_boundaries.append(position)
        
        # 부족하면 균등 분할로 보완
        while len(selected_boundaries) < needed_boundaries:
            # 가장 큰 간격 찾기
            gaps = []
            all_positions = sorted(selected_boundaries + [0, total_words])
            
            for i in range(len(all_positions) - 1):
                gap_size = all_positions[i + 1] - all_positions[i]
                gap_mid = all_positions[i] + gap_size // 2
                if gap_size > 2 and gap_mid not in selected_boundaries and 0 < gap_mid < total_words:
                    gaps.append((gap_size, gap_mid))
            
            if gaps:
                gaps.sort(reverse=True)
                new_pos = gaps[0][1]
                selected_boundaries.append(new_pos)
            else:
                break
        
        return sorted(selected_boundaries[:needed_boundaries])
        
    except Exception as e:
        logger.warning(f"경계 결정 실패, 균등 분할 적용: {e}")
        if total_words <= needed_boundaries + 1:
            return list(range(1, min(needed_boundaries + 1, total_words)))
        step = total_words // (needed_boundaries + 1)
        return [i * step for i in range(1, needed_boundaries + 1) if i * step < total_words]

def _split_by_boundaries(words: List[str], boundaries: List[int]) -> List[str]:
    """결정된 경계로 어절들을 분할 (안전 버전)"""
    
    try:
        if not boundaries or not words:
            return [' '.join(words)] if words else ['']
        
        result = []
        start = 0
        
        for boundary in sorted(boundaries):
            if boundary > start and boundary <= len(words):
                segment = ' '.join(words[start:boundary])
                result.append(segment if segment else '')
                start = boundary
        
        # 마지막 부분
        if start < len(words):
            segment = ' '.join(words[start:])
            result.append(segment if segment else '')
        
        return result
        
    except Exception as e:
        logger.warning(f"경계 분할 실패: {e}")
        return [' '.join(words)] if words else ['']

def _adjust_to_target_count(units: List[str], target_count: int) -> List[str]:
    """결과를 목표 개수에 맞춰 조정 (안전 버전)"""
    
    try:
        if not units:
            return [''] * target_count
        
        if len(units) == target_count:
            return units
        elif len(units) < target_count:
            # 부족하면 빈 문자열로 패딩
            return units + [''] * (target_count - len(units))
        else:
            # 초과하면 마지막 단위들을 병합
            if target_count <= 0:
                return [' '.join(units)]
            result = units[:target_count-1]
            merged_last = ' '.join(units[target_count-1:])
            result.append(merged_last)
            return result
            
    except Exception as e:
        logger.warning(f"개수 조정 실패: {e}")
        return [''] * target_count

def _analyze_with_mecab(text: str) -> List[tuple]:
    """MeCab 분석 (내부용, 안전 버전)"""
    
    if not mecab or not text:
        return []
    
    try:
        result = mecab.parse(text)
        analysis = []
        for line in result.split('\n'):
            if line and line != 'EOS':
                parts = line.split('\t')
                if len(parts) >= 2:
                    surface = parts[0]
                    features = parts[1].split(',')
                    pos = features[0] if features else 'UNKNOWN'
                    analysis.append((surface, pos))
        return analysis
    except:
        return []

def _handle_colon_split(text: str, target_count: int) -> List[str]:
    """전각 콜론 처리 (안전 버전)"""
    
    try:
        if '：' not in text:
            return None
            
        parts = text.split('：')
        if len(parts) != 2:
            return None
        
        left_part = parts[0].strip()
        right_part = parts[1].strip()
        
        if not left_part and not right_part:
            return None
            
        left_part = left_part + '：'
        
        if target_count == 2:
            return [left_part, right_part]
        elif target_count > 2:
            # 오른쪽을 추가 분할
            right_words = right_part.split()
            remaining_count = target_count - 1
            
            if len(right_words) <= remaining_count:
                result = [left_part] + right_words
                result.extend([''] * (target_count - len(result)))
                return result
            else:
                # 오른쪽을 균등 분할
                right_splits = _distribute_words_evenly(right_words, remaining_count)
                return [left_part] + right_splits
        
        return None
        
    except Exception as e:
        logger.warning(f"콜론 분할 실패: {e}")
        return None

def _distribute_words_evenly(words: List[str], target_count: int) -> List[str]:
    """어절을 균등 분배 (안전 버전)"""
    
    try:
        if not words:
            return [''] * target_count
            
        if target_count <= 0:
            return [' '.join(words)]
        
        if target_count >= len(words):
            return words + [''] * (target_count - len(words))
        
        words_per_unit = len(words) // target_count
        remainder = len(words) % target_count
        
        result = []
        start_idx = 0
        
        for i in range(target_count):
            current_size = words_per_unit + (1 if i < remainder else 0)
            end_idx = start_idx + current_size
            
            if end_idx > len(words):
                end_idx = len(words)
            
            if start_idx < end_idx:
                result.append(' '.join(words[start_idx:end_idx]))
            else:
                result.append('')
            
            start_idx = end_idx
        
        return result
        
    except Exception as e:
        logger.warning(f"균등 분배 실패: {e}")
        return [''] * target_count

def _calculate_simple_similarity(src_text: str, tgt_text: str) -> float:
    """간단한 유사도 계산 (안전 버전)"""
    
    try:
        if not src_text or not tgt_text:
            return 0.0
            
        src_text = str(src_text).strip()
        tgt_text = str(tgt_text).strip()
        
        if not src_text or not tgt_text:
            return 0.0
        
        src_tokens = set(src_text.split())
        tgt_tokens = set(tgt_text.split())
        
        if not src_tokens or not tgt_tokens:
            return 0.0
        
        intersection = len(src_tokens & tgt_tokens)
        union = len(src_tokens | tgt_tokens)
        similarity = intersection / union if union > 0 else 0.0
        
        return round(max(0.0, min(1.0, similarity)), 3)
        
    except Exception as e:
        logger.warning(f"유사도 계산 실패: {e}")
        return 0.0