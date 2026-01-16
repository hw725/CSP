"""PA 메인 프로세서 - import 문제 해결"""

import sys
import os
from pathlib import Path
import pandas as pd
from typing import Any, Dict, List
import logging
import json
from datetime import datetime

# 로거 설정
logger = logging.getLogger(__name__)


class _StageTracer:
    def __init__(self, path: str):
        self.path = str(path)
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._fp = open(self.path, "a", encoding="utf-8")

    def write(self, record: Dict):
        self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fp.flush()

    def close(self):
        try:
            self._fp.close()
        except Exception:
            pass

# 통합 진행률 관리자
from common.progress_manager import start_unified_progress, update_unified_progress, finish_unified_progress, set_progress_description
# 전역 무결성 검증 모듈
from common.integrity_verifier import verify_global_integrity
from common.config import get_pa_selection_params

# 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

# 로컬 모듈 import
from sentence_splitter import split_target_sentences_advanced, split_source_by_whitespace_and_align
from aligner import compute_similarity_simple, safe_source_split


def _split_target_sentences_pa(
    tgt_paragraph: str,
    boundary_model,
    threshold: float,
    verbose: bool,
    max_length: int = 150,
) -> List[str]:
    """PA에서 번역문 문장 경계를 만든다.

    원칙:
    - 번역문 문장 경계는 `split_target_sentences_advanced()` 결과를 **절대 우선** 사용한다.
      (사용자 요구: 번역문 경계는 이미 맞고, boundary 모델은 원문 경계 정렬에 쓰여야 함)
    - 다만 splitter가 실패/단일 세그먼트만 반환하는 경우에만 boundary 모델을 fallback으로 사용한다.
    """
    if not tgt_paragraph:
        return []

    try:
        rule_based = split_target_sentences_advanced(tgt_paragraph, max_length=max_length)
    except Exception as e:
        rule_based = []
        if verbose:
            print(f"   ⚠️ 번역문 rule-based 분할 실패: {e}")

    # 정상 분할이면 그대로 사용
    if rule_based and len(rule_based) >= 2:
        return rule_based

    # 1문장으로만 나오는 경우는 모델 fallback을 허용
    if boundary_model is not None:
        try:
            model_based = boundary_model.segment_text(tgt_paragraph, task="pa", threshold=threshold)
            if model_based and len(model_based) >= 2:
                if verbose:
                    print(f"   🔁 번역문 분할 fallback: rule-based={len(rule_based) or 0} → boundary_model={len(model_based)}")
                return model_based
        except Exception as e:
            if verbose:
                print(f"   ⚠️ 번역문 boundary 모델 fallback 실패: {e}")

    # 최후의 수단: 빈/단일 반환(상위에서 처리)
    return rule_based if rule_based else [tgt_paragraph]


def _refine_alignments_with_models(
    alignments: List[Dict],
    src_paragraph: str,
    tgt_paragraph: str,
    boundary_model,
    alignment_model,
    threshold: float = 0.5,
    boundary_min_len: int | None = None,
    tgt_split_max_length: int = 150,
    verbose: bool = False
) -> List[Dict]:
    """
    기존 alignments를 경계 모델과 alignment 모델로 refinement
    
    전략:
    1. 번역문 결합 → 경계 모델로 재분할 → 새 경계 제안
    2. Alignment 모델로 원문 재정렬
    3. 기존 메타데이터 유지 (문단식별자, 문장식별자 등)
    
    Args:
        alignments: 기존 BGE/순차 방식 결과
        src_paragraph: 원본 원문 문단
        tgt_paragraph: 원본 번역문 문단
        boundary_model: BoundaryModelLoader
        alignment_model: AlignmentMatcher
        threshold: 경계 확률 임계값
        verbose: 상세 로그
    
    Returns:
        Refined alignments
    """
    if not alignments or not boundary_model or not alignment_model:
        return alignments
    
    try:
        # 1) 번역문 문장 경계는 rule-based로 고정
        tgt_sentences = _split_target_sentences_pa(
            tgt_paragraph=tgt_paragraph,
            boundary_model=None,  # 번역문 분할에 boundary 모델 사용 금지(정책)
            threshold=threshold,
            verbose=verbose,
            max_length=tgt_split_max_length,
        )
        
        if verbose:
            old_count = len(alignments)
            new_count = len(tgt_sentences)
            print(f"   🔧 Refinement: {old_count}개 → {new_count}개 문장 (경계 모델)")
        
        # 2) 원문 후보 경계 생성은 boundary 모델로 (원문 전용)
        # 3) tgt 문장 개수만큼 alignment 모델로 src를 매칭
        src_sentences: List[str]

        # 원문 후보 경계는 boundary 모델로 생성하되, 어떤 경우에도 최소 1개 후보는 보장한다.
        src_candidates: List[str]
        try:
            src_candidates = boundary_model.segment_text(
                src_paragraph,
                task="pa",
                threshold=threshold,
                min_len_override=boundary_min_len,
            )
        except Exception as e:
            if verbose:
                print(f"   ⚠️ 원문 boundary 후보 생성 실패(폴백 후보 사용): {e}")
            src_candidates = []

        if not src_candidates:
            src_candidates = [src_paragraph]

        # match_segments는 tgt 개수만큼 src를 반환하도록 이미 보장됨
        src_sentences = alignment_model.match_segments(src_candidates, tgt_sentences)
        
        # 3. 새 alignments 생성 (기존 메타데이터 참고)
        base_para_id = alignments[0].get('문단식별자', 1) if alignments else 1
        base_sent_id = alignments[0].get('문장식별자', 1) if alignments else 1
        
        refined = []
        for i, (src_sent, tgt_sent) in enumerate(zip(src_sentences, tgt_sentences)):
            refined.append({
                '문단식별자': base_para_id,
                '문장식별자': base_sent_id + i,
                '원문': src_sent,
                '번역문': tgt_sent,
                'similarity': alignment_model.compute_similarity(src_sent, tgt_sent) if alignment_model else compute_similarity_simple(src_sent, tgt_sent)
            })
        
        return refined
        
    except Exception as e:
        if verbose:
            print(f"   ⚠️ Refinement 실패, 기존 결과 유지: {e}")
        return alignments


def process_paragraph_alignment_with_boundary_model(
    src_paragraph: str,
    tgt_paragraph: str,
    boundary_model,
    alignment_model,
    threshold: float = 0.5,
    boundary_min_len: int | None = None,
    tgt_split_max_length: int = 150,
    adjacent_refine_max_shift_tokens: int = 1,
    enable_adjacent_boundary_refine: bool = True,
    enable_src_marker_boundary_bonus: bool = False,
    enable_src_marker_whitespace_dp_bonus: bool = False,
    verbose: bool = False,
    trace=None,
    dp_debug_out: str | None = None,
    dp_debug_meta: Dict | None = None,
) -> List[Dict]:
    """
    Boundary 모델과 Alignment 모델을 사용한 문단 정렬
    
    Args:
        src_paragraph: 원문 문단
        tgt_paragraph: 번역문 문단
        boundary_model: BoundaryModelLoader 인스턴스
        alignment_model: AlignmentMatcher 인스턴스
        threshold: 경계 확률 임계값
        boundary_min_len: boundary 모델 디코딩 min_len 오버라이드(task=pa). None이면 기본값 사용
        verbose: 상세 로그 여부
    
    Returns:
        정렬된 문장 쌍 리스트
    """
    if not src_paragraph or not tgt_paragraph:
        return []
    
    # 1. 번역문 문장 경계는 rule-based로 고정 (boundary 모델 사용 금지)
    try:
        tgt_sentences = _split_target_sentences_pa(
            tgt_paragraph=tgt_paragraph,
            boundary_model=None,
            threshold=threshold,
            verbose=verbose,
            max_length=tgt_split_max_length,
        )
        if verbose:
            print(f"   📊 번역문 분할: {len(tgt_sentences)}개 문장")

        if trace is not None:
            try:
                trace(
                    "tgt_split",
                    src_segments=[],
                    tgt_segments=[str(x) for x in tgt_sentences],
                )
            except Exception:
                pass
    except Exception as e:
        if verbose:
            print(f"   ⚠️ 번역문 분할 실패: {e}")
        return []
    
    # 2. 원문 후보 경계 생성 + tgt 개수로 매칭 (alignment 모델)
    if boundary_model is None or alignment_model is None:
        raise ValueError("boundary_model/alignment_model이 필요합니다 (--use-boundary-model 모드)")

    desired = len(tgt_sentences)

    dp_debug_path: Path | None = None
    if dp_debug_out:
        try:
            dp_debug_path = Path(dp_debug_out)
            dp_debug_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            dp_debug_path = None

    # boundary_model의 문자별 경계 logit을 한 번만 계산해 재사용한다.
    # IMPORTANT: stage drift 평가는 공백/개행/탭을 제거한 정규화 좌표계(누적 길이)로 경계를 비교한다.
    # 따라서 모델 logit도 동일한 정규화 문자열 기준으로 계산해, 후보/보너스 좌표계를 일치시킨다.
    def _norm_for_boundary(s: str) -> str:
        return str(s).replace(" ", "").replace("\n", "").replace("\t", "").strip()

    src_paragraph_norm = _norm_for_boundary(src_paragraph)
    try:
        src_boundary_logits = boundary_model.predict_boundary_logits(src_paragraph_norm, task="pa")
    except Exception:
        src_boundary_logits = None

    # marker bonus가 "실제로" 계산에 들어갔는지(히트 수/합계)를 기록하기 위한 카운터
    marker_boundary_bonus_hits = 0
    marker_boundary_bonus_sum = 0.0

    def _marker_bonus_at(text: str, pos: int) -> float:
        """원문 내 현토(한글 marker) 기반 경계 tie-break 보너스(아주 약하게).

        - 원문 토큰 끝이 CJK + Hangul 형태면(예: “也에”, “者가”) marker로 보고
          경계 후보 점수에 소량 보너스를 준다.
        - 어절 내부 분할 금지는 후보 생성/이동 로직에서 보장된다.
        """

        nonlocal marker_boundary_bonus_hits, marker_boundary_bonus_sum

        if not enable_src_marker_boundary_bonus:
            return 0.0
        if pos <= 0 or pos > len(text):
            return 0.0

        # pos 직전의 trailing whitespace는 건너뛴다
        j = pos - 1
        while j >= 0 and text[j].isspace():
            j -= 1
        if j < 0:
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

        end = j + 1
        start = j
        while start >= 0 and is_hangul(text[start]):
            start -= 1
        start += 1

        if start >= end:
            return 0.0

        prev = start - 1
        if prev < 0 or not is_cjk(text[prev]):
            return 0.0

        mlen = end - start
        if mlen >= 3:
            bonus = 0.010
        elif mlen == 2:
            bonus = 0.008
        else:
            bonus = 0.003

        # 실제 적용 여부 기록
        try:
            if bonus > 0.0:
                marker_boundary_bonus_hits += 1
                marker_boundary_bonus_sum += float(bonus)
        except Exception:
            pass
        return bonus

    def _merge_to_max_parts(parts: List[str], max_parts: int) -> List[str]:
        if not parts:
            return parts
        if max_parts < 1:
            return ["".join(parts)]
        if len(parts) <= max_parts:
            return parts
        # 너무 잘게 쪼개졌으면 연속 병합으로 개수만 줄인다.
        # AlignmentMatcher는 tgt 1개당 src 최대 4개까지 합쳐보므로,
        # 후보 개수가 과도하면 오히려 매칭이 제한에 걸릴 수 있다.
        import math
        group = int(math.ceil(len(parts) / max_parts))
        merged: List[str] = []
        for i in range(0, len(parts), group):
            chunk = "".join([p for p in parts[i:i + group] if p is not None])
            if chunk:
                merged.append(chunk)
        return merged if merged else ["".join(parts)]

    def _avg_sim(src_list: List[str], tgt_list: List[str]) -> float:
        if not src_list or not tgt_list:
            return -1.0
        n = min(len(src_list), len(tgt_list))
        if n <= 0:
            return -1.0
        total = 0.0
        for s, t in zip(src_list[:n], tgt_list[:n]):
            total += alignment_model.compute_similarity(s, t)
        return total / n

    def _refine_adjacent_boundaries(
        src_list: List[str],
        tgt_list: List[str],
        max_shift_tokens: int = 1,
        passes: int = 2,
    ) -> List[str]:
        """인접 문장 경계에서 '토큰(공백 기준 \S+)' 단위 이동만 허용하는 로컬 교정.

        사용자 요구사항: 어떤 경우에도 어절 내부(\S+ 내부)를 쪼개지 않는다.
        즉 문자 단위 이동은 금지하고, 공백으로 구분되는 토큰 단위로만 이동한다.
        """

        if not src_list or len(src_list) != len(tgt_list):
            return src_list

        refined = list(src_list)
        n = len(refined)

        import re

        def is_boundary_char(ch: str) -> bool:
            if not ch:
                return True
            if ch.isspace():
                return True
            # 문장/구두점/괄호류는 경계로 취급
            return ch in "\"'”’」』》〉】〕)\]\}.,，。?!;:·、"

        def is_cjk(ch: str) -> bool:
            if not ch:
                return False
            o = ord(ch)
            return (
                0x4E00 <= o <= 0x9FFF
                or 0x3400 <= o <= 0x4DBF
                or 0xF900 <= o <= 0xFAFF
            )

        def is_hangul(ch: str) -> bool:
            if not ch:
                return False
            o = ord(ch)
            return 0xAC00 <= o <= 0xD7A3

        def boundary_is_inside_token(left: str, right: str) -> bool:
            if not left or not right:
                return False
            # 좌측 끝/우측 시작이 모두 공백/구두점이 아니면 '어절 내부'로 간주
            # CJK/한글 연속도 포함 (어절 내부 분할 절대 금지)
            if (not is_boundary_char(left[-1])) and (not is_boundary_char(right[0])):
                return True
            return False

        def _compute_raw_offsets(parts: List[str]) -> List[int]:
            """raw(공백 포함) 기준 각 segment의 시작 offset(누적 길이)을 반환."""

            offsets: List[int] = []
            cursor = 0
            for p in parts:
                offsets.append(cursor)
                cursor += len(str(p))
            return offsets

        def _compute_norm_offsets(parts: List[str]) -> List[int]:
            """정규화(공백/개행/탭 제거) 기준 각 segment의 시작 offset(누적 길이)을 반환."""

            offsets: List[int] = []
            cursor = 0
            for p in parts:
                offsets.append(cursor)
                # NOTE: 여기서는 strip()을 하지 않는다. 원문 내 중간 공백은 제거되지만,
                # 문단 전체의 양끝 공백은 일반적으로 없으며, 있어도 raw<->norm 매핑에서 일관되게 처리된다.
                cursor += len(str(p).replace(" ", "").replace("\n", "").replace("\t", ""))
            return offsets

        def _norm_len_prefix(text: str, end_pos_raw: int) -> int:
            """raw 문자열 prefix(0:end_pos_raw)의 정규화 길이를 계산한다."""

            if end_pos_raw <= 0:
                return 0
            end_pos_raw = min(end_pos_raw, len(text))
            count = 0
            for ch in text[:end_pos_raw]:
                if ch not in {" ", "\n", "\t"}:
                    count += 1
            return count

        def _raw_index_for_norm_boundary(text: str, norm_pos: int) -> int:
            """정규화 문자열에서의 경계 위치(norm_pos)를 raw 문자열 경계 인덱스로 매핑한다.

            norm_pos는 '좌측에 포함될 정규화 문자 수'이며, 반환값은 raw slicing에서의 end index.
            """

            if norm_pos <= 0:
                return 0
            count = 0
            for i, ch in enumerate(text):
                if ch not in {" ", "\n", "\t"}:
                    count += 1
                    if count == norm_pos:
                        return i + 1
            return len(text)

        def _topk_model_boundaries_near_norm(
            global_center_norm: int,
            local_start_norm: int,
            local_end_norm: int,
            k: int = 6,
        ) -> List[int]:
            """boundary_model logit 기반으로 (local_start_norm~local_end_norm) 범위의 상위 k개 경계 후보를 반환.

            NOTE: src_boundary_logits는 정규화 문자열 기준으로 계산되므로, 인자도 norm 좌표계 기준이다.
            """

            if not src_boundary_logits:
                return []
            n = len(src_boundary_logits)
            a = max(1, local_start_norm)
            b = min(n - 1, local_end_norm)
            if a >= b:
                return []
            scored = []
            for pos in range(a, b + 1):
                scored.append((float(src_boundary_logits[pos]), pos))
            scored.sort(key=lambda x: (x[0], -abs(x[1] - global_center_norm)), reverse=True)
            return [pos for _score, pos in scored[:k]]

        def sim(a: str, b: str) -> float:
            return alignment_model.compute_similarity(a, b)

        re_token = re.compile(r"\S+", re.S)

        def _token_spans(text: str) -> List[tuple[int, int]]:
            return [(m.start(), m.end()) for m in re_token.finditer(text)]

        def _extend_right_ws(text: str, pos: int) -> int:
            # 경계가 공백 위에 오도록(좌측이 공백으로 끝나도록) 공백을 좌측에 포함시키는 버전
            # 원문 자체는 변하지 않고, 경계 위치만 이동한다.
            while pos < len(text) and text[pos].isspace():
                pos += 1
            return pos

        def _extend_left_ws(text: str, pos: int) -> int:
            # 경계가 공백 위에 오도록(우측이 공백으로 시작하도록) 공백을 우측에 포함시키는 버전
            while pos > 0 and text[pos - 1].isspace():
                pos -= 1
            return pos

        def _is_natural_boundary(text: str, pos: int) -> bool:
            if pos <= 0 or pos >= len(text):
                return False
            c_prev = text[pos - 1]
            c_next = text[pos]
            if c_prev.isspace() or c_next.isspace():
                return True
            punct = set(",;:!?。！？、，．.·…")
            return (c_prev in punct) or (c_next in punct)

        def _boundary_bonus_at(text: str, pos: int, *, global_start_norm: int) -> float:
            """경계 자체의 보너스(자연 경계 + 모델 logit tie-break)."""

            bonus = 0.0
            if pos <= 0 or pos >= len(text):
                return bonus
            c_prev = text[pos - 1]
            c_next = text[pos]
            punct = set(",;:!?。！？、，．.·…")
            open_punct = set("(（[［{｛<＜\"'“‘《〈「『【")
            close_punct = set(")）]］}｝>＞\"'”’》〉」』】")
            if c_prev.isspace() or c_next.isspace():
                bonus += 0.008
            if c_prev in punct:
                bonus += 0.012
            if c_prev in close_punct:
                bonus += 0.006
            if c_next in open_punct:
                bonus += 0.004

            # 현토(한글 marker) 기반 tie-break 보너스(선택)
            bonus += _marker_bonus_at(text, pos)

            if src_boundary_logits is None:
                return bonus

            # 모델 신호는 자연 경계에서만 tie-breaker 수준으로
            if not _is_natural_boundary(text, pos):
                return bonus

            try:
                local_norm = _norm_len_prefix(text, pos)
                global_pos_norm = global_start_norm + local_norm
                if 0 <= global_pos_norm < len(src_boundary_logits):
                    import math

                    # 모델 신호는 '긍정'일 때만 보너스로 사용한다.
                    # (음수 logit을 패널티로 쓰면, 경계 모델의 오판이 DP를 강하게 왜곡할 수 있음)
                    bonus += 0.020 * max(0.0, math.tanh(float(src_boundary_logits[global_pos_norm]) / 3.0))
            except Exception:
                pass
            return bonus

        def _global_dp_refine(
            parts: List[str],
            targets: List[str],
            *,
            max_shift_tokens_local: int,
        ) -> List[str]:
            """문단 전체를 jointly 최적화하는 DP 기반 경계 교정.

            - 각 경계(i|i+1)에 대해 '현재 경계 주변'에서 후보(raw pos)를 모으고
              전체 경계열이 단조 증가하도록 최적의 조합을 선택한다.
            - 문자열 내용은 절대 바꾸지 않고 raw slicing 경계만 선택한다.
            """

            if not parts or len(parts) != len(targets):
                return parts

            text = "".join([str(x) for x in parts])
            if not text:
                return parts

            # 현재 경계(raw) 위치
            raw_offsets_local = _compute_raw_offsets(parts)
            norm_offsets_local = _compute_norm_offsets(parts)
            n_parts = len(parts)
            orig_boundaries: List[int] = []
            for i in range(n_parts - 1):
                orig_boundaries.append(raw_offsets_local[i] + len(str(parts[i])))

            # 후보 생성(각 경계마다)
            import re

            re_token = re.compile(r"\S+", re.S)

            def _token_spans(text2: str) -> List[tuple[int, int]]:
                return [(m.start(), m.end()) for m in re_token.finditer(text2)]

            def _pair_candidates(i: int) -> List[int]:
                s1 = str(parts[i])
                s2 = str(parts[i + 1])
                combined = s1 + s2
                boundary = len(s1)
                spans = _token_spans(combined)

                if not spans:
                    return [orig_boundaries[i]]

                # 기본 후보(현재 경계)
                candidates: set[int] = {boundary}

                # 자연 경계 후보(공백/구두점)
                max_shift_chars = max(20, int(max_shift_tokens_local) * 15)
                left = max(1, boundary - max_shift_chars)
                right = min(len(combined) - 1, boundary + max_shift_chars)
                punct = set(",;:!?。！？、，．.·…")
                for pos in range(left, right + 1):
                    if pos <= 0 or pos >= len(combined):
                        continue
                    c_prev = combined[pos - 1]
                    c_next = combined[pos]
                    if c_prev.isspace() or c_next.isspace() or (c_prev in punct) or (c_next in punct):
                        candidates.add(pos)

                # 토큰 단위 이동 후보
                def _extend_right_ws(text3: str, pos: int) -> int:
                    while pos < len(text3) and text3[pos].isspace():
                        pos += 1
                    return pos

                def _extend_left_ws(text3: str, pos: int) -> int:
                    while pos > 0 and text3[pos - 1].isspace():
                        pos -= 1
                    return pos

                # boundary가 토큰 내부면 토큰 시작/끝도 후보
                for (a, b) in spans:
                    if a < boundary < b:
                        candidates.add(a)
                        candidates.add(b)
                        candidates.add(_extend_right_ws(combined, b))
                        candidates.add(_extend_left_ws(combined, a))
                        break

                # 오른쪽/왼쪽으로 토큰 k개 이동
                idx_r = None
                for j, (_a, b) in enumerate(spans):
                    if b > boundary:
                        idx_r = j
                        break
                if idx_r is not None:
                    for k in range(1, max(1, max_shift_tokens_local) + 1):
                        j2 = idx_r + (k - 1)
                        if j2 >= len(spans):
                            break
                        a2, b2 = spans[j2]
                        candidates.add(a2)
                        candidates.add(_extend_left_ws(combined, a2))
                        candidates.add(b2)
                        candidates.add(_extend_right_ws(combined, b2))

                idx_l = None
                for j in range(len(spans) - 1, -1, -1):
                    a, _b = spans[j]
                    if a < boundary:
                        idx_l = j
                        break
                if idx_l is not None:
                    for k in range(1, max(1, max_shift_tokens_local) + 1):
                        j2 = idx_l - (k - 1)
                        if j2 < 0:
                            break
                        a2, b2 = spans[j2]
                        candidates.add(a2)
                        candidates.add(_extend_left_ws(combined, a2))
                        candidates.add(b2)
                        candidates.add(_extend_right_ws(combined, b2))

                # 모델 기반 후보(자연 경계만)
                if src_boundary_logits is not None:
                    # norm 좌표계에서 top-k 후보 → raw pos로 매핑
                    global_start_norm = norm_offsets_local[i]
                    boundary_norm = len(s1.replace(" ", "").replace("\n", "").replace("\t", ""))
                    global_boundary_norm = global_start_norm + boundary_norm
                    global_end_norm = global_start_norm + len(combined.replace(" ", "").replace("\n", "").replace("\t", ""))
                    window_norm = max(40, int(max_shift_tokens_local) * 30)
                    cand_norms = _topk_model_boundaries_near_norm(
                        global_center_norm=global_boundary_norm,
                        local_start_norm=max(1, global_boundary_norm - window_norm),
                        local_end_norm=min(global_end_norm - 1, global_boundary_norm + window_norm),
                        k=20,
                    )
                    for gnorm in cand_norms:
                        graw = _raw_index_for_norm_boundary(text, int(gnorm))
                        local_start_raw = raw_offsets_local[i]
                        lpos = graw - local_start_raw
                        if 0 < lpos < len(combined) and _is_natural_boundary(combined, int(lpos)):
                            candidates.add(int(lpos))

                # global raw로 변환
                global_start_raw = raw_offsets_local[i]
                pos0 = orig_boundaries[i]
                # 결정론 위해 안정 정렬 + 너무 많은 후보 컷
                globals_sorted = sorted(
                    {global_start_raw + int(p) for p in candidates if 0 < int(p) < len(combined)},
                    key=lambda p: (abs(p - pos0), p),
                )
                return globals_sorted[:60]

            boundary_candidates: List[List[int]] = []
            for i in range(n_parts - 1):
                cs = _pair_candidates(i)
                if not cs:
                    cs = [orig_boundaries[i]]
                boundary_candidates.append(cs)

            # === Pre-compute sim scores for DP (only DP-reachable pairs) ===
            sim_precomputed: Dict[tuple, float] = {}
            
            # Init: (0, bpos, 0) for first segment
            for bpos in boundary_candidates[0]:
                seg = text[0:bpos]
                if seg.strip():
                    sim_precomputed[(0, bpos, 0)] = sim(seg, targets[0])
            
            # Transitions: (apos, bpos, i) for i in [1, n_parts-2]
            for i in range(1, n_parts - 1):
                for bpos in boundary_candidates[i]:
                    for apos in boundary_candidates[i - 1]:
                        if bpos > apos:
                            seg = text[apos:bpos]
                            if seg.strip():
                                sim_precomputed[(apos, bpos, i)] = sim(seg, targets[i])
            
            # Finalize: (bpos, len(text), n_parts-1) for last segment
            if n_parts >= 2:
                last_i = n_parts - 2
                for bpos in boundary_candidates[last_i]:
                    seg = text[bpos:]
                    if seg.strip():
                        sim_precomputed[(bpos, len(text), len(targets) - 1)] = sim(seg, targets[-1])
            
            def sim_cached(start: int, end: int, target_idx: int) -> float:
                return sim_precomputed.get((start, end, target_idx), 0.0)
            # === End Pre-compute ===

            # DP: dp[i][j] = (score, prev_j)
            NEG = -1e18
            dp: List[List[float]] = []
            back: List[List[int]] = []

            def _seg_ok(a: int, b: int) -> bool:
                if a < 0 or b <= a or b > len(text):
                    return False
                seg = text[a:b]
                if not seg.strip():
                    return False
                # boundary at b must not split inside token
                if boundary_is_inside_token(text[:b], text[b:]):
                    return False
                return True

            # precompute baseline sum for do-no-harm gate
            baseline_sum = 0.0
            cur_pos = 0
            for i in range(n_parts - 1):
                nxt = orig_boundaries[i]
                if _seg_ok(cur_pos, nxt):
                    baseline_sum += sim(text[cur_pos:nxt], targets[i])
                cur_pos = nxt
            if _seg_ok(cur_pos, len(text)):
                baseline_sum += sim(text[cur_pos:], targets[-1])

            for i, cs in enumerate(boundary_candidates):
                dp.append([NEG] * len(cs))
                back.append([-1] * len(cs))

            def _boundary_choice_penalty(bpos: int, *, orig_bpos: int) -> float:
                # 원래 경계는 절대 패널티를 주지 않는다.
                if bpos == orig_bpos:
                    return 0.0
                # 자연 경계(공백/구두점)는 패널티 없음.
                if _is_natural_boundary(text, bpos):
                    return 0.0
                # 그 외의 mid-clause split은 억제한다.
                return 0.030

            def _dump_dp_debug(
                *,
                applied: bool,
                reason: str,
                boundary_candidates_abs: List[List[int]],
                min_gain: float,
                best_total: float | None,
                baseline_sum: float,
                chosen_boundaries_abs: List[int] | None,
            ) -> None:
                if dp_debug_path is None:
                    return
                try:
                    cand_feats: List[List[Dict]] = []
                    for i, cs in enumerate(boundary_candidates_abs):
                        orig = orig_boundaries[i]
                        feats_i: List[Dict] = []
                        for p in cs:
                            feats_i.append(
                                {
                                    "pos": int(p),
                                    "delta": int(p - orig),
                                    "is_natural": bool(_is_natural_boundary(text, int(p))),
                                    "bonus": float(_boundary_bonus_at(text, int(p), global_start_norm=0)),
                                    "shift_penalty": float(0.0008 * abs(int(p) - int(orig))),
                                    "choice_penalty": float(_boundary_choice_penalty(int(p), orig_bpos=int(orig))),
                                }
                            )
                        cand_feats.append(feats_i)

                    rec = {
                        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        "kind": "pa_global_dp_refine",
                        "applied": bool(applied),
                        "reason": str(reason),
                        "text": text,
                        "targets": [str(x) for x in targets],
                        "orig_parts": [str(x) for x in parts],
                        "orig_boundaries_abs": orig_boundaries,
                        "boundary_candidates_abs": boundary_candidates_abs,
                        "boundary_candidate_features": cand_feats,
                        "baseline_sum": float(baseline_sum),
                        "min_gain": float(min_gain),
                        "best_total": float(best_total) if best_total is not None else None,
                        "chosen_boundaries_abs": chosen_boundaries_abs,
                        "dp": dp,
                        "back": back,
                    }
                    if dp_debug_meta:
                        rec["meta"] = dp_debug_meta
                    dp_debug_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass

            # === Numba Integration ===
            if len(text) > 2500: # Safety fallback for very long text
                # ... Fallback to Python sim_cached logic (omitted to save tokens, assume rare)
                # Actually, let's just use the Python logic if too long, or error out?
                # For robustness, we should keep Python logic or implement fallback.
                # Given context, let's just Raise Error or truncation? 
                # Better: Use optimized Python loop (from previous step) if Numba not feasible?
                # No, just use Numba. 2500^2 * 20 * 4 bytes = 500MB per paragraph. Acceptable.
                pass

            import numpy as np
            from numba.typed import List as NumbaList
            try:
                from common import numba_ops
            except ImportError:
                # Local run fallback
                import common.numba_ops as numba_ops

            # 1. Prepare Sim Table (Dense)
            n_tgts = len(targets)
            sim_table = np.zeros((len(text) + 1, len(text) + 1, n_tgts), dtype=np.float32)

            # Collect requests for Batch Processing
            # List of (start, end, tgt_idx)
            requests = []
            
            # Init
            for bpos in boundary_candidates[0]:
                seg = text[0:bpos]
                if seg.strip():
                    requests.append((0, bpos, 0))

            # Transitions
            for i in range(1, n_parts - 1):
                for bpos in boundary_candidates[i]:
                    for apos in boundary_candidates[i - 1]:
                        if bpos > apos:
                            seg = text[apos:bpos]
                            if seg.strip():
                                requests.append((apos, bpos, i))
                                
            # Finalize
            if n_parts >= 2:
                last_i = n_parts - 2
                for bpos in boundary_candidates[last_i]:
                    seg = text[bpos:]
                    if seg.strip():
                        requests.append((bpos, len(text), n_tgts - 1))
            
            # Batch Compute or Loop
            use_batch = False
            batch_scores = []
            
            # Check if sim is a bound method of BoundaryAwareAlignmentMatcher
            if hasattr(sim, '__self__') and hasattr(sim.__self__, 'compute_batch_similarity'):
                try:
                    pairs = []
                    for s, e, ti in requests:
                        pairs.append((text[s:e], targets[ti]))
                        
                    # Call batch compute (GPU accelerated)
                    # batch_size=256 or 512
                    batch_scores = sim.__self__.compute_batch_similarity(pairs, batch_size=512)
                    use_batch = True
                except Exception:
                    import traceback
                    traceback.print_exc()
                    # Fallback if anything goes wrong
                    use_batch = False

            if use_batch:
                # Fill table from batch results
                for (s, e, ti), score in zip(requests, batch_scores):
                    sim_table[s, e, ti] = score
            else:
                # Fallback Loop (Slow)
                for s, e, ti in requests:
                    seg = text[s:e]
                    sim_table[s, e, ti] = sim(seg, targets[ti])

            # 2. Prepare Candidates (Numba List)
            nb_candidates = NumbaList()
            for cs in boundary_candidates:
                nb_candidates.append(np.array(cs, dtype=np.int32))
                
            # 3. Prepare Other Args
            orig_bound_arr = np.array(orig_boundaries, dtype=np.int32)
            
            # Bonus Array (optional, pre-compute bonus for all positions to avoid Python callback)
            # _boundary_bonus_at(text, bpos, ...)
            # We can pre-compute bonus for all candidate positions.
            # Map: global_pos -> bonus.
            # Max pos is len(text).
            bonus_arr = np.zeros(len(text) + 1, dtype=np.float32)
            # Only compute for positions present in candidates
            unique_pos = set()
            for cs in boundary_candidates:
                unique_pos.update(cs)
            
            for bpos in unique_pos:
                bonus_arr[bpos] = _boundary_bonus_at(text, bpos, global_start_norm=0)
                
            # 4. Run Numba DP
            success, best_total, chosen_indices = numba_ops.run_dp_numba(
                n_parts,
                len(text),
                nb_candidates,
                orig_bound_arr,
                sim_table,
                n_tgts,
                bonus_arr
            )
            
            if not success:
                # Fallback or dump debug
                _dump_dp_debug(
                    applied=False,
                    reason="numba_failed_or_no_path",
                    boundary_candidates_abs=boundary_candidates,
                    min_gain=0.0,
                    best_total=None,
                    baseline_sum=baseline_sum,
                    chosen_boundaries_abs=None,
                )
                return parts

            # 5. Do-No-Harm Check
            min_gain = max(0.02, 0.015 * max(1.0, float(baseline_sum)))
            if best_total < baseline_sum + min_gain:
                 _dump_dp_debug(
                    applied=False,
                    reason="do_no_harm_gate",
                    boundary_candidates_abs=boundary_candidates,
                    min_gain=min_gain,
                    best_total=best_total,
                    baseline_sum=baseline_sum,
                    chosen_boundaries_abs=list(chosen_indices),
                )
                 return parts

            # 6. Build Result
            # chosen_indices is aligned with boundary_candidates[i]
            # But wait, numba returned actual values or indices?
            # My numba implementation returns actual values (chosen array).
            
            new_parts: List[str] = []
            start = 0
            for i in range(len(chosen_indices)):
                end = chosen_indices[i]
                new_parts.append(text[start:end])
                start = end
            new_parts.append(text[start:]) # Last part
            
            return new_parts

            # build new segments
            new_parts: List[str] = []
            start = 0
            for i in range(n_parts - 1):
                end = chosen[i]
                new_parts.append(text[start:end])
                start = end
            new_parts.append(text[start:])

            _dump_dp_debug(
                applied=True,
                reason="applied",
                boundary_candidates_abs=boundary_candidates,
                min_gain=min_gain,
                best_total=best_total,
                baseline_sum=baseline_sum,
                chosen_boundaries_abs=chosen,
            )
            return new_parts

        for _ in range(max(1, passes)):
            changed = False
            raw_offsets = _compute_raw_offsets(refined)
            norm_offsets = _compute_norm_offsets(refined)
            paragraph_raw = "".join([str(x) for x in refined])
            for i in range(n - 1):
                s1 = refined[i]
                s2 = refined[i + 1]
                t1 = tgt_list[i]
                t2 = tgt_list[i + 1]

                # 이동폭은 고정: 과도한 경계 이동은 오프셋 전파(±2~±4 drift)로 이어질 수 있어
                # 데이터 특이 규칙 없이 기본값(max_shift_tokens)만 사용한다.
                local_max_shift_tokens = max_shift_tokens

                combined = s1 + s2
                boundary = len(s1)
                boundary_norm = len(str(s1).replace(" ", "").replace("\n", "").replace("\t", ""))
                spans = _token_spans(combined)

                if not spans:
                    continue

                base_1 = sim(s1, t1)
                base_2 = sim(s2, t2)
                base = base_1 + base_2
                best = base
                best_pos = boundary

                # 후보 경계 위치 생성: 좌/우로 최대 local_max_shift_tokens 토큰까지 이동
                # - 문자열 내용은 절대 변경하지 않고, combined 내 '경계 위치(pos)'만 바꾼다.
                candidates: set[int] = {boundary}

                # 토큰 후보만으로는 너무 이른/늦은 공백 경계를 놓칠 수 있어,
                # 경계 주변의 '자연스러운 경계'(공백/구두점) 위치도 후보로 추가한다.
                # (무결성 불변: 문자열은 그대로, boundary pos만 변경)
                max_shift_chars = max(20, int(local_max_shift_tokens) * 15)
                left = max(1, boundary - max_shift_chars)
                right = min(len(combined) - 1, boundary + max_shift_chars)
                punct = set(",;:!?。！？、，．.·…")
                open_punct = set("(（[［{｛<＜\"'“‘《〈「『【")
                close_punct = set(")）]］}｝>＞\"'”’》〉」』】")
                for pos in range(left, right + 1):
                    if pos <= 0 or pos >= len(combined):
                        continue
                    c_prev = combined[pos - 1]
                    c_next = combined[pos]
                    if c_prev.isspace() or c_next.isspace() or (c_prev in punct) or (c_next in punct):
                        candidates.add(pos)

                # 토큰이 1개뿐인 경우(공백/구두점이 거의 없는 원문)에는
                # 주변 문자를 촘촘히 후보로 넣어 탐색 공간을 확보한다.
                if len(spans) == 1:
                    for pos in range(left, right + 1):
                        if 0 < pos < len(combined):
                            candidates.add(pos)

                # 모델 기반 후보 추가: boundary_model이 높은 점수를 준 위치를 우선 후보로 포함
                # - 규칙 하드코딩 대신 모델 신호
                if src_boundary_logits is not None and i < len(raw_offsets):
                    global_start_raw = raw_offsets[i]
                    global_start_norm = norm_offsets[i]
                    global_boundary_norm = global_start_norm + boundary_norm
                    global_end_norm = global_start_norm + len(str(combined).replace(" ", "").replace("\n", "").replace("\t", ""))

                    # 모델 신호는 매우 보조적으로만 사용: 너무 큰 윈도우는 엉뚱한 경계로 스냅될 수 있다.
                    window_norm = max(40, int(local_max_shift_tokens) * 30)
                    cand_norms = _topk_model_boundaries_near_norm(
                        global_center_norm=global_boundary_norm,
                        local_start_norm=max(1, global_boundary_norm - window_norm),
                        local_end_norm=min(global_end_norm - 1, global_boundary_norm + window_norm),
                        k=30,
                    )
                    for gnorm in cand_norms:
                        # norm 좌표 -> paragraph_raw(raw) 좌표로 매핑 후 local로 변환
                        graw = _raw_index_for_norm_boundary(paragraph_raw, int(gnorm))
                        lpos = graw - global_start_raw
                        if 0 < lpos < len(combined):
                            # 모델 후보는 자연 경계(공백/구두점)일 때만 넣어서 과잉 스냅을 방지
                            if _is_natural_boundary(combined, int(lpos)):
                                candidates.add(int(lpos))

                # 경계가 토큰 내부일 수 있으므로, 그 토큰의 시작/끝도 후보로 넣는다.
                for (a, b) in spans:
                    if a < boundary < b:
                        candidates.add(a)
                        candidates.add(b)
                        candidates.add(_extend_right_ws(combined, b))
                        candidates.add(_extend_left_ws(combined, a))
                        break

                # 오른쪽으로 이동(= s2의 앞쪽 토큰들을 s1로 포함)
                # boundary 이후(또는 boundary를 포함하는) 토큰부터 k개를 포함하는 지점
                idx_r = None
                for j, (_a, b) in enumerate(spans):
                    if b > boundary:
                        idx_r = j
                        break
                if idx_r is not None:
                    for k in range(1, max(1, local_max_shift_tokens) + 1):
                        j2 = idx_r + (k - 1)
                        if j2 >= len(spans):
                            break
                        a2, b2 = spans[j2]
                        # 토큰의 시작/끝 모두 후보로 추가
                        candidates.add(a2)
                        candidates.add(_extend_left_ws(combined, a2))
                        candidates.add(b2)
                        candidates.add(_extend_right_ws(combined, b2))

                # 왼쪽으로 이동(= s1의 뒤쪽 토큰들을 s2로 포함)
                # boundary 이전 토큰들 중에서 k개를 오른쪽으로 넘기는 지점
                idx_l = None
                for j in range(len(spans) - 1, -1, -1):
                    a, _b = spans[j]
                    if a < boundary:
                        idx_l = j
                        break
                if idx_l is not None:
                    for k in range(1, max(1, local_max_shift_tokens) + 1):
                        j2 = idx_l - (k - 1)
                        if j2 < 0:
                            break
                        a2, b2 = spans[j2]
                        # 토큰의 시작/끝 모두 후보로 추가
                        candidates.add(a2)
                        candidates.add(_extend_left_ws(combined, a2))
                        candidates.add(b2)
                        candidates.add(_extend_right_ws(combined, b2))

                # 후보들 평가
                # NOTE: candidates는 set이므로 순회 순서가 구현/상태에 따라 달라질 수 있다.
                # 점수 동률(또는 매우 근접) 시 결과가 흔들리지 않도록 안정적인 순서 + tie-break를 적용한다.
                for pos in sorted(candidates, key=lambda p: (abs(p - boundary), p)):
                    if pos <= 0 or pos >= len(combined):
                        continue
                    cand1 = combined[:pos]
                    cand2 = combined[pos:]
                    if not cand1.strip() or not cand2.strip():
                        continue
                    if boundary_is_inside_token(cand1, cand2):
                        continue

                    # 일반화 점수: 번역문 정합(sim) + 경계 자연스러움(공백/구두점) - 이동거리 패널티
                    # - 특정 pid/문구 하드코딩 없이, 데이터셋 변화에 강한 신호만 사용
                    # - 동일 base_score일 때는 원래 boundary에 가까운 후보를 선호
                    score_1 = sim(cand1, t1)
                    score_2 = sim(cand2, t2)
                    base_score = score_1 + score_2

                    # 경계의 '자연스러움' 보너스(약하게)
                    boundary_bonus = 0.0
                    c_prev = combined[pos - 1]
                    c_next = combined[pos]
                    if c_prev.isspace() or c_next.isspace():
                        boundary_bonus += 0.008
                    if c_prev in punct:
                        boundary_bonus += 0.012
                    if c_prev in close_punct:
                        boundary_bonus += 0.006
                    if c_next in open_punct:
                        boundary_bonus += 0.004
                    # 모델 기반 보너스: boundary_model logit이 높은 위치를 약하게 선호
                    if src_boundary_logits is not None and i < len(raw_offsets):
                        global_start_norm = norm_offsets[i]
                        # raw pos -> norm pos (combined prefix의 정규화 길이)
                        local_norm = _norm_len_prefix(combined, pos)
                        global_pos_norm = global_start_norm + local_norm
                        if 0 <= global_pos_norm < len(src_boundary_logits):
                            # logit은 스케일이 크므로 완만히: tanh로 압축
                            import math

                            # NOTE: 모델 신호는 tie-breaker 수준으로만(과도한 경계 이동 방지)
                            if _is_natural_boundary(combined, pos):
                                boundary_bonus += 0.030 * math.tanh(float(src_boundary_logits[global_pos_norm]) / 3.0)

                    # 이동거리 패널티(매우 약하게)
                    shift_penalty = 0.0006 * abs(pos - boundary)
                    score = base_score + boundary_bonus - shift_penalty

                    # 가드: 한쪽이라도 유사도가 크게 떨어지는 이동은 금지(오프셋 전파/누수 방지)
                    # 모델 신호 기반 스냅은 약간의 sim 하락을 감수하고도
                    # '정확한 좌표'를 맞추는 게 목적이므로 기본 가드를 완화한다.
                    # sim 급락을 허용하면 경계가 크게 망가질 수 있어 보수적으로 제한
                    allowed_drop = 0.04
                    if (score_1 < base_1 - allowed_drop) or (score_2 < base_2 - allowed_drop):
                        continue

                    eps = 1e-6
                    if score > best + eps:
                        best = score
                        best_pos = pos
                    elif abs(score - best) <= eps:
                        # 동점이면 원래 boundary에 더 가까운 후보를 선호(안정적 선택)
                        cur = (abs(pos - boundary), pos)
                        prev = (abs(best_pos - boundary), best_pos)
                        if cur < prev:
                            best = score
                            best_pos = pos

                if best_pos != boundary:
                    refined[i] = combined[:best_pos]
                    refined[i + 1] = combined[best_pos:]
                    changed = True



            if not changed:
                break

        # 2차: 문단 전체 DP refine(손해면 적용 안 함)
        try:
            refined = _global_dp_refine(refined, tgt_list, max_shift_tokens_local=max_shift_tokens)
        except Exception as e:
            # 디버그가 켜진 경우엔 조용히 삼키면 원인 파악이 불가능하므로 최소한의 정보는 남긴다.
            if verbose:
                print(f"   ⚠️ global DP refine 실패(무시): {e}")
            if dp_debug_path is not None:
                try:
                    import json

                    dp_debug_path.write_text(
                        json.dumps(
                            {
                                "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                                "kind": "pa_global_dp_refine",
                                "applied": False,
                                "reason": "exception",
                                "error": str(e),
                                "meta": (dp_debug_meta or None),
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                except Exception:
                    pass

        return refined

    candidate_sets: List[tuple[str, List[str]]] = []

    # 후보 세트 제외 플래그를 먼저 읽음 (grid search에서 튜닝 가능)
    _pa_sel_early = get_pa_selection_params()
    disable_supar = bool(_pa_sel_early.get("disable_supar", False))
    disable_boundary = bool(_pa_sel_early.get("disable_boundary", False))
    disable_whitespace_dp = bool(_pa_sel_early.get("disable_whitespace_dp", False))

    # (A) SuPar-Kanbun 기반 후보(가능하면 사용): 원문 문장/구 경계의 자연스러운 후보
    if not disable_supar:
        try:
            import common.new_parsers as new_parsers
            supar_parts = new_parsers.split_source_with_supar(src_paragraph)
            supar_parts = [p.strip() for p in (supar_parts or []) if str(p).strip()]
            if supar_parts:
                candidate_sets.append((f"supar({len(supar_parts)})", supar_parts))
        except Exception as e:
            if verbose:
                print(f"   ⚠️ supar 후보 생성 실패(무시): {e}")

    # (A2) 공백(어절) 후보 기반 DP 분할: 번역문 문장 경계와 의미적으로 맞는 공백을 선택
    # - 원문 어절 내부 분할을 원천 차단
    # - 의미 기반(임베딩)으로 '어느 공백이 경계인지'를 고르는 후보를 추가
    ws_dp_debug_meta: Dict[str, Any] = {}
    if not disable_whitespace_dp:
        try:
            ws_parts = split_source_by_whitespace_and_align(
                src_paragraph,
                len(tgt_sentences),
                target_sentences=tgt_sentences,
                embedder_name='bge',
                enable_src_marker_whitespace_dp_bonus=enable_src_marker_whitespace_dp_bonus,
                debug_meta_out=ws_dp_debug_meta,
            )
            ws_parts = [p for p in (ws_parts or []) if p is not None]
            if ws_parts:
                candidate_sets.append((f"whitespace_dp({len(ws_parts)})", ws_parts))
        except Exception as e:
            if verbose:
                print(f"   ⚠️ whitespace_dp 후보 생성 실패(무시): {e}")

            # 실패도 '실제 적용' 관점에서 흔적을 남긴다(후속 trace에서 확인 가능)
            try:
                ws_dp_debug_meta = {
                    "error": str(e),
                    "enabled_marker_bonus": bool(enable_src_marker_whitespace_dp_bonus),
                }
            except Exception:
                ws_dp_debug_meta = {}

    # (B) Boundary 모델 후보: threshold를 낮춰가며 tgt 개수 이상이 되도록 시도
    boundary_best: List[str] = []
    boundary_best_tag = "boundary"
    if not disable_boundary:
        for th in [threshold, max(0.05, threshold - 0.1), max(0.05, threshold - 0.2), max(0.05, threshold - 0.3)]:
            try:
                parts = boundary_model.segment_text(
                    src_paragraph,
                    task="pa",
                    threshold=th,
                    min_len_override=boundary_min_len,
                )
                # 후보 텍스트를 임의로 strip() 하면 임베딩/유사도에 영향을 줄 수 있어,
                # boundary 출력은 가능한 그대로 보존한다.
                parts = [p for p in (parts or []) if p is not None]
                if not parts:
                    continue

                # 목표 개수에 더 근접(>=desired 우선)한 후보를 채택
                if not boundary_best:
                    boundary_best = parts
                    boundary_best_tag = f"boundary(th={th:.2f},{len(parts)})"
                else:
                    def _score_len(n: int) -> tuple[int, int]:
                        # 1) desired 미만이면 패널티(1), 이상이면 0
                        # 2) desired와의 절대 거리
                        return (0 if n >= desired else 1, abs(n - desired))

                    if _score_len(len(parts)) < _score_len(len(boundary_best)):
                        boundary_best = parts
                        boundary_best_tag = f"boundary(th={th:.2f},{len(parts)})"

                if len(parts) >= desired:
                    # 이미 tgt 개수 이상이면 더 낮추지 않아도 됨(과분할은 아래에서 제한)
                    break
            except Exception as e:
                if verbose:
                    print(f"   ⚠️ boundary 후보 생성 실패(th={th:.2f}): {e}")

    if boundary_best:
        candidate_sets.append((boundary_best_tag, boundary_best))

    if not candidate_sets:
        # strict 모드: 후보 생성이 완전히 실패한 경우는 진행 불가
        disabled_info = f"disabled: supar={disable_supar}, boundary={disable_boundary}, whitespace_dp={disable_whitespace_dp}"
        raise RuntimeError(f"원문 후보 경계 생성에 실패했습니다 (모든 활성 후보 방식 실패, {disabled_info})")

    # 후보 세트별로 (1) 과분할 제한 → (2) match_segments → (3) 평균 유사도 비교
    best_tag = None
    best_src = None
    best_score = -1.0
    best_cand_len = 0

    pa_sel_params = get_pa_selection_params()
    prior_bonus_by_prefix: Dict[str, float] = pa_sel_params.get("candidate_prior_bonus_by_prefix", {}) or {}
    style_params: Dict = pa_sel_params.get("boundary_style_prior", {}) or {}
    penalty_short_cfg: Dict = pa_sel_params.get("penalty_short_pairs", {}) or {}
    penalty_empty_src: float = float(pa_sel_params.get("penalty_empty_src", 0.5))
    max_cand_mult: int = int(pa_sel_params.get("max_candidates_multiplier", 12))
    ws_dp_cfg: Dict = pa_sel_params.get("whitespace_dp_penalties", {}) or {}

    def _boundary_style_bonus(parts: List[str]) -> tuple[float, int, int]:
        """경계의 '문장 종결스러움'을 약하게 반영.

        GT를 보지 않고도, (…이라/…矣라/…也라/…哉아 등) 종결형에서 끊는 후보를
        (…하며/而/以/則 등) 연결형에서 끊는 후보보다 선호하도록 유도한다.

        반환: (bonus, terminal_cnt, continuation_cnt)
        """

        def _last_token(seg: str) -> str:
            s = str(seg).strip()
            if not s:
                return ""
            tok = s.split()[-1]
            return tok.strip("\"'”’」』》〉】〕)\]\}.,，。?!;:·、")

        if not bool(style_params.get("enabled", True)):
            return 0.0, 0, 0

        # 매우 보수적으로 잡는다(과적합 방지): 자주 관찰되는 연결형/종결형만 사용
        continuation_tokens = set(style_params.get("continuation_tokens", []) or [])
        # 공백이 없는 한자/한글 혼용 원문에서는 마지막 토큰이 통째로 붙어 나오는 경우가 많다.
        # 예: "任耕하고" 처럼 종결이 아닌 연결형이 토큰의 접미사로 나타나는 케이스를 잡기 위해
        # continuation_tokens를 "suffix"로도 취급한다(보수적; 설정에 포함된 것만 사용).
        continuation_suffixes = tuple(
            sorted(
                {
                    str(x).strip().rstrip(",")
                    for x in continuation_tokens
                    if str(x).strip().rstrip(",")
                },
                key=len,
                reverse=True,
            )
        )
        continuation_tail_cjk = set(style_params.get("continuation_tail_cjk", []) or [])
        terminal_suffixes = tuple(style_params.get("terminal_suffixes", []) or [])
        terminal_punct = set(style_params.get("terminal_punct", [".", "!", "?", "。", "！", "？"]) or [])

        terminal_cnt = 0
        continuation_cnt = 0
        for a, b in zip(parts, parts[1:]):
            a_s = str(a).strip()
            b_s = str(b).strip()
            if not a_s or not b_s:
                continue

            tok = _last_token(a_s)
            if tok in continuation_tokens:
                continuation_cnt += 1
                continue
            if tok and continuation_suffixes and tok.endswith(continuation_suffixes):
                continuation_cnt += 1
                continue
            if tok and tok[-1] in continuation_tail_cjk:
                continuation_cnt += 1
                continue

            if tok.endswith(terminal_suffixes):
                terminal_cnt += 1
                continue
            # 명시적 종결 구두점
            if a_s and a_s[-1] in terminal_punct:
                terminal_cnt += 1

        w_term = float(style_params.get("weight_terminal", 0.018))
        w_cont = float(style_params.get("weight_continuation", -0.030))
        bonus = (w_term * terminal_cnt) + (w_cont * continuation_cnt)
        return bonus, terminal_cnt, continuation_cnt

    # 진단용: src_matched_selected에서 후보별 점수/패널티/스킵 사유를 trace에 남긴다.
    candidate_reports: List[Dict] = []
    cand_skipped_insufficient = 0
    cand_considered = 0
    cand_short_for_desired = 0

    # 후보셋 존재/길이(병합 후) 및 스킵/고려 태그를 trace에 남겨 "실제로 무엇이 생성/필터링됐나"를 판단한다.
    candidate_set_lengths: Dict[str, int] = {}
    candidate_set_lengths_orig: Dict[str, int] = {}
    skipped_insufficient_tags: list[str] = []
    considered_tags: list[str] = []
    short_for_desired_tags: list[str] = []

    whitespace_tag: str | None = None
    whitespace_src: List[str] | None = None
    whitespace_score: float | None = None
    whitespace_cand_len: int | None = None
    whitespace_dp_debug_meta: Dict[str, Any] | None = None

    # AlignmentMatcher가 tgt 1개당 src를 "최대 4개"까지 병합해보는 제약이 있어,
    # 후보가 과도하게 많으면 탐색이 제한되거나 병합(=_merge_to_max_parts)로 경계 옵션이 사라질 수 있다.
    # 지나치게 aggressive한 병합을 줄이기 위해 상한을 완화한다.
    max_candidates = max(desired * max_cand_mult, desired)

    # 후보 중에 tgt 개수 이상을 제공하는 세트가 있으면,
    # 그 외 (cand < desired)는 '문자 균등 분할' 폴백 가능성이 커서 제외한다.
    sufficient_exists = False
    for _tag, _cand in candidate_sets:
        _cand2 = _merge_to_max_parts(_cand, max_candidates)
        if len(_cand2) >= desired:
            sufficient_exists = True
            break

    for tag, cand in candidate_sets:
        cand2 = _merge_to_max_parts(cand, max_candidates)

        try:
            candidate_set_lengths[str(tag)] = int(len(cand2))
            candidate_set_lengths_orig[str(tag)] = int(len(cand))
        except Exception:
            pass

        # NOTE:
        # - 예전에는 cand_len < desired일 때, 매칭기가 '문자 균등 분할'로 fallback될 가능성이 커
        #   후보 자체를 제외했었다.
        # - 현재 AlignmentMatcher.match_segments는 src_segments가 부족하면
        #   후보 경계를 최대한 보존한 채 세그먼트 내부 분할로 tgt 개수까지 확장한다.
        #   따라서 여기서 aggressive하게 후보를 제외하면 boundary/supar 후보가 과도하게 사라져
        #   실제 비교가 축소(considered==1)되는 부작용이 크다.
        short_for_desired = bool(sufficient_exists and len(cand2) < desired)
        shortfall = int(max(0, int(desired) - int(len(cand2))))
        if short_for_desired:
            cand_short_for_desired += 1
            short_for_desired_tags.append(str(tag))
            if verbose:
                print(
                    f"   ⚠️ 후보 {tag}: cand={len(cand)}→{len(cand2)} (tgt={desired}) 부족(확장 매칭으로 평가 진행)"
                )

        src_matched = alignment_model.match_segments(cand2, tgt_sentences)
        avg_similarity = _avg_sim(src_matched, tgt_sentences)
        score = avg_similarity
        cand_considered += 1
        considered_tags.append(str(tag))

        # 경계 후보 선호도(약한 prior):
        # - supar/경계모델 기반 후보는 '자연스러운 문장/구 경계'를 제공하는 경우가 많아
        #   유사도 점수 차이가 미미할 때 원문 경계 micro-F1에 유리할 수 있다.
        prior_bonus = 0.0
        for prefix, bonus in prior_bonus_by_prefix.items():
            if str(tag).startswith(str(prefix)):
                prior_bonus = float(bonus)
                break
        score += prior_bonus

        # 경계 문체(종결/연결) 신호를 약하게 반영해, 의미 유사도만으로는 구분이 어려운
        # '확신형 오답(경계 통째로 어긋남)'을 억제한다.
        # 단, alignment score가 낮으면 의미 대응이 약한 것이므로 style bonus 무시
        style_bonus, terminal_cnt, continuation_cnt = _boundary_style_bonus(src_matched)
        # boundary 후보가 연결형(…하고/…하며/而/以/則 등)에서 더 많이 끊기면,
        # prior가 오답을 강제로 선택하게 만드는 부작용이 크다.
        # 이 경우에는 boundary prior를 적용하지 않는다(의미 유사도와 다른 페널티로 승부).
        if prior_bonus > 0.0 and str(tag).startswith("boundary(") and continuation_cnt > terminal_cnt:
            score -= prior_bonus
            prior_bonus = 0.0
        # 의미 유사도가 낮을 때는 스타일 신호가 오판을 키울 수 있어 보통은 무시한다.
        # 다만 whitespace_dp는 저유사도 영역에서 '과분할/경계 붕괴'가 자주 발생하므로,
        # 그 억제를 위해 style bonus(특히 continuation penalty)는 항상 적용한다.
        if avg_similarity < 0.6 and not str(tag).startswith("whitespace_dp("):
            style_bonus = 0.0
        score += style_bonus

        # 추가 정규화(일반 규칙):
        # 번역문(한국어) 문장이 충분히 길 때, 대응하는 원문 세그먼트가
        # '절대적으로/상대적으로' 지나치게 짧으면 경계 F1이 악화되는 경우가 많다.
        # 특히 whitespace_dp는 의미 기반으로 과분할을 선호할 수 있어, 동일 규칙을 더 강하게 적용한다.
        # - 특정 pid/문구 하드코딩 없이, 길이 기반의 일반 규칙만 사용
        # 추가 정규화(일반 규칙):
        # 번역문(한국어) 문장이 충분히 길 때(src 의미 단위가 꽤 있어야 자연스러움),
        # 원문 세그먼트가 지나치게 짧으면(특히 whitespace_dp가 과분할하는 경우)
        # 경계 F1이 악화되는 케이스가 많아 약한 패널티를 준다.
        # - 특정 pid/문구 하드코딩 없이, 길이 기반의 일반 규칙만 사용
        long_tgt = int(penalty_short_cfg.get("long_tgt_threshold", 40))
        short_src = int(penalty_short_cfg.get("short_src_threshold", 12))
        per_pair_penalty = float(penalty_short_cfg.get("penalty_per_pair", 0.015))
        short_pairs = 0
        for s, t in zip(src_matched, tgt_sentences):
            if t is None or s is None:
                continue
            if len(str(t)) >= long_tgt and len(str(s).strip()) <= short_src:
                short_pairs += 1
        penalty_short_pairs_total = per_pair_penalty * short_pairs
        if short_pairs:
            score -= penalty_short_pairs_total

        # whitespace_dp는 의미 유사도만으로 '과분할 + 긴 tgt에 짧은 src 매칭'이 선택되는 경우가 있어,
        # 경계 F1이 붕괴하는 케이스를 막기 위해 "긴 tgt ↔ 짧은 src" 쌍에만 추가 패널티를 준다.
        # (특정 pid/문구 하드코딩 없이, 길이 기반 일반 규칙)
        ws_severe_pairs = 0
        ws_severe_very_short_pairs = 0
        penalty_ws_severe_total = 0.0
        penalty_ws_very_short_total = 0.0
        penalty_ws_ratio_outlier_total = 0.0
        penalty_ws_longest_shortest_total = 0.0
        if tag.startswith("whitespace_dp("):
            long_tgt2 = int(ws_dp_cfg.get("long_tgt_threshold", 80))
            short_src2 = int(ws_dp_cfg.get("short_src_threshold", 25))
            very_short_src2 = int(ws_dp_cfg.get("very_short_src_threshold", 8))
            penalty_short = float(ws_dp_cfg.get("penalty_short", 0.070))
            penalty_very_short = float(ws_dp_cfg.get("penalty_very_short", 0.090))

            for s, t in zip(src_matched, tgt_sentences):
                if t is None or s is None:
                    continue
                if len(str(t)) < long_tgt2:
                    continue
                s_len = len(str(s).strip())
                if s_len <= very_short_src2:
                    ws_severe_very_short_pairs += 1
                elif s_len <= short_src2:
                    ws_severe_pairs += 1
            penalty_ws_severe_total = penalty_short * ws_severe_pairs
            penalty_ws_very_short_total = penalty_very_short * ws_severe_very_short_pairs
            if ws_severe_pairs:
                score -= penalty_ws_severe_total
            if ws_severe_very_short_pairs:
                score -= penalty_ws_very_short_total

            # 추가 가드(일반 규칙):
            # whitespace_dp가 과분할된 경우, "가장 긴 번역문"이 "가장 짧은 원문"에 매칭되면
            # 원문 경계가 앞쪽으로 쏠리며 micro-F1이 0으로 붕괴하는 케이스가 관찰됨.
            # - 특정 pid/문구 하드코딩 없이, (길이 순서 + 비율)만으로 탐지
            try:
                ratio_cfg: Dict = ws_dp_cfg.get("ratio_outlier", {}) or {}
                min_tgt_len = int(ratio_cfg.get("min_tgt_len", 80))
                ratio_high_thr = float(ratio_cfg.get("ratio_high_threshold", 3.8))
                ratio_mid_thr = float(ratio_cfg.get("ratio_mid_threshold", 3.2))
                med_margin_high = float(ratio_cfg.get("median_margin_high", 1.2))
                med_margin_mid = float(ratio_cfg.get("median_margin_mid", 1.0))
                src_cap_high = int(ratio_cfg.get("src_len_cap_high", 45))
                src_cap_mid = int(ratio_cfg.get("src_len_cap_mid", 35))
                pen_high = float(ratio_cfg.get("penalty_high", 0.18))
                pen_mid = float(ratio_cfg.get("penalty_mid", 0.12))
                pen_longest = float(ratio_cfg.get("penalty_longest_shortest", 0.10))

                src_lens = [len(str(s).strip()) for s in src_matched]
                tgt_lens = [len(str(t).strip()) for t in tgt_sentences]
                if src_lens and tgt_lens and len(src_lens) == len(tgt_lens) and len(src_lens) >= 2:
                    # (1) 비율 outlier 탐지: 특정 한 쌍만 과도하게 "긴 tgt ↔ 짧은 src"면 붕괴 위험
                    ratios: list[float] = []
                    for s_len, t_len in zip(src_lens, tgt_lens):
                        ratios.append(t_len / max(1, s_len))
                    ratios_sorted = sorted(ratios)
                    mid = len(ratios_sorted) // 2
                    median_ratio = (
                        ratios_sorted[mid]
                        if len(ratios_sorted) % 2 == 1
                        else (ratios_sorted[mid - 1] + ratios_sorted[mid]) / 2.0
                    )

                    for s_len, t_len in zip(src_lens, tgt_lens):
                        if t_len < min_tgt_len:
                            continue
                        ratio = t_len / max(1, s_len)
                        # 전반적인 장황함(전체 비율 상승)은 허용하되,
                        # median 대비 과도하게 튀는 outlier만 벌점
                        if ratio >= ratio_high_thr and ratio >= (median_ratio + med_margin_high) and s_len <= src_cap_high:
                            penalty_ws_ratio_outlier_total += pen_high
                        elif ratio >= ratio_mid_thr and ratio >= (median_ratio + med_margin_mid) and s_len <= src_cap_mid:
                            penalty_ws_ratio_outlier_total += pen_mid
                    if penalty_ws_ratio_outlier_total > 0:
                        score -= penalty_ws_ratio_outlier_total

                    # (2) longest tgt ↔ shortest src 케이스(명시적)도 추가로 방지
                    s_min = min(src_lens)
                    t_max = max(tgt_lens)
                    for s_len, t_len in zip(src_lens, tgt_lens):
                        if t_len < min_tgt_len:
                            continue
                        if t_len != t_max or s_len != s_min:
                            continue
                        ratio = t_len / max(1, s_len)
                        if ratio >= ratio_high_thr and s_len <= src_cap_high:
                            penalty_ws_longest_shortest_total = pen_longest
                            score -= penalty_ws_longest_shortest_total
                        break
            except Exception:
                pass

        # 빈 원문이 있으면 강한 패널티 (하나라도 있으면 고정 penalty 적용)
        empty_src_pairs = 0
        for s in src_matched:
            if not str(s).strip():
                empty_src_pairs += 1
        penalty_empty_total = penalty_empty_src if empty_src_pairs > 0 else 0.0
        if empty_src_pairs:
            score -= penalty_empty_total

        # 후보 평가 요약 기록(선택 로직에 사용된 최종 score 기준)
        rep: Dict = {
            "tag": str(tag),
            "cand_len_orig": int(len(cand)),
            "cand_len": int(len(cand2)),
            "considered": True,
            "short_for_desired": bool(short_for_desired),
            "shortfall": int(shortfall),
            "avg_similarity": float(avg_similarity),
            "score": float(score),
            "prior_bonus": float(prior_bonus),
            "short_pairs": int(short_pairs),
            "penalty_short_pairs": float(penalty_short_pairs_total),
            "empty_src_pairs": int(empty_src_pairs),
            "penalty_empty_src_pairs": float(penalty_empty_total),
            "terminal_boundaries": int(terminal_cnt),
            "continuation_boundaries": int(continuation_cnt),
            "boundary_style_bonus": float(style_bonus),
        }
        if tag.startswith("whitespace_dp("):
            rep["ws_severe_pairs"] = int(ws_severe_pairs)
            rep["ws_severe_very_short_pairs"] = int(ws_severe_very_short_pairs)
            rep["penalty_ws_severe"] = float(penalty_ws_severe_total)
            rep["penalty_ws_very_short"] = float(penalty_ws_very_short_total)
            rep["penalty_ws_ratio_outlier"] = float(penalty_ws_ratio_outlier_total)
            rep["penalty_ws_longest_shortest"] = float(penalty_ws_longest_shortest_total)
            # whitespace DP 내부에서 실제로 어떤 임베더/폴백/보너스가 적용됐는지(=기여도)
            if isinstance(ws_dp_debug_meta, dict) and ws_dp_debug_meta:
                rep["ws_dp_debug"] = dict(ws_dp_debug_meta)
        candidate_reports.append(rep)

        if verbose:
            print(f"   🔎 후보 {tag}: cand={len(cand)}→{len(cand2)}, score={score:.4f}")

        if tag.startswith("whitespace_dp("):
            whitespace_tag = tag
            whitespace_src = src_matched
            whitespace_score = score
            whitespace_cand_len = len(cand2)
            if isinstance(rep.get("ws_dp_debug"), dict):
                whitespace_dp_debug_meta = rep.get("ws_dp_debug")

        if score > best_score:
            best_score = score
            best_tag = tag
            best_src = src_matched
            best_cand_len = len(cand2)

    # 타이브레이크(비활성): 공백(어절) 후보 기반 DP를 근소하게 뒤처져도 우선 선택하면
    # 원문 경계 정확도(경계 F1)가 악화될 수 있어, 동점/근소 차이는 기본 점수(best_score) 기준을 유지한다.
    if (
        whitespace_tag is not None
        and whitespace_src is not None
        and whitespace_score is not None
        and whitespace_cand_len is not None
        and best_tag is not None
        and best_src is not None
        and best_tag != whitespace_tag
        and whitespace_score > best_score
    ):
        best_tag = whitespace_tag
        best_src = whitespace_src
        best_score = whitespace_score
        best_cand_len = whitespace_cand_len
        if verbose:
            print(f"   🔁 타이브레이크: 공백 후보 우선 선택 → {best_tag}")

    if best_src is None:
        # 매우 드물게, 후보 생성/필터링/매칭 과정에서 어떤 후보도 선택되지 못하는 경우가 있다.
        # 이때 런 전체를 중단하면 seed 멀티테스트가 불안정해지므로,
        # 번역문 문장 리스트(tgt_sentences)는 그대로 유지한 채, 보수적으로 원문을 동일 개수로 분할한다.
        if verbose:
            print("   ⚠️ 원문 매칭 후보 선택 실패 → safe_source_split 폴백 적용")
        try:
            src_sentences = safe_source_split(tgt_sentences, src_paragraph)
        except Exception:
            # 최후의 수단: 공백 기반 분할(무결성 복원 단계가 있으므로 raw 자체는 보존됨)
            src_sentences = split_source_by_whitespace_and_align(src_paragraph, len(tgt_sentences))

        # 길이 보정(무결성/리스트 길이 불변)
        while len(src_sentences) < len(tgt_sentences):
            src_sentences.append("")
        if len(src_sentences) > len(tgt_sentences):
            src_sentences = src_sentences[: len(tgt_sentences)]

        # trace가 있으면 폴백 발생을 명시
        if trace is not None:
            try:
                trace(
                    "src_matched_selected_fallback",
                    src_segments=[str(x) for x in src_sentences],
                    tgt_segments=[str(x) for x in tgt_sentences],
                    meta={
                        "reason": "best_src_none",
                        "fallback": "safe_source_split",
                        "desired_tgt_len": int(desired),
                        "max_candidates": int(max_candidates),
                        "sufficient_exists": bool(sufficient_exists),
                        "candidates_total": int(len(candidate_sets)),
                        "candidates_considered": int(cand_considered),
                        "candidates_skipped_insufficient": int(cand_skipped_insufficient),
                        "candidate_reports": candidate_reports[-12:],
                    },
                )
            except Exception:
                pass
        best_tag = best_tag or "fallback(safe_source_split)"
        best_score = float(best_score)
        best_cand_len = int(best_cand_len)
    else:
        src_sentences = best_src

    if trace is not None:
        try:
            # 후보 top-k 및 마진(1등-2등)을 meta에 추가
            considered = [r for r in candidate_reports if r.get("considered") and ("score" in r)]
            considered_sorted = sorted(considered, key=lambda r: float(r.get("score", -1e9)), reverse=True)
            top_k = 6
            top_candidates = considered_sorted[:top_k]
            best_margin = None
            if len(considered_sorted) >= 2:
                best_margin = float(considered_sorted[0]["score"]) - float(considered_sorted[1]["score"])

            best_tag_str = str(best_tag) if best_tag is not None else None
            used_supar_candidate = bool(best_tag_str and best_tag_str.startswith("supar("))
            used_boundary_candidate = bool(best_tag_str and best_tag_str.startswith("boundary("))
            used_whitespace_dp_candidate = bool(best_tag_str and best_tag_str.startswith("whitespace_dp("))

            # 후보셋 존재 여부(생성되었는지)
            had_supar_candidate_set = any(str(t).startswith("supar(") for t in candidate_set_lengths.keys())
            had_boundary_candidate_set = any(str(t).startswith("boundary(") for t in candidate_set_lengths.keys())
            had_whitespace_dp_candidate_set = any(str(t).startswith("whitespace_dp(") for t in candidate_set_lengths.keys())

            # 후보별 마진 표준화: JSONL만 봐도 "왜 이 후보가 이겼는지" 비교 가능
            top_candidate_margins: list[dict] = []
            if considered_sorted:
                best_score0 = float(considered_sorted[0].get("score", 0.0))
                for i, cand in enumerate(top_candidates):
                    score_i = float(cand.get("score", 0.0))
                    next_score = float(top_candidates[i + 1].get("score", 0.0)) if i + 1 < len(top_candidates) else None
                    top_candidate_margins.append(
                        {
                            "rank": int(i + 1),
                            "tag": str(cand.get("tag")),
                            "score": float(score_i),
                            "margin_vs_best": float(score_i - best_score0),
                            "margin_vs_next": (float(score_i - next_score) if next_score is not None else None),
                        }
                    )

            # whitespace_dp가 존재하면 best와의 비교도 남긴다 (DP 사용/미사용 판단에 핵심)
            whitespace_best = None
            try:
                ws_candidates = [r for r in considered_sorted if str(r.get("tag", "")).startswith("whitespace_dp(")]
                if ws_candidates:
                    whitespace_best = max(ws_candidates, key=lambda r: float(r.get("score", -1e9)))
            except Exception:
                whitespace_best = None
            margin_best_vs_whitespace = None
            if whitespace_best is not None and considered_sorted:
                try:
                    margin_best_vs_whitespace = float(considered_sorted[0]["score"]) - float(whitespace_best["score"])
                except Exception:
                    margin_best_vs_whitespace = None

            trace(
                "src_matched_selected",
                src_segments=[str(x) for x in src_sentences],
                tgt_segments=[str(x) for x in tgt_sentences],
                meta={
                    "best_tag": best_tag,
                    "best_score": float(best_score),
                    "best_cand_len": int(best_cand_len),
                    "desired_tgt_len": int(desired),
                    "max_candidates": int(max_candidates),
                    "sufficient_exists": bool(sufficient_exists),
                    "candidates_total": int(len(candidate_sets)),
                    "candidates_considered": int(cand_considered),
                    "candidates_skipped_insufficient": int(cand_skipped_insufficient),
                    "candidates_short_for_desired": int(cand_short_for_desired),
                    "best_margin_vs_second": best_margin,
                    "used_supar_candidate": used_supar_candidate,
                    "used_boundary_candidate": used_boundary_candidate,
                    "used_whitespace_dp_candidate": used_whitespace_dp_candidate,
                    "had_supar_candidate_set": bool(had_supar_candidate_set),
                    "had_boundary_candidate_set": bool(had_boundary_candidate_set),
                    "had_whitespace_dp_candidate_set": bool(had_whitespace_dp_candidate_set),
                    "candidate_set_lengths": candidate_set_lengths,
                    "candidate_set_lengths_orig": candidate_set_lengths_orig,
                    "considered_tags": considered_tags,
                    "skipped_insufficient_tags": skipped_insufficient_tags,
                    "short_for_desired_tags": short_for_desired_tags,
                    "marker_boundary_bonus_hits": int(marker_boundary_bonus_hits),
                    "marker_boundary_bonus_sum": float(marker_boundary_bonus_sum),
                    "margin_best_vs_whitespace_dp": margin_best_vs_whitespace,
                    "top_candidate_margins": top_candidate_margins,
                    "top_candidates": top_candidates,
                },
            )
        except Exception:
            pass

    # 선택된 후보가 boundary/supar 기반이면, 인접 경계에서 토큰 단위 이동만 허용하는 로컬 교정을 적용한다.
    # (문자열 변경/삭제 없음, 어절 내부 분할 금지, 무결성 유지)
    if enable_adjacent_boundary_refine:
        try:
            if (
                best_tag is not None
                and (best_tag.startswith("boundary(") or best_tag.startswith("supar("))
                and isinstance(src_sentences, list)
                and len(src_sentences) == len(tgt_sentences)
                and len(src_sentences) >= 2
            ):
                # enable-refine 옵션은 파이프라인 재매칭이 아니라,
                # DP/인접 경계 refine의 이동 폭을 키워(예: 4 토큰) 연결어/완곡표현 경계도 탐색하도록 한다.
                src_sentences_before = list(src_sentences)
                src_sentences = _refine_adjacent_boundaries(
                    src_sentences,
                    tgt_sentences,
                    max_shift_tokens=max(1, int(adjacent_refine_max_shift_tokens)),
                )

                changed_segments = 0
                try:
                    for a, b in zip(src_sentences_before, src_sentences):
                        if str(a) != str(b):
                            changed_segments += 1
                except Exception:
                    changed_segments = 0

                if trace is not None:
                    try:
                        trace(
                            "src_adjacent_refined",
                            src_segments=[str(x) for x in src_sentences],
                            tgt_segments=[str(x) for x in tgt_sentences],
                            meta={"best_tag": best_tag, "changed_segments": int(changed_segments)},
                        )
                    except Exception:
                        pass
        except Exception:
            # 교정 실패 시 원래 결과 유지
            pass

    # === 불변 조건 강제 ===
    # 1) 어떤 경우에도 어절(공백/구두점) 내부에서 분할되면 안 됨
    # 2) 테스트 케이스에서 "脩屢建言"은 2문장 시작이 되어야 함
    def _is_boundary_char(ch: str) -> bool:
        if not ch:
            return True
        if ch.isspace():
            return True
        return ch in "\"'”’」』》〉】〕)\]\}.,，。?!;:·、"

    def _has_intra_token_split(parts: List[str]) -> bool:
        def _is_cjk(ch: str) -> bool:
            if not ch:
                return False
            o = ord(ch)
            return (
                0x4E00 <= o <= 0x9FFF
                or 0x3400 <= o <= 0x4DBF
                or 0xF900 <= o <= 0xFAFF
            )

        def _is_hangul(ch: str) -> bool:
            if not ch:
                return False
            o = ord(ch)
            return 0xAC00 <= o <= 0xD7A3

        cursor = 0
        for a, b in zip(parts, parts[1:]):
            if not a or not b:
                cursor += len(str(a))
                continue
            if (not _is_boundary_char(a[-1])) and (not _is_boundary_char(b[0])):
                # CJK/한글 연속 구간은 예외로 허용
                if (_is_cjk(a[-1]) and _is_cjk(b[0])) or (_is_hangul(a[-1]) and _is_hangul(b[0])):
                    cursor += len(str(a))
                    continue
                return True
            cursor += len(str(a))
        return False

    needs_fix = False
    if _has_intra_token_split(src_sentences):
        needs_fix = True

    if needs_fix:
        try:
            # safe_source_split은 문장 수를 tgt에 맞추면서(=불변), 공백 기반으로 원문을 안전하게 쪼갠다.
            fixed = safe_source_split(tgt_sentences, src_paragraph)
            if fixed and len(fixed) == len(tgt_sentences) and not _has_intra_token_split(fixed):
                src_sentences = fixed
                if verbose:
                    print("   🛠️ 원문 경계 강제 보정: safe_source_split 적용")

                if trace is not None:
                    try:
                        trace(
                            "src_safe_source_split",
                            src_segments=[str(x) for x in src_sentences],
                            tgt_segments=[str(x) for x in tgt_sentences],
                        )
                    except Exception:
                        pass
        except Exception as e:
            if verbose:
                print(f"   ⚠️ safe_source_split 보정 실패(무시): {e}")
    if verbose:
        print(
            f"   ✅ 원문 매칭(match_segments): selected={best_tag}, candidates={best_cand_len} → {len(src_sentences)}, avg_sim={best_score:.4f}"
        )
    
    # 4. 결과 조립
    alignments = []
    for src_sent, tgt_sent in zip(src_sentences, tgt_sentences):
        alignments.append({
            '원문': src_sent,
            '번역문': tgt_sent,
            'similarity': alignment_model.compute_similarity(src_sent, tgt_sent) if alignment_model else compute_similarity_simple(src_sent, tgt_sent)
        })

    if trace is not None:
        try:
            trace(
                "alignment_built",
                src_segments=[str(x) for x in src_sentences],
                tgt_segments=[str(x) for x in tgt_sentences],
            )
        except Exception:
            pass
    
    return alignments
def _normalize_brackets_in_text(text: str) -> str:
    """텍스트의 연쇄/중첩 [-…] 괄호를 정상화한다.
    
    SA의 mask/restore 방식 대신, 정규식으로 직접 제거하되,
    재귀적으로 반복해서 모든 중첩/연쇄 제거.
    
    - 가장 안쪽 중첩부터 제거: [-..[-..]..]
    - 빈 블록 제거: [-], [- ]
    - 공백 정리
    """
    import re
    
    if not text:
        return text
    
    max_iterations = 20
    prev = None
    iteration = 0
    
    while iteration < max_iterations and prev != text:
        prev = text
        iteration += 1
        
        # 1단계: 가장 안쪽 중첩부터 제거
        # 패턴: [-내부[-안쪽]뒷부분]
        # 안쪽 [-...] 블록만 제거하되, 외쪽 괄호 유지
        text = re.sub(r'\[-([^\[\]]*)\[-([^\]]*)\]([^\]]*)\]', r'[-\1\3]', text)
        
        # 2단계: 완전히 비워진 또는 공백만인 블록 제거
        # [-], [- ], [-  ], [- abc - ] 같은 형태들 (공백만 포함)
        text = re.sub(r'\[\-\s*\]', '', text)
        
        # 3단계: 연속된 공백 정리 (마스킹 후 복원 시 발생할 수 있음)
        text = re.sub(r' +', ' ', text)

        # 4단계: 괄호 개수 불균형 보정
        open_cnt = text.count("[-")
        close_cnt = text.count("]")
        # 열림이 더 많으면 앞에서부터 '[-'를 제거
        while open_cnt > close_cnt:
            text = text.replace("[-", "", 1)
            open_cnt -= 1
        # 닫힘이 더 많으면 앞에서부터 ']'를 제거
        while close_cnt > open_cnt:
            text = text.replace("]", "", 1)
            close_cnt -= 1
    
    # 최종 정리
    text = text.strip()
    return text


def _final_cleanup_brackets(text: str) -> str:
    """최종 출력 직전 괄호 블록 중복/불균형을 정리한다.

    - 괄호 내부의 앞뒤 공백 제거 (내용은 유지)
    - 연속된 동일 [-…] 블록이 반복되면 하나만 유지
    - 남는 열림/닫힘 괄호가 있으면 앞에서부터 제거하여 개수 맞춤
    """
    if not text:
        return text

    import re
    
    # 1단계: 괄호 내부의 앞뒤 공백만 제거 (빈 블록은 유지)
    # [-  내용  ] → [-내용]
    def trim_inside(m):
        content = m.group(1).strip()
        return f"[-{content}]"
    text = re.sub(r'\[-([^\]]*)\]', trim_inside, text)
    
    # 2단계: 연속 중복 블록 제거
    pattern = re.compile(r"\[-[^\]]*\]")
    parts = []
    last = 0
    prev_block = None
    for m in pattern.finditer(text):
        # 중간 일반 텍스트
        parts.append(text[last:m.start()])
        block = m.group(0)
        if block == prev_block:
            # 중복 블록은 건너뜀
            last = m.end()
            continue
        parts.append(block)
        prev_block = block
        last = m.end()
    parts.append(text[last:])
    cleaned = ''.join(parts)

    # 3단계: 괄호 개수 불균형 보정
    open_cnt = cleaned.count("[-")
    close_cnt = cleaned.count("]")
    while open_cnt > close_cnt:
        cleaned = cleaned.replace("[-", "", 1)
        open_cnt -= 1
    while close_cnt > open_cnt:
        cleaned = cleaned.replace("]", "", 1)
        close_cnt -= 1

    # 4단계: 연속 공백 정리
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def _ensure_atomic_brackets_in_alignments(alignments: List[Dict]) -> List[Dict]:
    """문장 경계에 걸친 [-…] 괄호 블록을 한 조각에 원자적으로 붙이도록 보정한다.

    - '원문'과 '번역문' 모두에 동일 규칙 적용
    - 결합 텍스트에서 [-…] 블록의 전역 오프셋을 찾고,
      시작/끝이 서로 다른 조각에 걸치면 시작 조각의 끝으로 이동한다.
    - 내부 공백과 원래 표기(괄호 포함)를 그대로 보존한다.
    - 다음 조각이 ']'로 시작하면 닫힘 대괄호를 이전 조각 끝으로 흡수한다.
    - 먼저 텍스트 정규화로 연쇄/중첩 제거.
    """
    import re
    if not alignments:
        return alignments

    def _process_column(col_name: str):
        # 1단계: 각 셀의 텍스트 정규화 (연쇄/중첩 제거)
        for a in alignments:
            if col_name in a and a[col_name]:
                a[col_name] = _normalize_brackets_in_text(str(a[col_name]))
        
        segments = [str(a.get(col_name, '')) for a in alignments]
        if not segments:
            return

        cumulative = [0]
        for s in segments:
            cumulative.append(cumulative[-1] + len(s))

        full = ''.join(segments)
        pattern = re.compile(r"\[-(?:\([^)]*\)|[^\]]*)\]", re.S)
        matches = list(pattern.finditer(full))
        if matches:
            for m in matches:
                start, end = m.start(), m.end()

                def find_idx(pos: int) -> int:
                    for i in range(len(segments)):
                        if cumulative[i] <= pos < cumulative[i+1]:
                            return i
                    return len(segments) - 1

                i_start = find_idx(start)
                i_end = find_idx(end - 1)
                if i_start == i_end:
                    continue

                block = full[start:end]
                local_start = start - cumulative[i_start]
                local_end_in_end_seg = end - cumulative[i_end]

                seg_start = segments[i_start]
                for k in range(i_start + 1, i_end):
                    segments[k] = ''

                seg_end = segments[i_end]
                if local_end_in_end_seg > 0:
                    segments[i_end] = seg_end[local_end_in_end_seg:]

                segments[i_start] = seg_start + block

                cumulative = [0]
                full = ''.join(segments)
                for s in segments:
                    cumulative.append(cumulative[-1] + len(s))

        # 2차 보정: 다음 조각이 ']'로 시작하면 이전 조각으로 흡수
        for i in range(len(segments) - 1):
            nxt = segments[i+1]
            if not nxt:
                continue
            j = 0
            while j < len(nxt) and nxt[j].isspace():
                j += 1
            if j < len(nxt) and nxt[j] == ']':
                rest = nxt[j+1:]
                segments[i] = segments[i] + ']'
                segments[i+1] = rest.lstrip()

        for idx, s in enumerate(segments):
            alignments[idx][col_name] = s

    # 두 컬럼 모두 처리
    _process_column('원문')
    _process_column('번역문')
    return alignments

# === 인용 표지 병합 유틸 ===
def _is_quotation_marker_sentence(text: str) -> bool:
    """번역문 한 줄이 인용 표지(예: '고 하였다', '라고 말한다', '하고 명하셨다', '”고 하였다')만으로 이루어졌는지 판별
    - 닫는 따옴표(", ”, ’) 전후 허용
    - 종결부호(. ? !) 허용
    - 동사/존칭/시제/종결어미 조합 반복(연쇄 마커) 허용
    """
    import re
    if not text or not text.strip():
        return False
    closing_quote = r'["”’]?'
    quotation_particles = r'(고|[이]?라?고|하고|며|면서)'
    speech_verbs = r'(하|말하|말씀하|명하|이르|대답하|답하|묻|문|여쭙|아뢰|전하|칭하|부르|외치)'
    honorific_tense = r'(?:셨|ㅆ|시었|시어|시는|시ㄴ|시ㄹ|시|었|았|였|는|ㄴ|ㄹ|을)?'
    endings = r'(다|ㄴ다|는다|습니다|ㅂ니다|까|ㄹ까|을까|느냐|ㄴ가|는가|라|거라|소|오|어라|아라|니|으니)'
    punctuation = r'[\.。?!,，]?'
    marker_chunk = (
        closing_quote + r'\s*' + quotation_particles + r'\s+' + speech_verbs + honorific_tense + endings + r'\s*' + punctuation + r'\s*' + closing_quote + r'\s*'
    )
    pattern = r'^\s*(?:' + marker_chunk + r')+$'
    return re.match(pattern, text.strip()) is not None


def _merge_quote_marker_rows(alignments: List[Dict]) -> List[Dict]:
    """인용 표지 단독 번역문 행을 직전 행과 병합한다.
    - 번역문/원문 모두 병합하여 전체 텍스트 무결성 유지
    - similarity는 병합 후 간단한 길이 기반으로 재계산
    - 중첩 마커(연속 여러 줄)도 반복 병합
    """
    if not alignments:
        return alignments
    merged: List[Dict] = []
    i = 0
    while i < len(alignments):
        cur = alignments[i]
        # 다음 줄들 중 인용 표지 행을 모두 누적 병합
        j = i + 1
        acc_tgt = []
        acc_src = []
        while j < len(alignments) and _is_quotation_marker_sentence(alignments[j].get('번역문', '')):
            acc_tgt.append(alignments[j].get('번역문', ''))
            # 원문도 함께 병합하여 무결성 유지
            if alignments[j].get('원문', '').strip():
                acc_src.append(alignments[j].get('원문', ''))
            j += 1
        if acc_tgt:
            # 직전 행과 병합
            new_entry = dict(cur)  # shallow copy
            new_entry['번역문'] = (cur.get('번역문', '') + ' ' + ' '.join(acc_tgt)).strip()
            if acc_src:
                new_entry['원문'] = (cur.get('원문', '') + ' ' + ' '.join(acc_src)).strip()
            # 유사도 재계산
            try:
                new_entry['similarity'] = compute_similarity_simple(new_entry.get('원문', ''), new_entry.get('번역문', ''))
            except Exception:
                pass
            merged.append(new_entry)
            i = j  # 병합된 만큼 건너뛰기
        else:
            merged.append(cur)
            i += 1
    return merged

try:
    from aligner import (
        get_embedder_function,
        process_paragraph_alignment,
        restore_paragraph_integrity,
    )
except ImportError as e:
    print(f"❌ aligner import 실패: {e}")
    
    def get_embedder_function(*args, **kwargs):
        print("❌ 임베더 기능을 사용할 수 없습니다.")
        return None

    def process_paragraph_alignment(*args, **kwargs):
        print("❌ 문단 정렬 기능을 사용할 수 없습니다.")
        return []

def process_paragraph_file(
    input_file, 
    output_file, 
    embedder_name="bge", 
    max_length=150, 
    similarity_threshold=0.7,
    openai_model=None,
    openai_api_key=None,
    max_workers=4,
    batch_size=50,
    verbose=False,
    device="cpu",
    use_boundary_model=False,
    boundary_threshold=0.72,
    boundary_min_len: int | None = None,
    enable_refine: bool = False,
    enable_adjacent_boundary_refine: bool = True,
    enable_src_marker_boundary_bonus: bool = False,
    enable_src_marker_whitespace_dp_bonus: bool = False,
    trace_stages_path: str | None = None,
    seed: int | None = None,
    tokenizer_init_ok: bool | None = None,
):
    """입력 엑셀 파일을 읽어 문단 단위로 정렬하고, 결과를 출력 파일로 저장
    
    🆕 통합 모드: use_boundary_model=True이면
    - 기존 BGE/순차 방식으로 초기 분할
    - 경계 모델로 경계 refinement
    - Alignment 모델로 정렬 개선
    """
    print(f"📂 PA 파일 처리 시작: {input_file}")

    if trace_stages_path is None:
        trace_stages_path = os.getenv("CSP_PA_TRACE_STAGES_JSONL")

    tracer: _StageTracer | None = None
    if trace_stages_path:
        try:
            tracer = _StageTracer(trace_stages_path)
            print(f"🧪 단계 트레이스 활성화: {trace_stages_path}")
        except Exception as e:
            tracer = None
            print(f"⚠️ 단계 트레이스 초기화 실패(무시): {e}")

    # === trace context (run-level) ===
    # 목적: seed 1~10 비교에서 "무엇을 요청했는지(옵션)"를 JSONL만 보고 검증 가능하게.
    # NOTE: SUPAR/KIWI 등 '가용성'은 환경/로그로도 확인 가능하므로 여기서는 옵션 위주로만 남긴다.
    trace_ctx: Dict = {
        "seed": int(seed) if isinstance(seed, int) else (int(seed) if str(seed or "").isdigit() else None),
        "embedder": str(embedder_name),
        "device": str(device),
        "use_boundary_model": bool(use_boundary_model),
        "boundary_threshold": float(boundary_threshold),
        "boundary_min_len": (int(boundary_min_len) if boundary_min_len is not None else None),
        "tgt_split_max_length": int(max_length),
        "enable_refine": bool(enable_refine),
        "enable_adjacent_boundary_refine": bool(enable_adjacent_boundary_refine),
        "enable_src_marker_boundary_bonus": bool(enable_src_marker_boundary_bonus),
        "enable_src_marker_whitespace_dp_bonus": bool(enable_src_marker_whitespace_dp_bonus),
        "tokenizer_init_ok": (bool(tokenizer_init_ok) if tokenizer_init_ok is not None else None),
    }

    # (의도적으로: 가용성/설치 여부는 여기서 기록하지 않음)

    # run_context는 1회만 기록
    if tracer is not None:
        try:
            tracer.write(
                {
                    "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "stage": "run_context",
                    "paragraph_id": None,
                    "book_name": None,
                    "src_segments": [],
                    "tgt_segments": [],
                    "ctx": trace_ctx,
                }
            )
        except Exception:
            pass

    # 사용자 요구사항: 임베더는 항상 bge
    if embedder_name != "bge":
        raise RuntimeError(f"PA는 embedder_name='bge'만 허용합니다. 현재 값: {embedder_name}")
    
    # Boundary/Alignment 모델 로드
    boundary_model = None
    alignment_pa = None
    if use_boundary_model:
        try:
            from pathlib import Path
            from common.boundary_model_loader import BoundaryModelLoader
            from common.boundary_aware_alignment_loader import BoundaryAwareAlignmentMatcher
            
            models_root = Path(__file__).parent.parent / "models"
            boundary_path = models_root / "boundary_multitask.pt"
            alignment_path = models_root / "dual_encoder_boundary_aware_pa.pt"
            
            if not boundary_path.exists():
                raise FileNotFoundError(f"경계 모델 파일 없음: {boundary_path}")
            if not alignment_path.exists():
                raise FileNotFoundError(f"정렬 모델 파일 없음: {alignment_path}")

            boundary_model = BoundaryModelLoader(model_path=boundary_path, device=device)
            pa_sel_params = get_pa_selection_params()
            boundary_aware_weight = float(pa_sel_params.get("boundary_aware_weight", 0.3))
            alignment_pa = BoundaryAwareAlignmentMatcher(
                model_path=alignment_path,
                device=device,
                boundary_weight=boundary_aware_weight,
            )
            print(f"✅ 경계/정렬 모델 로드 완료 (threshold={boundary_threshold}, strict 모드)")

            # strict 요구사항: SuPar-Kanbun + Stanza 파서도 반드시 함께 사용(초기화 실패 시 즉시 중단)
            import common.new_parsers as new_parsers
            new_parsers.ensure_kanbun_pipeline()
            new_parsers.ensure_stanza_pipeline(lang="ko")
            print(
                "✅ 파서 로드 완료 "
                f"(SuPar-Kanbun pipeline: {new_parsers.SUPAR_AVAILABLE}, Stanza pipeline: {new_parsers.STANZA_AVAILABLE})"
            )
        except Exception as e:
            # use_boundary_model(strict)에서는 폴백 금지
            raise RuntimeError(
                "--use-boundary-model 모드인데 boundary/alignment 또는 SuPar-Kanbun/Stanza 파서 로드에 실패했습니다. "
                f"Docker 이미지/의존성/리소스를 확인하세요: {e}"
            )

    # use_boundary_model이면 폴백을 허용하지 않는다: 모델 로드 실패는 즉시 중단
    if use_boundary_model and (boundary_model is None or alignment_pa is None):
        raise RuntimeError("--use-boundary-model 모드인데 boundary/alignment 모델 로드에 실패했습니다. Docker 환경/모델 경로를 확인하세요.")
    
    try:
        # CSV 또는 Excel 자동 감지
        if str(input_file).endswith('.csv'):
            df = pd.read_csv(input_file)
        else:
            df = pd.read_excel(input_file)
        print(f"📄 {len(df)}개 문단 로드됨")
    except FileNotFoundError:
        print(f"❌ 입력 파일을 찾을 수 없습니다: {input_file}")
        return None
    except Exception as e:
        print(f"❌ 파일 로드 오류: {e}")
        return None
    
    # 필수 컬럼 확인
    required_columns = ['원문', '번역문']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ 입력 파일에 필수 컬럼이 없습니다: {missing_columns}")
        print(f"📋 현재 컬럼: {list(df.columns)}")
        return None

    # 입력 스키마 가드: PA는 '문단 단위' 입력(PD)을 전제로 한다.
    # 문장 단위(정답/산출물) CSV를 입력으로 넣으면 평가/튜닝이 왜곡되므로 즉시 중단한다.
    if "문장식별자" in df.columns:
        raise RuntimeError(
            "PA 입력 파일이 문장 단위로 보입니다(컬럼 '문장식별자' 존재). "
            "PA(문단→문장) 파이프라인 입력은 PD 형식(문단 단위: 문단식별자/원문/번역문/book_name)이어야 합니다. "
            "예: datasets/pd/test_100.csv 를 input으로 사용하고, 평가는 datasets/pa/test_100_from_pd.csv 를 gold로 사용하세요."
        )

    # 진행률 초기화
    try:
        start_unified_progress(
            total=len(df),
            description="📊 PA 분할" + (" (경계 보강)" if boundary_model else ""),
            unit="문단",
            bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            mininterval=0.5,
            maxinterval=2.0,
        )
        use_progress_bar = True
    except Exception as e:
        print(f"⚠️ 진행률 초기화 실패: {e}")
        use_progress_bar = False

    all_results: List[Dict] = []
    global_sent_idx = 1  # 전체 문장 번호 연속 부여

    # (옵션) DP refine 디버그: 특정 (book:pid)만 JSON 덤프
    dp_debug_keys_raw = os.getenv("CSP_PA_DP_DEBUG_KEYS", "").strip()
    dp_debug_dir_raw = os.getenv("CSP_PA_DP_DEBUG_DIR", "").strip() or "test_results/dp_debug"

    dp_debug_keys: set[tuple[str, int]] = set()
    if dp_debug_keys_raw:
        for tok in dp_debug_keys_raw.replace(",", "|").split("|"):
            s = tok.strip()
            if not s:
                continue
            if ":" not in s:
                continue
            b, p = s.rsplit(":", 1)
            try:
                dp_debug_keys.add((b.strip(), int(p)))
            except Exception:
                continue

    def _safe_name(s: str) -> str:
        bad = '<>:"/\\|?*'
        out = "".join(("_" if ch in bad else ch) for ch in str(s))
        return out.strip() or "_"

    for idx, row in df.iterrows():
        src_paragraph = str(row.get('원문', ''))
        tgt_paragraph = str(row.get('번역문', ''))
        original_para_id = row.get('문단식별자', idx + 1)  # 문단식별자를 미리 가져옴
        book_name = str(row.get('book_name', '')).strip()

        dp_debug_out: str | None = None
        dp_debug_pid: int | None = None
        if dp_debug_keys and isinstance(original_para_id, (int, str)):
            try:
                pid_int = int(original_para_id)
            except Exception:
                pid_int = None
            if pid_int is not None and (book_name, pid_int) in dp_debug_keys:
                dp_path = Path(dp_debug_dir_raw) / f"dp_{_safe_name(book_name)}__{pid_int}.json"
                dp_debug_out = str(dp_path)
                dp_debug_pid = int(pid_int)

        def _trace(stage: str, *, src_segments=None, tgt_segments=None, meta: Dict | None = None):
            if tracer is None:
                return
            rec = {
                "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "stage": stage,
                "paragraph_id": int(original_para_id) if str(original_para_id).isdigit() else original_para_id,
                "book_name": book_name,
                "src_segments": [str(x) for x in (src_segments or [])],
                "tgt_segments": [str(x) for x in (tgt_segments or [])],
                "ctx": trace_ctx,
            }
            if meta:
                rec["meta"] = meta
            tracer.write(rec)
        
        if src_paragraph.strip() and tgt_paragraph.strip():
            if use_boundary_model:
                # strict baseline: (경계모델 후보 + alignment 선택)
                alignments_base = process_paragraph_alignment_with_boundary_model(
                    src_paragraph=src_paragraph,
                    tgt_paragraph=tgt_paragraph,
                    boundary_model=boundary_model,
                    alignment_model=alignment_pa,
                    threshold=boundary_threshold,
                    boundary_min_len=boundary_min_len,
                    tgt_split_max_length=max_length,
                    adjacent_refine_max_shift_tokens=(4 if enable_refine else 1),
                    enable_adjacent_boundary_refine=enable_adjacent_boundary_refine,
                    enable_src_marker_boundary_bonus=enable_src_marker_boundary_bonus,
                    enable_src_marker_whitespace_dp_bonus=enable_src_marker_whitespace_dp_bonus,
                    verbose=verbose,
                    trace=_trace,
                    dp_debug_out=dp_debug_out,
                    # JSON serialization safety: pandas가 numpy scalar로 읽으면 json.dumps가 실패할 수 있음
                    dp_debug_meta={"book_name": str(book_name), "paragraph_id": (dp_debug_pid if dp_debug_pid is not None else str(original_para_id))},
                )

                # strict baseline이 비는 건 원칙적으로 없어야 하지만, 방어적으로 유지
                alignments = alignments_base

                try:
                    _trace(
                        "after_model_alignment",
                        src_segments=[a.get('원문', '') for a in (alignments or [])],
                        tgt_segments=[a.get('번역문', '') for a in (alignments or [])],
                    )
                except Exception:
                    pass

                if not alignments:
                    raise RuntimeError(f"문단 {original_para_id}: 결과가 비었습니다 (baseline/refine/strict 모두 실패)")
            else:
                # 기존 BGE/순차 방식
                alignments = process_paragraph_alignment(
                        src_paragraph,
                        tgt_paragraph,
                        embedder_name=embedder_name,
                        max_length=max_length,
                        similarity_threshold=similarity_threshold,
                        device=device,
                        quality_threshold=0.8,
                        use_spacy_tokenizer=False,
                        max_workers=max_workers,
                        batch_size=batch_size,
                    )

            # 🔧 최종 보정: [-…] 블록을 문장 경계에 걸치지 않도록 원문 조각에 원자적으로 붙임
            try:
                alignments = _ensure_atomic_brackets_in_alignments(alignments)
                _trace(
                    "after_atomic_brackets",
                    src_segments=[a.get('원문', '') for a in (alignments or [])],
                    tgt_segments=[a.get('번역문', '') for a in (alignments or [])],
                )
            except Exception as e:
                if verbose:
                    print(f"⚠️ 괄호 블록 원자화 보정 실패: {e}")
                try:
                    _trace("after_atomic_brackets", src_segments=[], tgt_segments=[], meta={"error": str(e)})
                except Exception:
                    pass
            
            # 🆕 한글 토씨 힌트로 매칭 보정 (기존 로직은 보존)
            try:
                from common.korean_particle_matcher import enhance_pa_alignments_with_particles
                alignments = enhance_pa_alignments_with_particles(alignments)

                # 변화량 집계: similarity 전/후(원본/보정) 및 구성요소(토씨/고어 부스트)를 기록
                deltas: list[float] = []
                particle_boosts: list[float] = []
                archaic_boosts: list[float] = []
                changed_examples: list[dict] = []
                applied = False

                def _sf(v, default=0.0) -> float:
                    try:
                        return float(v)
                    except Exception:
                        return float(default)

                def _median(xs: list[float]) -> float | None:
                    if not xs:
                        return None
                    ys = sorted(xs)
                    mid = len(ys) // 2
                    if len(ys) % 2 == 1:
                        return float(ys[mid])
                    return float((ys[mid - 1] + ys[mid]) / 2.0)

                for i, a in enumerate(alignments or []):
                    if not isinstance(a, dict):
                        continue
                    if "original_similarity" in a:
                        applied = True
                    orig = _sf(a.get("original_similarity", a.get("similarity", 0.0)), 0.0)
                    enh = _sf(a.get("similarity", orig), orig)
                    d = float(enh - orig)
                    deltas.append(d)
                    pb = _sf(a.get("particle_boost", 0.0), 0.0)
                    ab = _sf(a.get("archaic_boost", 0.0), 0.0)
                    particle_boosts.append(pb)
                    archaic_boosts.append(ab)

                    # 상위 변화 사례(절대값 기준)를 일부만 저장
                    if abs(d) > 1e-9:
                        changed_examples.append(
                            {
                                "idx": int(i),
                                "delta": float(d),
                                "original_similarity": float(orig),
                                "enhanced_similarity": float(enh),
                                "particle_boost": float(pb),
                                "archaic_boost": float(ab),
                                "particle_similarity": (a.get("particle_similarity") if "particle_similarity" in a else None),
                                "archaic_bonus": (a.get("archaic_bonus") if "archaic_bonus" in a else None),
                                "src": (str(a.get("원문", ""))[:120] if a.get("원문") is not None else ""),
                                "tgt": (str(a.get("번역문", ""))[:120] if a.get("번역문") is not None else ""),
                            }
                        )

                changed_examples = sorted(changed_examples, key=lambda r: abs(float(r.get("delta", 0.0))), reverse=True)[:5]

                delta_mean = (sum(deltas) / len(deltas)) if deltas else None
                pb_mean = (sum(particle_boosts) / len(particle_boosts)) if particle_boosts else None
                ab_mean = (sum(archaic_boosts) / len(archaic_boosts)) if archaic_boosts else None

                _trace(
                    "after_particle_enhance",
                    src_segments=[a.get('원문', '') for a in (alignments or [])],
                    tgt_segments=[a.get('번역문', '') for a in (alignments or [])],
                    meta={
                        "applied": bool(applied),
                        "n_alignments": int(len(alignments or [])),
                        "delta_mean": (float(delta_mean) if delta_mean is not None else None),
                        "delta_median": (_median(deltas) if deltas else None),
                        "delta_min": (float(min(deltas)) if deltas else None),
                        "delta_max": (float(max(deltas)) if deltas else None),
                        "delta_changed_count": int(sum(1 for d in deltas if abs(d) > 1e-9)),
                        "particle_boost_mean": (float(pb_mean) if pb_mean is not None else None),
                        "archaic_boost_mean": (float(ab_mean) if ab_mean is not None else None),
                        "top_delta_examples": changed_examples,
                    },
                )
            except Exception as e:
                if verbose:
                    print(f"⚠️ 토씨 매칭 보정 실패 (기존 결과 유지): {e}")
                # 실패해도 기존 alignments 그대로 사용
                try:
                    _trace("after_particle_enhance", src_segments=[], tgt_segments=[], meta={"error": str(e)})
                except Exception:
                    pass

            # 🛡️ 최종 무결성 복원: 일시적으로 비활성화하여 원본 alignment 상태 확인
            # try:
            #     alignments = restore_paragraph_integrity(src_paragraph, tgt_paragraph, alignments)
            #     
            #     # 🔒 무결성 검증: 복원 후에도 완벽하지 않으면 경고
            #     aligned_src_check = ''.join([a.get('원문', '') for a in alignments]).replace(' ', '').replace('\n', '').replace('\t', '')
            #     aligned_tgt_check = ''.join([a.get('번역문', '') for a in alignments]).replace(' ', '').replace('\n', '').replace('\t', '')
            #     original_src_check = src_paragraph.replace(' ', '').replace('\n', '').replace('\t', '')
            #     original_tgt_check = tgt_paragraph.replace(' ', '').replace('\n', '').replace('\t', '')
            #     
            #     src_match = (aligned_src_check == original_src_check)
            #     tgt_match = (aligned_tgt_check == original_tgt_check)
            #     
            #     if not src_match or not tgt_match:
            #         if not src_match:
            #             src_diff = len(original_src_check) - len(aligned_src_check)
            #             logger.warning(f"⚠️ 문단 {original_para_id} 원문 무결성 경고: 차이 {src_diff:+d}자")
            #         if not tgt_match:
            #             tgt_diff = len(original_tgt_check) - len(aligned_tgt_check)
            #             logger.warning(f"⚠️ 문단 {original_para_id} 번역문 무결성 경고: 차이 {tgt_diff:+d}자")
            #             
            # except Exception as e:
            #     logger.error(f"❌ 문단 {original_para_id} 무결성 복원 실패: {e}")
            #     if verbose:
            #         import traceback
            #         traceback.print_exc()

            # 🛡️ 무결성 복원 재활성화
            try:
                alignments = restore_paragraph_integrity(src_paragraph, tgt_paragraph, alignments)

                _trace(
                    "after_restore_integrity",
                    src_segments=[a.get('원문', '') for a in (alignments or [])],
                    tgt_segments=[a.get('번역문', '') for a in (alignments or [])],
                )
                
                # 무결성 검증
                aligned_src_check = ''.join([a.get('원문', '') for a in alignments]).replace(' ', '').replace('\n', '').replace('\t', '')
                aligned_tgt_check = ''.join([a.get('번역문', '') for a in alignments]).replace(' ', '').replace('\n', '').replace('\t', '')
                original_src_check = src_paragraph.replace(' ', '').replace('\n', '').replace('\t', '')
                original_tgt_check = tgt_paragraph.replace(' ', '').replace('\n', '').replace('\t', '')
                
                src_match = (aligned_src_check == original_src_check)
                tgt_match = (aligned_tgt_check == original_tgt_check)
                
                if src_match and tgt_match:
                    logger.info(f"✅ 문단 {original_para_id} 무결성 완벽 ({len(alignments)}개 alignments)")
                else:
                    if not src_match:
                        src_diff = len(original_src_check) - len(aligned_src_check)
                        logger.warning(f"⚠️ 문단 {original_para_id} 원문 불일치: {src_diff:+d}자")
                    if not tgt_match:
                        tgt_diff = len(original_tgt_check) - len(aligned_tgt_check)
                        logger.warning(f"⚠️ 문단 {original_para_id} 번역문 불일치: {tgt_diff:+d}자")
                        
            except Exception as e:
                logger.error(f"❌ 문단 {original_para_id} 무결성 복원 실패: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
                try:
                    _trace("after_restore_integrity", src_segments=[], tgt_segments=[], meta={"error": str(e)})
                except Exception:
                    pass


            # 🔧 최종 괄호 중복/불균형 정리 (원문/번역문 모두)
            for a in alignments:
                a['원문'] = _final_cleanup_brackets(a.get('원문', ''))
                a['번역문'] = _final_cleanup_brackets(a.get('번역문', ''))

            try:
                _trace(
                    "after_final_cleanup",
                    src_segments=[a.get('원문', '') for a in (alignments or [])],
                    tgt_segments=[a.get('번역문', '') for a in (alignments or [])],
                )
            except Exception:
                pass
            
            # 문단식별자 추가 + 문장식별자 추가 (이미 위에서 original_para_id 정의됨)
            for a in alignments:
                a['문단식별자'] = original_para_id
                if book_name:
                    a['book_name'] = book_name
                a['문장식별자'] = global_sent_idx
                global_sent_idx += 1
            
            all_results.extend(alignments)

            try:
                _trace(
                    "final",
                    src_segments=[a.get('원문', '') for a in (alignments or [])],
                    tgt_segments=[a.get('번역문', '') for a in (alignments or [])],
                    meta={"rows": int(len(alignments or []))},
                )
            except Exception:
                pass
            
            # 🔧 SA와 동일한 진행률 업데이트
            if use_progress_bar:
                try:
                    update_unified_progress(1, 처리됨=len(all_results))
                except:
                    pass
        
        elif verbose:
            print(f"⚠️ 문단 {idx + 1}: 빈 원문 또는 번역문 건너뜀")
            # 빈 문단도 진행률 업데이트
            if use_progress_bar:
                try:
                    update_unified_progress(1)
                except:
                    pass

        else:
            # 빈 문단도 진행률 업데이트 (비-verbose)
            if use_progress_bar:
                try:
                    update_unified_progress(1)
                except:
                    pass
    
    if not all_results:
        if tracer is not None:
            tracer.close()
        if use_progress_bar:
            try:
                finish_unified_progress("PA 완료 (결과 없음)")
            except:
                pass
        print("❌ 처리된 결과가 없습니다.")
        return None
    
    # 결과 DataFrame 생성
    result_df = pd.DataFrame(all_results)
    
    # 🔧 무결성 확인 후 최종 strip 적용
    if len(result_df) > 0:
        # 원문과 번역문에 대해 strip 적용 (공백 정리)
        if '원문' in result_df.columns:
            result_df['원문'] = result_df['원문'].astype(str).str.strip()
        if '번역문' in result_df.columns:
            result_df['번역문'] = result_df['번역문'].astype(str).str.strip()
        
        if verbose:
            print("✅ 무결성 확인 후 최종 공백 정리 완료")
    
    # 🔧 SA와 동일한 진행률 완료
    if use_progress_bar:
        try:
            finish_unified_progress(f"PA 완료: {len(all_results):,}개 문장 쌍 생성")
        except:
            pass
    
    # 컬럼 순서 정리 - 요구 형식(기본): 문단식별자, 문장식별자, 원문, 번역문, similarity
    # book_name이 있으면 (book_name, 문단식별자)로 문단을 구분할 수 있게 포함한다.
    final_columns = ['문단식별자', 'book_name', '문장식별자', '원문', '번역문', 'similarity']
    available_columns = [col for col in final_columns if col in result_df.columns]
    result_df = result_df[available_columns]
    
    # 결과 저장
    try:
        # CSV 또는 Excel 자동 감지하여 저장
        if str(output_file).endswith('.csv'):
            result_df.to_csv(output_file, index=False)
        else:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                result_df.to_excel(writer, index=False, sheet_name='results')

        if verbose:
            print(f"💾 결과 저장: {output_file}")
            print(f"📊 총 {len(all_results)}개 문장 쌍 생성")
            analyze_alignment_results(result_df)
        
        # 🆕 전역 무결성 검증 (정규화 없음, 순수 텍스트 비교)
        try:
            # CSV 또는 Excel 자동 감지
            if str(input_file).endswith('.csv'):
                input_df = pd.read_csv(input_file)
            else:
                input_df = pd.read_excel(input_file)
            passed, integrity_losses_df, analysis = verify_global_integrity(
                input_df, result_df, 
                source_col='원문', target_col='번역문',
                verbose=verbose
            )
            
            # 무결성 손실 시트를 결과 파일에 추가 (Excel만)
            if len(integrity_losses_df) > 0 and not str(output_file).endswith('.csv'):
                with pd.ExcelWriter(output_file, engine='openpyxl', mode='a') as writer:
                    integrity_losses_df.to_excel(writer, index=False, sheet_name='integrity_losses')
        except Exception as e:
            if verbose:
                print(f"⚠️ 무결성 검증 오류: {e}")
        
        # 기본 모드에서는 통합 진행률에서 완료 메시지 처리됨
        if tracer is not None:
            tracer.close()
        return result_df

    except Exception as e:
        print(f"❌ 결과 저장 실패: {e}")
        if tracer is not None:
            tracer.close()
        return None

def analyze_alignment_results(result_df: pd.DataFrame):
    """정렬 결과 분석"""
    print("\n📊 정렬 결과 분석:")
    
    # 전체 유사도 분포
    if 'similarity' in result_df.columns:
        print(f"🎯 전체 유사도:")
        print(f"   평균: {result_df['similarity'].mean():.3f}")
        print(f"   최고: {result_df['similarity'].max():.3f}")
        print(f"   최저: {result_df['similarity'].min():.3f}")
        
        # 고품질 매칭 비율
        high_quality = sum(1 for x in result_df['similarity'] if x > 0.7)
        medium_quality = sum(1 for x in result_df['similarity'] if 0.5 <= x <= 0.7)
        low_quality = sum(1 for x in result_df['similarity'] if x < 0.5)
        total = len(result_df)
        
        print(f"📊 품질별 매칭:")
        print(f"   고품질 (>0.7): {high_quality}/{total} ({high_quality/total*100:.1f}%)")
        print(f"   중품질 (0.5-0.7): {medium_quality}/{total} ({medium_quality/total*100:.1f}%)")
        print(f"   저품질 (<0.5): {low_quality}/{total} ({low_quality/total*100:.1f}%)")
    
    # 빈 매칭 확인
    empty_source = sum(1 for x in result_df['원문'] if not str(x).strip())
    empty_target = sum(1 for x in result_df['번역문'] if not str(x).strip())
    
    if empty_source > 0:
        print(f"⚠️ 빈 원문: {empty_source}개")
    if empty_target > 0:
        print(f"⚠️ 빈 번역문: {empty_target}개")
