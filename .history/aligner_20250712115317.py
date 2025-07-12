<<<<<<< HEAD
"""원문과 번역문 구 간의 정렬 알고리즘 모듈"""

import logging
import numpy as np
from typing import List, Tuple, Any, Callable

logger = logging.getLogger(__name__)

def cosine_similarity(vec1: Any, vec2: Any) -> float:
    """Calculate cosine similarity (handling zero vectors)."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def align_src_tgt(
    src_units: List[str], 
    tgt_units: List[str], 
    embed_func: Callable
) -> List[Tuple[str, str]]:
    """Align source and target units."""
    logger.info(f"Source units: {len(src_units)} items, Target units: {len(tgt_units)} items")

    if len(src_units) != len(tgt_units):
        try:
            # 지연 임포트로 순환 참조 방지
            from tokenizer import split_tgt_by_src_units_semantic
            
            flatten_tgt = " ".join(tgt_units)
            new_tgt_units = split_tgt_by_src_units_semantic(
                src_units, flatten_tgt, embed_func, min_tokens=1
            )
            if len(new_tgt_units) == len(src_units):
                logger.info("Semantic re-alignment successful")
                return list(zip(src_units, new_tgt_units))
            else:
                logger.warning(f"Length mismatch after re-alignment: Source={len(src_units)}, Target={len(new_tgt_units)}")
        except Exception as e:
            logger.error(f"Error during semantic re-alignment: {e}")

        # 길이가 맞지 않으면 패딩
        if len(src_units) > len(tgt_units):
            tgt_units.extend([""] * (len(src_units) - len(tgt_units)))
        else:
            src_units.extend([""] * (len(tgt_units) - len(src_units)))

    return list(zip(src_units, tgt_units))

def calculate_alignment_matrix(src_embs, tgt_embs, batch_size=512):
    """Optimized function for calculating large similarity matrices."""
    src_len, tgt_len = len(src_embs), len(tgt_embs)
    similarity_matrix = np.zeros((src_len, tgt_len))

    for i in range(0, src_len, batch_size):
        batch_src = src_embs[i:i + batch_size]
        for j in range(0, tgt_len, batch_size):
            batch_tgt = tgt_embs[j:j + batch_size]
            batch_src_norm = np.linalg.norm(batch_src, axis=1, keepdims=True)
            batch_tgt_norm = np.linalg.norm(batch_tgt, axis=1, keepdims=True)

            dots = np.matmul(batch_src, batch_tgt.T)
            norms = np.matmul(batch_src_norm, batch_tgt_norm.T)
            batch_sim = dots / (norms + 1e-8)

            similarity_matrix[i:i + batch_size, j:j + batch_size] = batch_sim

    return similarity_matrix

def align_with_dynamic_programming(
    src_units: List[str],
    tgt_units: List[str],
    embed_func: Callable,
    min_similarity: float = 0.3
) -> List[Tuple[str, str]]:
    """
    동적 프로그래밍을 사용한 고급 정렬 알고리즘
    
    Args:
        src_units: 원문 단위 리스트
        tgt_units: 번역문 단위 리스트
        embed_func: 임베딩 함수
        min_similarity: 최소 유사도 임계값
    
    Returns:
        정렬된 (원문, 번역문) 튜플 리스트
    """
    if not src_units or not tgt_units:
        return []
    
    # 임베딩 계산
    src_embs = embed_func(src_units)
    tgt_embs = embed_func(tgt_units)
    
    # 유사도 매트릭스 계산
    similarity_matrix = calculate_alignment_matrix(src_embs, tgt_embs)
    
    # DP 테이블 초기화
    m, n = len(src_units), len(tgt_units)
    dp = np.full((m + 1, n + 1), -np.inf)
    backtrack = np.zeros((m + 1, n + 1, 2), dtype=int)
    
    dp[0, 0] = 0.0
    
    # DP 계산
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                continue
                
            # 원문 단위 건너뛰기 (삭제)
            if i > 0 and dp[i-1, j] > dp[i, j]:
                dp[i, j] = dp[i-1, j] - 0.1  # 패널티
                backtrack[i, j] = [i-1, j]
            
            # 번역문 단위 건너뛰기 (삽입)
            if j > 0 and dp[i, j-1] > dp[i, j]:
                dp[i, j] = dp[i, j-1] - 0.1  # 패널티
                backtrack[i, j] = [i, j-1]
            
            # 매칭
            if i > 0 and j > 0:
                sim_score = similarity_matrix[i-1, j-1]
                if sim_score >= min_similarity:
                    match_score = dp[i-1, j-1] + sim_score
                    if match_score > dp[i, j]:
                        dp[i, j] = match_score
                        backtrack[i, j] = [i-1, j-1]
    
    # 백트래킹으로 정렬 경로 찾기
    aligned_pairs = []
    i, j = m, n
    
    while i > 0 or j > 0:
        prev_i, prev_j = backtrack[i, j]
        
        if prev_i == i - 1 and prev_j == j - 1:
            # 매칭
            aligned_pairs.append((src_units[i-1], tgt_units[j-1]))
        elif prev_i == i - 1:
            # 원문만 (번역문 없음)
            aligned_pairs.append((src_units[i-1], ""))
        else:
            # 번역문만 (원문 없음)
            aligned_pairs.append(("", tgt_units[j-1]))
        
        i, j = prev_i, prev_j
    
    return aligned_pairs[::-1]

def align_with_greedy_matching(
    src_units: List[str],
    tgt_units: List[str], 
    embed_func: Callable,
    similarity_threshold: float = 0.4
) -> List[Tuple[str, str]]:
    """
    탐욕적 매칭 알고리즘을 사용한 정렬
    
    Args:
        src_units: 원문 단위 리스트
        tgt_units: 번역문 단위 리스트
        embed_func: 임베딩 함수
        similarity_threshold: 유사도 임계값
        
    Returns:
        정렬된 (원문, 번역문) 튜플 리스트
    """
    if not src_units or not tgt_units:
        return []
    
    # 임베딩 계산
    src_embs = embed_func(src_units)
    tgt_embs = embed_func(tgt_units)
    
    # 유사도 매트릭스 계산
    similarity_matrix = calculate_alignment_matrix(src_embs, tgt_embs)
    
    aligned_pairs = []
    used_tgt = set()
    
    for i, src_unit in enumerate(src_units):
        best_j = -1
        best_score = similarity_threshold
        
        for j, tgt_unit in enumerate(tgt_units):
            if j in used_tgt:
                continue
                
            score = similarity_matrix[i, j]
            if score > best_score:
                best_score = score
                best_j = j
        
        if best_j >= 0:
            aligned_pairs.append((src_unit, tgt_units[best_j]))
            used_tgt.add(best_j)
        else:
            aligned_pairs.append((src_unit, ""))
    
    # 매칭되지 않은 번역문 단위 추가
    for j, tgt_unit in enumerate(tgt_units):
        if j not in used_tgt:
            aligned_pairs.append(("", tgt_unit))
    
    return aligned_pairs
=======
# aligner.py - 원문/번역문 구 간 정렬을 위한 DP 기반 매칭 알고리즘

import numpy as np
from typing import List, Callable, Optional, Any
from embedder import compute_embeddings_with_cache
from tokenizer import split_src_meaning_units, split_tgt_meaning_units, split_tgt_by_src_units_semantic

# ── IO 모듈에서 초기화해서 쓰기 위한 전역 DP 배열 선언
dp_prev_global = np.array([])
dp_curr_global = np.array([])

def cosine_similarity(vec1: Any, vec2: Any) -> float:
    """코사인 유사도 계산 (벡터가 0인 경우 대비)"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def align_src_tgt(src_units, tgt_units, embed_func=compute_embeddings_with_cache):
    if len(src_units) != len(tgt_units):
        flatten_tgt = " ".join(tgt_units)
        new_tgt_units = split_tgt_by_src_units_semantic(src_units, flatten_tgt, embed_func)
        assert len(new_tgt_units) == len(src_units)
        return list(zip(src_units, new_tgt_units))
    else:
        return list(zip(src_units, tgt_units))
    print(f"[DEBUG] align_src_tgt 호출됨 → src_units 개수={len(src_units)}, tgt_units 개수={len(tgt_units)}")


    # 1) 임베딩 계산 (배치 처리)
    src_embs = []
    tgt_embs = []

    # src 임베딩
    for i in range(0, src_len, batch_size):
        batch = src_units[i : i + batch_size]
        emb_batch = embed_func(batch)  # 실제 임베딩 수행 결과를 emb_batch에 할당
        # 디버그 출력: 배치 크기와 반환된 벡터 모양 확인
        print(f"[DEBUG] src 임베딩 → batch 크기={len(batch)}, 반환 벡터 모양={np.array(emb_batch).shape}")
        src_embs.extend(emb_batch)    # tgt 임베딩
    # tgt 임베딩
    for i in range(0, tgt_len, batch_size):
        batch = tgt_units[i : i + batch_size]
        emb_batch = embed_func(batch)  # 실제 임베딩 수행 결과를 emb_batch에 할당
        # 디버그 출력: 배치 크기와 반환된 벡터 모양 확인
        print(f"[DEBUG] tgt 임베딩 → batch 크기={len(batch)}, 반환 벡터 모양={np.array(emb_batch).shape}")
        tgt_embs.extend(emb_batch)
    # 2) DP 배열 초기화 (전역 배열 활용)
    # dp_prev_global, dp_curr_global는 크기가 tgt_len + 1 이어야 함
    if dp_prev_global is None or len(dp_prev_global) < tgt_len + 1:
        dp_prev_global = np.full(tgt_len + 1, -np.inf)
    if dp_curr_global is None or len(dp_curr_global) < tgt_len + 1:
        dp_curr_global = np.full(tgt_len + 1, -np.inf)

    # dp_prev_global[0] = 0으로 시작
    dp_prev_global.fill(-np.inf)
    dp_curr_global.fill(-np.inf)
    dp_prev_global[0] = 0.0

    # 3) DP 테이블 및 backtracking 테이블 초기화
    backtrack = [[None] * (tgt_len + 1) for _ in range(src_len + 1)]

    # 최대 병합 길이 제한
    max_merge_len_cap = min(max_merge_len_cap, tgt_len)

    # 4) DP 알고리즘 수행
    for i in range(1, src_len + 1):
        dp_curr_global.fill(-np.inf)
        for j in range(tgt_len + 1):
            # 이전 상태에서 dp 값 유지 (아무것도 안하는 경우)
            # (실제로 j=0인 경우 제외, 그냥 dp_curr_global[j] = -inf 유지)
            pass

        for j in range(tgt_len + 1):
            if dp_prev_global[j] == -np.inf:
                continue
            # 4-1) src 단위 i (1-based)
            src_idx = i - 1

            # 4-2) tgt 병합 후보 길이: 1 ~ max_merge_len_cap
            for merge_len in range(1, max_merge_len_cap + 1):
                tgt_start = j
                tgt_end = j + merge_len
                if tgt_end > tgt_len:
                    break

                # 병합된 tgt 임베딩 평균 계산
                merged_vec = np.mean(tgt_embs[tgt_start:tgt_end], axis=0)

                # 유사도 계산
                if similarity_metric == "cosine":
                    sim = cosine_similarity(src_embs[src_idx], merged_vec)
                else:
                    raise ValueError(f"지원하지 않는 similarity_metric: {similarity_metric}")

                # 유사도 임계값 체크 (있으면)
                if similarity_threshold is not None and sim < similarity_threshold:
                    continue

                new_score = dp_prev_global[j] + sim
                if new_score > dp_curr_global[tgt_end]:
                    dp_curr_global[tgt_end] = new_score
                    backtrack[i][tgt_end] = (j, merge_len)

        # dp_prev_global 갱신
        dp_prev_global, dp_curr_global = dp_curr_global, dp_prev_global

    # 5) 최적 경로 역추적
    aligned_tgt_units = []
    i = src_len
    j = tgt_len

    # BUG FIX: 역추적이 불가능한 경우(즉, backtrack[i][j]가 None이고 i>0)에는
    # j를 1씩 줄여가며 backtrack[i][j]가 존재하는 위치를 찾고, 그 사이의 tgt_units는 마지막에 합쳐서 반환
    while i > 0:
        if backtrack[i][j] is None:
            # 역추적 실패 시, j를 줄여가며 마지막 가능한 구간을 합쳐서 할당
            found = False
            for jj in range(j-1, -1, -1):
                if backtrack[i][jj] is not None:
                    # 남은 구간 합치기
                    merged_text = "".join(tgt_units[jj:j])
                    aligned_tgt_units.append(merged_text)
                    i -= 1
                    j = jj
                    found = True
                    break
            if not found:
                aligned_tgt_units.append("")
                i -= 1
                # j는 그대로 유지
        else:
            prev_j, merge_len = backtrack[i][j]
            # tgt 병합 구간
            merged_text = "".join(t + " " for t in tgt_units[prev_j : j]).strip()
            aligned_tgt_units.append(merged_text)
            i -= 1
            j = prev_j

    aligned_tgt_units.reverse()

    # 만약 src 길이보다 tgt 병합된 결과가 부족한 경우 빈 문자열로 채움
    while len(aligned_tgt_units) < src_len:
        aligned_tgt_units.append("")

    return aligned_tgt_units
>>>>>>> 0b62e7a (문장-구 분할 스크립트)
