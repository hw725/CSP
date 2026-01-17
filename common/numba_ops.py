import numpy as np
from numba import njit
from numba.typed import List as NumbaList

@njit(cache=True, fastmath=True)
def run_dp_numba(
    n_parts: int,
    text_len: int,
    candidates: NumbaList,     # List[np.ndarray(int32)]
    orig_boundaries: np.ndarray, # array(int32)
    sim_table: np.ndarray,       # 3D array(float32) [start, end, target_idx]
    targets_len: int,
    boundary_bonus_arr: np.ndarray, # 1D array(float32) pre-computed bonus for each global index
    shift_penalty_factor: float = 0.0008,
) -> tuple: # (success_bool, best_total, chosen_indices_array)

    NEG = -1e9
    
    # Pre-scan max candidates for array allocation
    max_cands = 0
    for i in range(len(candidates)):
        l = len(candidates[i])
        if l > max_cands:
            max_cands = l
            
    # dp[i, j]
    # i range: 0 to n_parts-2
    dp = np.full((n_parts - 1, max_cands), NEG, dtype=np.float32)
    back = np.full((n_parts - 1, max_cands), -1, dtype=np.int32)
    
    # 1. Init (i=0)
    cs0 = candidates[0]
    orig0 = orig_boundaries[0]
    
    for j in range(len(cs0)):
        bpos = cs0[j]
        # _seg_ok check assumes bpos > 0.
        
        # sim(0, bpos, 0)
        # Note: 3D array access is fast
        s = sim_table[0, bpos, 0]
        
        # bonus
        # bonus is looked up ideally. 
        # For now, let's assume bonus is 0 or passed in separate array?
        # Using passed boundary_bonus_arr
        bonus = 0.0
        if bpos < len(boundary_bonus_arr):
            bonus = boundary_bonus_arr[bpos]
            
        shift_penalty = shift_penalty_factor * abs(bpos - orig0) 
        # choice_penalty ignored or needs logic
        
        dp[0, j] = s + bonus - shift_penalty

    # 2. Transitions
    for i in range(1, n_parts - 1):
        curr_cs = candidates[i]
        prev_cs = candidates[i-1]
        orig_curr = orig_boundaries[i]
        orig_prev = orig_boundaries[i-1]
        
        for j in range(len(curr_cs)):
            bpos = curr_cs[j]
            best_val = NEG
            best_k = -1
            
            # Bonus/Penalty for current bpos
            bonus = 0.0
            if bpos < len(boundary_bonus_arr):
                bonus = boundary_bonus_arr[bpos]
            shift_penalty = shift_penalty_factor * abs(bpos - orig_curr)
            
            for k in range(len(prev_cs)):
                apos = prev_cs[k]
                
                prev_score = dp[i-1, k]
                if prev_score <= NEG / 2:
                    continue
                if bpos <= apos:
                    continue
                    
                # sim(apos, bpos, i)
                s = sim_table[apos, bpos, i]
                
                val = prev_score + s + bonus - shift_penalty
                
                # Tie-break
                update = False
                if val > best_val + 1e-9:
                    update = True
                elif abs(val - best_val) <= 1e-9 and best_k >= 0:
                    # Tie-break: smaller movement preferred
                    # cur: movement sum of (curr bpos, prev apos)
                    cur_move = abs(bpos - orig_curr) + abs(apos - orig_prev)
                    
                    # prev: movement sum of (curr best's bpos==this bpos, prev best's apos)
                    # Wait, 'prev' logic in original code:
                    # prev = abs(cs[best_k]... ) -> This implies cs is CURRENT candidate set?
                    # No! best_k is index in PREV_CS.
                    # Original Code:
                    # prev = abs(cs[best_k] - orig_i) + abs(prev_cs[best_k] - orig_i_1)
                    # THIS LOOKS BUGGY in Original Python Code or I misread it.
                    # "cs" usually refers to current candidates. "cs[best_k]"? 
                    # best_k is index for prev_cs. 
                    # If the python code said cs[best_k], it might mean j? No best_k is for prev layer.
                    
                    # Let's re-read python code snippet:
                    # cur = abs(bpos - orig_boundaries[i]) + abs(apos - orig_boundaries[i - 1])
                    # prev = abs(cs[best_k] - orig_boundaries[i]) + abs(prev_cs[best_k] - orig_boundaries[i - 1])
                    # Wait, if best_k is candidate index for LAYER i-1, 
                    # why use it to index 'cs' (LAYER i)?
                    # Ah, in the original code, 'best_k' is iterating 'prev_cs'.
                    # But 'best_val' tracks the best accumulated score arriving AT 'bpos'.
                    # So 'best_k' is the index of the best predecessor.
                    
                    # The comparison should be:
                    # "Keep current (apos) vs Replace with (prev_cs[best_k])"
                    # both arrive at same 'bpos'.
                    # So movement of 'bpos' is identical for both.
                    # The only difference is movement of 'apos'.
                    # So: abs(apos - orig_prev) vs abs(prev_cs[best_k] - orig_prev)
                    
                    # The original code's `abs(cs[best_k] - ...)` seems to assume best_k is used for current layer? 
                    # Or maybe it meant `cs[j]` (current bpos)?
                    # If original code has `cs[best_k]`, and `best_k` is from `prev_cs` size... 
                    # unless `cs` and `prev_cs` have same size, this is IndexOutOfBounds potential.
                    # But Python code says `best_k` is `k`. `k` iterates `prev_cs`.
                    # If I strictly follow:
                    #   prev = abs(cs[best_k]...
                    # This implies best_k should be used on current candidates? That makes no sense.
                    # It likely meant: comparing (apos) with (prev_cs[best_k]).
                    # Since bpos is fixed for this inner loop (j fixed), 
                    # `abs(bpos - orig)` is constant.
                    # So we just compare `abs(apos - orig_prev)` vs `abs(prev_apos - orig_prev)`.
                    
                    prev_best_apos = prev_cs[best_k]
                    prev_move = abs(bpos - orig_curr) + abs(prev_best_apos - orig_prev)
                    
                    if cur_move < prev_move:
                        update = True
                        
                if update:
                    best_val = val
                    best_k = k
                    
            dp[i, j] = best_val
            back[i, j] = best_k

    # 3. Finalize
    last_i = n_parts - 2
    best_total = NEG
    best_j = -1
    
    last_cs = candidates[last_i]
    for j in range(len(last_cs)):
        if dp[last_i, j] <= NEG / 2:
            continue
            
        bpos = last_cs[j]
        if bpos >= text_len: # _seg_ok check
            continue
            
        # sim(bpos, len, last_tgt)
        s = sim_table[bpos, text_len, targets_len - 1]
        
        total = dp[last_i, j] + s
        
        # Tie-break (similar to above)
        update = False
        if total > best_total + 1e-9:
            update = True
        elif abs(total - best_total) <= 1e-9 and best_j >= 0:
            # cur: movement sum of (bpos, prev_apos_from_j?) NO.
            # Here we are choosing Final Boundary `bpos`.
            # Comparisons are between different `bpos` (different j).
            # Previous was comparing different `apos` for SAME `bpos`.
            
            # Original code:
            # if abs(bpos - orig_last) < abs(cs[best_j] - orig_last):
            #    update
            
            orig_last_bpos = orig_boundaries[last_i]
            cur_diff = abs(bpos - orig_last_bpos)
            prev_diff = abs(last_cs[best_j] - orig_last_bpos)
            if cur_diff < prev_diff:
                update = True
                
        if update:
            best_total = total
            best_j = j
            
    # Backtrack
    if best_j < 0:
        return (False, 0.0, np.zeros(1, dtype=np.int32))
        
    # We need to return n_parts-1 boundaries
    chosen = np.zeros(n_parts - 1, dtype=np.int32)
    curr = best_j
    for i in range(last_i, -1, -1):
        chosen[i] = candidates[i][curr]
        curr = back[i, curr]
        if i > 0 and curr < 0:
             return (False, 0.0, np.zeros(1, dtype=np.int32))
             
    return (True, best_total, chosen)
