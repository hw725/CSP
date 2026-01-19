import argparse
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import difflib


def detect_columns(df: pd.DataFrame) -> Tuple[str, str, str, str | None]:
    def norm(s: str) -> str:
        s = (s or '').replace('\ufeff', '')
        s = re.sub(r"[\s_\-]+", "", s)
        return s.lower()

    # 원본명 -> 정규화명 맵
    norm_map = {c: norm(str(c)) for c in df.columns}

    # 후보(특정 -> 일반 순)
    cand_id = [
        '문장식별자', '문장식별', '문장번호', '문장번', '문장id', '문장id',
        'sentenceid', 'sentence', 'sentence_id', 'sentid', 'sent_id', 'id', '번호'
    ]
    cand_src = ['원문', 'source', 'sourcetext', 'original', 'chinese', 'src']
    cand_tgt = ['번역문', 'target', 'targettext', 'translation', 'korean', 'tgt']
    cand_seg = ['구식별자', '구식별', 'segmentid', 'segment_id', 'segid', 'seg_id']

    def pick(cands):
        # 1) 완전 일치
        for orig, nm in norm_map.items():
            for c in cands:
                if nm == norm(c):
                    return orig
        # 2) 부분 포함
        for orig, nm in norm_map.items():
            for c in cands:
                if norm(c) in nm:
                    return orig
        return None

    id_col = pick(cand_id)
    src_col = pick(cand_src)
    tgt_col = pick(cand_tgt)
    seg_col = pick(cand_seg)
    if not src_col or not tgt_col:
        raise ValueError(f"필수 컬럼 감지 실패: src={src_col}, tgt={tgt_col}")
    # id는 없을 수 있음(추후 보완)
    return id_col, src_col, tgt_col, seg_col


def normalize_ws(s: str) -> str:
    if s is None:
        return ''
    s = str(s)
    s = re.sub(r"\s+", "", s)
    return s


def seq_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def group_by_id(df: pd.DataFrame, id_col: str | None, src_col: str, tgt_col: str, seg_col: str | None) -> Dict[int, List[Tuple[str, str, int]]]:
    out: Dict[int, List[Tuple[str, str, int]]] = {}
    auto_id = 0
    for _, row in df.iterrows():
        if id_col and pd.notna(row.get(id_col)):
            try:
                sid = int(row[id_col])
            except Exception:
                auto_id += 1
                sid = auto_id
        else:
            auto_id += 1
            sid = auto_id
        src = '' if pd.isna(row[src_col]) else str(row[src_col])
        tgt = '' if pd.isna(row[tgt_col]) else str(row[tgt_col])
        seg = int(row[seg_col]) if (seg_col and pd.notna(row.get(seg_col))) else 10**9
        out.setdefault(sid, []).append((src, tgt, seg))
    # sort by seg then stable index
    for k in list(out.keys()):
        out[k] = sorted(out[k], key=lambda x: x[2])
    return out


def jaccard(a: List[str], b: List[str]) -> float:
    A = set(a)
    B = set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def avg_max_sim(gts: List[str], preds: List[str]) -> float:
    if not gts:
        return 1.0 if not preds else 0.0
    if not preds:
        return 0.0
    sims = []
    for gs in gts:
        sims.append(max(seq_sim(normalize_ws(gs), normalize_ws(ps)) for ps in preds) if preds else 0.0)
    return float(np.mean(sims)) if sims else 0.0


def evaluate_sentence(gt_rows: List[Tuple[str, str, int]], pred_rows: List[Tuple[str, str, int]]) -> Dict[str, float]:
    gt_srcs = [r[0] for r in gt_rows]
    gt_tgts = [r[1] for r in gt_rows]
    pd_srcs = [r[0] for r in pred_rows]
    pd_tgts = [r[1] for r in pred_rows]

    gt_src_full = ''.join(gt_srcs)
    pd_src_full = ''.join(pd_srcs)
    gt_tgt_full = ''.join(gt_tgts)
    pd_tgt_full = ''.join(pd_tgts)

    # Matches (whitespace-insensitive)
    src_match = normalize_ws(gt_src_full) == normalize_ws(pd_src_full)
    tgt_match = normalize_ws(gt_tgt_full) == normalize_ws(pd_tgt_full)
    text_match = src_match and tgt_match

    exact_match = gt_rows == pred_rows  # 엄밀 동등 비교(일반적으로 낮음)
    count_match = len(gt_rows) == len(pred_rows)

    # Similarities
    src_text_sim = seq_sim(normalize_ws(gt_src_full), normalize_ws(pd_src_full))
    tgt_text_sim = seq_sim(normalize_ws(gt_tgt_full), normalize_ws(pd_tgt_full))

    # Partial (source)
    src_j = jaccard([normalize_ws(s) for s in gt_srcs], [normalize_ws(s) for s in pd_srcs])
    src_avg_seg = avg_max_sim(gt_srcs, pd_srcs)
    src_partial = (src_j + src_text_sim + src_avg_seg) / 3.0

    # Partial (target)
    tgt_avg_seg = avg_max_sim(gt_tgts, pd_tgts)
    tgt_partial = tgt_avg_seg  # 간단화(정렬 미사용)
    partial = (src_partial + tgt_partial) / 2.0

    return {
        'exact_match': float(exact_match),
        'segment_count_match': float(count_match),
        'text_match': float(text_match),
        'source_text_match': float(src_match),
        'target_text_match': float(tgt_match),
        'source_text_similarity': src_text_sim,
        'target_text_similarity': tgt_text_sim,
        'source_partial': src_partial,
        'target_partial': tgt_partial,
        'partial': partial,
    }


def summarize(metrics: List[Dict[str, float]], label: str):
    if not metrics:
        print(f"[{label}] 평가할 문장이 없습니다.")
        return
    keys = list(metrics[0].keys())
    arrs = {k: np.array([m[k] for m in metrics], dtype=float) for k in keys}
    print(f"\n===== {label} 요약 =====")
    for k in keys:
        a = arrs[k]
        p50, p75, p90 = np.percentile(a, [50, 75, 90])
        mean = a.mean()
        print(f"{k:>24}: mean={mean:6.3f}  p50={p50:6.3f}  p75={p75:6.3f}  p90={p90:6.3f}")


def run(gt_path: str, pred_path: str, label: str):
    print(f"\n>>> {label}: gt={gt_path} pred={pred_path}")
    gt = pd.read_excel(gt_path)
    pred = pd.read_excel(pred_path)
    id_g, src_g, tgt_g, seg_g = detect_columns(gt)
    id_p, src_p, tgt_p, seg_p = detect_columns(pred)
    G = group_by_id(gt, id_g, src_g, tgt_g, seg_g)
    P = group_by_id(pred, id_p, src_p, tgt_p, seg_p)
    common_ids = sorted(set(G.keys()) & set(P.keys()))
    metrics = [evaluate_sentence(G[i], P[i]) for i in common_ids]
    summarize(metrics, label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pa-gt', default='accuracy/pa03.xlsx')
    ap.add_argument('--pa-pred', default='p2s/output_test.xlsx')
    ap.add_argument('--sa-gt', default='accuracy/sa01.xlsx')
    ap.add_argument('--sa-pred', default='s2p/output_test.xlsx')
    ap.add_argument('--base', default=None, help='옵션: 공통 기본 경로. 지정 시 상대 경로에 prefix로 사용')
    args = ap.parse_args()

    def resolve(p: str) -> str:
        # 절대 경로이면서 존재하면 그대로 사용
        if os.path.isabs(p) and os.path.exists(p):
            return p
        # 인자 base 우선
        if args.base:
            cand = os.path.join(args.base, p)
            if os.path.exists(cand):
                return cand
        # 현재 작업 디렉터리 기준(상대 경로) 존재 시
        if os.path.exists(p):
            return p
        # Docker 기본 경로도 시도
        docker_base = '/workspace'
        cand2 = os.path.join(docker_base, p)
        if os.path.exists(cand2):
            return cand2
        # 마지막으로 원본 문자열 반환(존재 검사는 호출부에서)
        return p

    paths = {
        'PA': (resolve(args.pa_gt), resolve(args.pa_pred)),
        'SA': (resolve(args.sa_gt), resolve(args.sa_pred)),
    }
    for label, (gt, pred) in paths.items():
        if not (os.path.exists(gt) and os.path.exists(pred)):
            print(f"\n>>> {label}: 파일이 없습니다 (gt={gt}, pred={pred}) — 건너뜀")
            continue
        run(gt, pred, label)


if __name__ == '__main__':
    main()
