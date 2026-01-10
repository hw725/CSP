#!/usr/bin/env python3
"""무결성 리포트

기본 동작(인자 없음): 전체 43권(엑셀) 길이 기반 무결성 리포트 생성.

검증 모드:
    python integrity_report.py --input <pa_output.csv|xlsx> [--source <paragraph_input.csv|xlsx>]

정답(gold) 비교 모드:
    python integrity_report.py --input <pa_output.csv|xlsx> --gold <gold_sentences.csv|xlsx> [--pids 10 12 ...] [--book-name <name>]

정답(gold) 부분 추출 모드:
    python integrity_report.py --extract --gold <gold_sentences.csv|xlsx> --out <subset.csv>
        [--pids 10 12 ...] [--book-name <name>] [--out-paragraph <subset_paragraphs.csv>]

검증 모드에서는 아래를 확인합니다.
    - 빈 원문/번역문 행 0
    - 문단별 결합 무결성(정규화: 공백/개행/탭 제거)
    - 번역문 분할(문장 수/순서) 불변: rule-based splitter 결과와 PA 출력 번역문 리스트가 동일

정답(gold) 비교 모드에서는 아래를 확인합니다.
    - 번역문 분할(문장 수/순서) 정답과 완전일치
    - 원문 경계 F1 (정규화 기준, 문장 경계 위치 집합)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import re
import difflib
import statistics

import pandas as pd

books = [
    "예기집설대전1", "예기집설대전2",
    "춘추좌씨전1", "춘추좌씨전2", "춘추좌씨전3", "춘추좌씨전4", 
    "춘추좌씨전5", "춘추좌씨전6", "춘추좌씨전7", "춘추좌씨전8",
    "자치통감강목1", "자치통감강목2", "자치통감강목3", "자치통감강목4",
    "자치통감강목5", "자치통감강목6", "자치통감강목7",
    "당시삼백수1", "당시삼백수2", "당시삼백수3",
    "당송팔대가문초한유1", "당송팔대가문초한유2", "당송팔대가문초한유3",
    "당송팔대가문초유종원1", "당송팔대가문초유종원2",
    "당송팔대가문초구양수1", "당송팔대가문초구양수2", "당송팔대가문초구양수3",
    "당송팔대가문초구양수4", "당송팔대가문초구양수5", "당송팔대가문초구양수6",
    "당송팔대가문초소순1",
    "당송팔대가문초소식1", "당송팔대가문초소식2", "당송팔대가문초소식3",
    "당송팔대가문초소식4", "당송팔대가문초소식5",
    "당송팔대가문초소철1", "당송팔대가문초소철2", "당송팔대가문초소철3",
    "당송팔대가문초왕안석1", "당송팔대가문초왕안석2",
    "당송팔대가문초증공1",
]


def _read_tabular(path: Path) -> pd.DataFrame:
    if str(path).lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)


def _norm(s: str) -> str:
    return str(s).replace(" ", "").replace("\n", "").replace("\t", "").strip()


def _first_mismatch(a: str, b: str) -> int:
    """두 문자열의 첫 mismatch 인덱스(없으면 -1)."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return -1


def _snippet(s: str, idx: int, radius: int = 40) -> str:
    if idx < 0:
        return ""
    start = max(0, idx - radius)
    end = min(len(s), idx + radius)
    return s[start:end]


def _check_in_order(full: str, parts: list[str]) -> bool:
    """parts를 순서대로 full에서 소비 가능한지(순서 보존) 확인."""
    cursor = 0
    for part in parts:
        if not part:
            return False
        idx = full.find(part, cursor)
        if idx < 0:
            return False
        cursor = idx + len(part)
    return True


def _boundary_positions_normed(segments: list[str]) -> set[int]:
    """정규화 문자열 기준 문장 경계 위치(누적 길이) 집합을 만든다.

    예: [A,B,C]면 {len(A), len(A)+len(B)}.
    """

    positions: set[int] = set()
    cursor = 0
    for i, seg in enumerate(segments):
        seg_norm = _norm(seg)
        cursor += len(seg_norm)
        if i < len(segments) - 1:
            positions.add(cursor)
    return positions


def _prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    # 경계가 원래 존재하지 않는(=문장 1개) 문단에서 pred도 경계가 없으면 완전 일치로 본다.
    # tp=fp=fn=0인 케이스를 0점으로 두면 최저 F1이 왜곡되고, 리포트가 불필요하게 실패 처리된다.
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def _parse_pids(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    out: list[int] = []
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        out.append(int(s))
    return out if out else None


def _load_pid_book_keys(path: Path) -> set[tuple[int, str]]:
    """CSV/XLSX에서 (문단식별자, book_name) 키 집합을 로드한다."""

    df = _read_tabular(path)
    required = {"문단식별자", "book_name"}
    if not required.issubset(set(df.columns)):
        missing = sorted(required - set(df.columns))
        raise SystemExit(f"키 파일에 필수 컬럼이 없습니다: {missing} (필요: 문단식별자, book_name)")

    df = df.copy()
    df["book_name"] = df["book_name"].fillna("")

    keys: set[tuple[int, str]] = set()
    for _, row in df.iterrows():
        pid = int(row["문단식별자"])
        bn = str(row["book_name"]).strip()
        if not bn:
            continue
        keys.add((pid, bn))
    if not keys:
        raise SystemExit(f"키 파일에서 유효한 (문단식별자, book_name) 쌍을 찾지 못했습니다: {path}")
    return keys


def extract_gold_subset(
    gold_path: Path,
    out_path: Path,
    pids: list[int] | None = None,
    book_name: str | None = None,
    out_paragraph_path: Path | None = None,
    keys_from: Path | None = None,
    sa_gold_path: Path | None = None,
    out_sa_path: Path | None = None,
) -> int:
    gold_df = _read_tabular(gold_path)

    required_gold = {"문단식별자", "문장식별자", "원문", "번역문", "book_name"}
    if not required_gold.issubset(set(gold_df.columns)):
        missing = sorted(required_gold - set(gold_df.columns))
        raise SystemExit(f"정답 파일에 필수 컬럼이 없습니다: {missing}")

    gold_df = gold_df.copy()
    for col in ("원문", "번역문", "book_name"):
        gold_df[col] = gold_df[col].fillna("")

    mask = pd.Series(True, index=gold_df.index)

    if keys_from is not None:
        keys = _load_pid_book_keys(keys_from)
        key_mask = gold_df.apply(lambda r: (int(r["문단식별자"]), str(r["book_name"]).strip()) in keys, axis=1)
        mask &= key_mask

    if pids:
        mask &= gold_df["문단식별자"].astype(int).isin([int(x) for x in pids])
    if book_name:
        mask &= (gold_df["book_name"].astype(str) == str(book_name))

    subset = gold_df.loc[mask].copy()
    subset = subset.sort_values(["book_name", "문단식별자", "문장식별자"], kind="stable")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] gold subset 저장: {out_path} (rows={len(subset)})")

    # 문장 단위 정답을 문단 단위로 재구성 (원문/번역문 각각 concat)
    if out_paragraph_path is None:
        # 기본: <out>_paragraphs.csv
        stem = out_path.stem
        out_paragraph_path = out_path.with_name(stem + "_paragraphs.csv")

    para_df = (
        subset.groupby(["book_name", "문단식별자"], sort=False)
        .agg({"원문": lambda xs: "".join([str(x) for x in xs]), "번역문": lambda xs: "".join([str(x) for x in xs])})
        .reset_index()
    )
    # SA 테스트 호환: pd/test_10.csv와 동일 컬럼/순서
    para_df = para_df[["문단식별자", "원문", "번역문", "book_name"]]
    out_paragraph_path.parent.mkdir(parents=True, exist_ok=True)
    para_df.to_csv(out_paragraph_path, index=False, encoding="utf-8-sig")
    print(f"[OK] gold paragraphs 저장: {out_paragraph_path} (rows={len(para_df)})")

    # SA 정답(구병렬)도 함께 추출: (문단식별자,book_name) -> (문장식별자,book_name) 기반
    if sa_gold_path is not None:
        if out_sa_path is None:
            out_sa_path = out_path.with_name(out_path.stem + "_sa_gold.csv")

        sa_df = _read_tabular(sa_gold_path)
        required_sa = {"문장식별자", "구식별자", "원문", "번역문", "book_name"}
        if not required_sa.issubset(set(sa_df.columns)):
            missing = sorted(required_sa - set(sa_df.columns))
            raise SystemExit(f"SA 정답 파일에 필수 컬럼이 없습니다: {missing}")

        sa_df = sa_df.copy()
        for col in ("원문", "번역문", "book_name"):
            sa_df[col] = sa_df[col].fillna("")

        # PA subset에서 (문장식별자, book_name) 키를 만든다.
        sent_keys = set(
            (int(r["문장식별자"]), str(r["book_name"]).strip())
            for _, r in subset[["문장식별자", "book_name"]].iterrows()
        )
        if not sent_keys:
            raise SystemExit("SA gold 추출을 위한 문장 키를 만들지 못했습니다. (PA subset이 비어있음)")

        sa_mask = sa_df.apply(
            lambda r: (int(r["문장식별자"]), str(r["book_name"]).strip()) in sent_keys,
            axis=1,
        )
        sa_subset = sa_df.loc[sa_mask].copy()
        sa_subset = sa_subset.sort_values(["book_name", "문장식별자", "구식별자"], kind="stable")

        out_sa_path.parent.mkdir(parents=True, exist_ok=True)
        sa_subset.to_csv(out_sa_path, index=False, encoding="utf-8-sig")
        print(f"[OK] SA gold(구병렬) subset 저장: {out_sa_path} (rows={len(sa_subset)})")

    return 0


def run_pa_output_vs_gold_report(
    pa_output: Path,
    gold_sentences: Path,
    pids: list[int] | None = None,
    book_name: str | None = None,
    keys_from: Path | None = None,
) -> int:
    """PA 출력(문장 병렬)과 정답(문장 단위)을 직접 비교한다."""

    pred_df = _read_tabular(pa_output)
    gold_df = _read_tabular(gold_sentences)

    required_pred = {"문단식별자", "원문", "번역문"}
    required_gold = {"문단식별자", "문장식별자", "원문", "번역문", "book_name"}
    if not required_pred.issubset(set(pred_df.columns)):
        missing = sorted(required_pred - set(pred_df.columns))
        raise SystemExit(f"PA 출력 파일에 필수 컬럼이 없습니다: {missing}")
    if not required_gold.issubset(set(gold_df.columns)):
        missing = sorted(required_gold - set(gold_df.columns))
        raise SystemExit(f"정답 파일에 필수 컬럼이 없습니다: {missing}")

    pred_df = pred_df.copy()
    gold_df = gold_df.copy()
    for col in ("원문", "번역문"):
        pred_df[col] = pred_df[col].fillna("")
        gold_df[col] = gold_df[col].fillna("")
    gold_df["book_name"] = gold_df["book_name"].fillna("")

    # pid 타입 정규화
    pred_df["문단식별자"] = pred_df["문단식별자"].astype(int)
    gold_df["문단식별자"] = gold_df["문단식별자"].astype(int)

    pred_has_book = "book_name" in pred_df.columns
    if pred_has_book:
        pred_df["book_name"] = pred_df["book_name"].fillna("").astype(str)

    # 필터
    if keys_from is not None:
        keys = _load_pid_book_keys(keys_from)
        key_pids = sorted({pid for pid, _ in keys})
        pred_df = pred_df[pred_df["문단식별자"].isin(key_pids)]
        gold_df = gold_df[gold_df.apply(lambda r: (int(r["문단식별자"]), str(r["book_name"]).strip()) in keys, axis=1)]

    if pids is not None:
        pred_df = pred_df[pred_df["문단식별자"].isin([int(x) for x in pids])]
        gold_df = gold_df[gold_df["문단식별자"].isin([int(x) for x in pids])]
    if book_name:
        gold_df = gold_df[gold_df["book_name"].astype(str) == str(book_name)]

    if pred_has_book:
        pred_groups = pred_df.groupby(["book_name", "문단식별자"], sort=False)
        gold_groups = gold_df.sort_values(["book_name", "문단식별자", "문장식별자"], kind="stable").groupby(
            ["book_name", "문단식별자"], sort=False
        )
        pred_keys = set(pred_groups.groups.keys())
        gold_keys = set(gold_groups.groups.keys())
        common_keys = sorted(pred_keys & gold_keys)
        missing_in_gold = sorted(pred_keys - gold_keys)
        missing_in_pred = sorted(gold_keys - pred_keys)
        total_paras = len(common_keys)
    else:
        # (구버전 출력) book_name이 없는 경우: pid만으로 그룹핑.
        # 단, pid는 book 간 중복될 수 있어 평가가 왜곡될 수 있으므로 경고한다.
        pred_groups = pred_df.groupby("문단식별자", sort=False)
        gold_groups = gold_df.sort_values(["문단식별자", "문장식별자"], kind="stable").groupby("문단식별자", sort=False)
        pred_keys = set(int(k) for k in pred_groups.groups.keys())
        gold_keys = set(int(k) for k in gold_groups.groups.keys())
        common_keys = sorted(pred_keys & gold_keys)
        missing_in_gold = sorted(pred_keys - gold_keys)
        missing_in_pred = sorted(gold_keys - pred_keys)
        total_paras = len(common_keys)

    tp = fp = fn = 0
    tp_ok = fp_ok = fn_ok = 0  # 번역문 문장리스트 완전일치 subset
    # 문장 단위 번역문 완전일치 집계(문단 내부 혼재 가능)
    tgt_sent_exact_ok = 0
    tgt_sent_total_gold = 0
    tgt_sent_total_pred = 0

    # tgt 문장 일치 subset에서 원문 유사도(정규화 문자열 기반)
    src_sim_tgt_sent_ok: list[float] = []
    src_sim_tgt_sent_fail: list[float] = []

    def _src_sim(a: str, b: str) -> float:
        a_n = _norm(a)
        b_n = _norm(b)
        if not a_n and not b_n:
            return 1.0
        if not a_n or not b_n:
            return 0.0
        return float(difflib.SequenceMatcher(None, a_n, b_n).ratio())

    def _stats(xs: list[float]) -> dict:
        if not xs:
            return {
                "n": 0,
                "mean": None,
                "median": None,
                "min": None,
                "p10": None,
                "p90": None,
            }
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        # 단순 분위수(인덱스 기반)
        def _q(p: float) -> float:
            if n == 1:
                return float(xs_sorted[0])
            idx = int(round((n - 1) * p))
            idx = max(0, min(n - 1, idx))
            return float(xs_sorted[idx])

        return {
            "n": n,
            "mean": float(statistics.fmean(xs_sorted)),
            "median": float(statistics.median(xs_sorted)),
            "min": float(xs_sorted[0]),
            "p10": _q(0.10),
            "p90": _q(0.90),
        }
    # book_name이 있는 경우 key는 (book_name, pid)이고, 없는 경우 pid(int)만 쓴다.
    KeyT = tuple[str, int] | int
    tgt_exact_ok: list[KeyT] = []
    tgt_exact_fail: list[KeyT] = []
    worst: tuple[KeyT, float] | None = None  # (key, f1)
    worst_ok: tuple[KeyT, float] | None = None  # (key, f1) for tgt exact ok subset

    for key in common_keys:
        if pred_has_book:
            bk, pid = key
            pred_g = pred_groups.get_group((bk, pid))
            gold_g = gold_groups.get_group((bk, pid))
            key_out: KeyT = (bk, int(pid))
        else:
            pid = int(key)
            pred_g = pred_groups.get_group(pid)
            gold_g = gold_groups.get_group(pid)
            key_out = pid

        pred_src = [str(x).strip() for x in pred_g["원문"].tolist()]
        pred_tgt = [str(x).strip() for x in pred_g["번역문"].tolist()]
        gold_src = [str(x).strip() for x in gold_g["원문"].tolist()]
        gold_tgt = [str(x).strip() for x in gold_g["번역문"].tolist()]

        pred_tgt_norm = [_norm(s) for s in pred_tgt]
        gold_tgt_norm = [_norm(s) for s in gold_tgt]
        tgt_match = (pred_tgt_norm == gold_tgt_norm)
        if tgt_match:
            tgt_exact_ok.append(key_out)
        else:
            tgt_exact_fail.append(key_out)

        # 문장 단위 tgt exact: 같은 문단 안에서도 일부 문장이 일치할 수 있음
        tgt_sent_total_gold += len(gold_tgt_norm)
        tgt_sent_total_pred += len(pred_tgt_norm)
        n_cmp = min(len(pred_tgt_norm), len(gold_tgt_norm))
        for i in range(n_cmp):
            is_ok = (pred_tgt_norm[i] == gold_tgt_norm[i])
            if is_ok:
                tgt_sent_exact_ok += 1
            # src 유사도는 위치가 대응된 구간에서만 기록(길이 불일치로 인덱스가 없으면 skip)
            if i < len(pred_src) and i < len(gold_src):
                sim = _src_sim(pred_src[i], gold_src[i])
                if is_ok:
                    src_sim_tgt_sent_ok.append(sim)
                else:
                    src_sim_tgt_sent_fail.append(sim)

        pred_b = _boundary_positions_normed(pred_src)
        gold_b = _boundary_positions_normed(gold_src)
        inter = pred_b & gold_b

        tp_i = len(inter)
        fp_i = len(pred_b - gold_b)
        fn_i = len(gold_b - pred_b)
        tp += tp_i
        fp += fp_i
        fn += fn_i

        if tgt_match:
            tp_ok += tp_i
            fp_ok += fp_i
            fn_ok += fn_i

        _, _, f1_i = _prf1(tp_i, fp_i, fn_i)
        if worst is None or f1_i < worst[1]:
            worst = (key_out, f1_i)
        if tgt_match and (worst_ok is None or f1_i < worst_ok[1]):
            worst_ok = (key_out, f1_i)

    p, r, f1 = _prf1(tp, fp, fn)
    p_ok, r_ok, f1_ok = _prf1(tp_ok, fp_ok, fn_ok)

    print("=" * 120)
    print("📌 PA 출력 vs 정답(gold) 비교 리포트")
    print("=" * 120)
    print(f"PA 출력: {pa_output}")
    print(f"정답: {gold_sentences}")
    if book_name:
        print(f"book_name 필터: {book_name}")
    if pids:
        print(f"문단식별자 필터: {pids}")
    print()
    print(f"비교 문단 수: {total_paras}")
    # 기존 출력(호환성 유지): 문단 단위(리스트 완전일치)
    print(f"[OK] 번역문 문장리스트 완전일치: {len(tgt_exact_ok)}/{total_paras}")
    # 신규 출력: 문장 단위
    if tgt_sent_total_gold > 0:
        print(f"[OK] 번역문 문장 완전일치(문장): {tgt_sent_exact_ok}/{tgt_sent_total_gold}")
    print(f"📏 원문 경계 Precision/Recall/F1 (micro, 전체): {p:.4f} / {r:.4f} / {f1:.4f}")
    if len(tgt_exact_ok) > 0:
        print(
            f"📏 원문 경계 Precision/Recall/F1 (micro, tgt 완전일치 subset): {p_ok:.4f} / {r_ok:.4f} / {f1_ok:.4f}"
        )

    # src 유사도 통계(문장 단위 tgt_exact 기준)
    ok_stats = _stats(src_sim_tgt_sent_ok)
    fail_stats = _stats(src_sim_tgt_sent_fail)
    if ok_stats["n"] > 0:
        print(
            "📎 원문 유사도(SequenceMatcher, tgt문장일치 subset): "
            f"mean={ok_stats['mean']:.4f} med={ok_stats['median']:.4f} min={ok_stats['min']:.4f} "
            f"p10={ok_stats['p10']:.4f} p90={ok_stats['p90']:.4f} (n={ok_stats['n']})"
        )
    if fail_stats["n"] > 0:
        print(
            "📎 원문 유사도(SequenceMatcher, tgt문장불일치): "
            f"mean={fail_stats['mean']:.4f} med={fail_stats['median']:.4f} min={fail_stats['min']:.4f} "
            f"p10={fail_stats['p10']:.4f} p90={fail_stats['p90']:.4f} (n={fail_stats['n']})"
        )
    if worst is not None:
        if isinstance(worst[0], tuple):
            _bk, _pid = worst[0]
            print(f"⚠️  최저 F1 문단: book_name={_bk} pid={_pid} f1={worst[1]:.4f}")
        else:
            print(f"⚠️  최저 F1 문단: pid={worst[0]} f1={worst[1]:.4f}")
    if worst_ok is not None:
        if isinstance(worst_ok[0], tuple):
            _bk, _pid = worst_ok[0]
            print(f"⚠️  (tgt 완전일치 subset) 최저 F1 문단: book_name={_bk} pid={_pid} f1={worst_ok[1]:.4f}")
        else:
            print(f"⚠️  (tgt 완전일치 subset) 최저 F1 문단: pid={worst_ok[0]} f1={worst_ok[1]:.4f}")
    if tgt_exact_fail:
        if pred_has_book:
            # 보기 좋게 "book:pid"로 출력 (pred_has_book이면 모두 (book,pid)여야 함)
            pretty: list[str] = []
            for item in tgt_exact_fail:
                if isinstance(item, tuple) and len(item) == 2:
                    bk, pid = item
                    pretty.append(f"{bk}:{pid}")
                else:
                    pretty.append(str(item))
            print(f"[FAIL] 번역문 불일치 문단: {pretty}")
        else:
            print(f"[FAIL] 번역문 불일치 문단: {tgt_exact_fail}")

    if missing_in_gold:
        print(f"⚠️  pred에만 존재(정답 누락): {len(missing_in_gold)}")
    if missing_in_pred:
        print(f"⚠️  gold에만 존재(PA 누락): {len(missing_in_pred)}")
    if not pred_has_book:
        dup_gold = gold_df.duplicated(["문단식별자", "book_name"]).sum()
        if dup_gold == 0:
            # gold는 (book,pid) 유일하지만 pid는 중복될 수 있음
            pid_dups = gold_df.duplicated(["문단식별자"], keep=False).sum()
            if pid_dups > 0:
                print("⚠️  경고: PA 출력에 book_name이 없어, book 간 pid 중복이 있으면 평가가 왜곡될 수 있습니다.")

    # 엄격 모드: 번역문 불일치가 있으면 실패 처리
    ok = (total_paras > 0 and len(tgt_exact_fail) == 0)
    print("\n" + ("[OK] 전체 통과" if ok else "[FAIL] 실패 항목 존재"))
    return 0 if ok else 2


def run_pa_output_integrity_report(pa_output: Path, source_paragraphs: Path) -> int:
    """PA 결과 CSV/XLSX를 원본 문단 CSV/XLSX와 대조 검증."""

    # 가능하면 실제 PA 분할기를 사용해 'PA가 만든 문장 경계'와 1:1로 비교한다.
    # (과거에는 pa.sentence_splitter가 torch import 때문에 Windows에서 실패했으나, 현재는 torch optional.)
    try:
        from pa.sentence_splitter import split_target_sentences_advanced as _pa_split_target
    except Exception:
        _pa_split_target = None

    def _merge_quotation_markers_in_list(sentences: list[str]) -> list[str]:
        if len(sentences) <= 1:
            return sentences

        quotation_particles = r"(고|[이]?라?고|하고|며|면서)"

        # PA splitter와 동일하게 축약형(말했-/답했-)을 포함
        speech_verbs = r"(?:"
        speech_verbs += r"하"
        speech_verbs += r"|말(?:하|했)"
        speech_verbs += r"|말씀(?:하|했)"
        speech_verbs += r"|명(?:하|했)"
        speech_verbs += r"|이르(?:렀)?"
        speech_verbs += r"|대답(?:하|했)"
        speech_verbs += r"|답(?:하|했)"
        speech_verbs += r"|묻|문|물"
        speech_verbs += r"|여쭙|아뢰"
        speech_verbs += r"|전(?:하|했)"
        speech_verbs += r"|칭(?:하|했)"
        speech_verbs += r"|부르|외치"
        speech_verbs += r")"
        honorific_tense = r"(?:셨|ㅆ|시었|시어|시는|시ㄴ|시ㄹ|시|었|았|였|는|ㄴ|ㄹ|을)?"
        endings = r"(다|ㄴ다|는다|습니다|ㅂ니다|까|ㄹ까|을까|느냐|ㄴ가|는가|라|거라|소|오|어라|아라|니|으니)"
        closing_quote = r"[\"\'”’」』》〉】〕\)\]\}]?"
        punctuation = r"[\.。?!,，]?"

        marker_chunk = (
            closing_quote
            + r"\s*"
            + quotation_particles
            + r"\s*"
            + speech_verbs
            + honorific_tense
            + endings
            + r"\s*"
            + punctuation
            + r"\s*"
            + closing_quote
            + r"\s*"
        )
        quotation_marker_pattern = r"^\s*(?:" + marker_chunk + r")+$"

        changed = True
        while changed:
            changed = False
            merged: list[str] = []
            i = 0
            while i < len(sentences):
                current = sentences[i]
                accumulated: list[str] = []
                j = i + 1
                while j < len(sentences):
                    nxt = sentences[j]
                    if re.match(quotation_marker_pattern, nxt, re.IGNORECASE):
                        accumulated.append(nxt)
                        j += 1
                        changed = True
                    else:
                        break
                if accumulated:
                    merged.append((current + " " + " ".join(accumulated)).strip())
                    i = j
                else:
                    merged.append(current)
                    i += 1
            sentences = merged
        return sentences

    def _merge_speaker_utterance_pairs(segs: list[str]) -> list[str]:
        if len(segs) <= 1:
            return segs

        speaker_end = re.compile(
            r"(?:"
            r"(?:말(?:했|하였)다|말(?:하|했)다|말하였다|이르(?:렀)?다|이르되|대답하였다|답하였다|묻(?:었)?다|문(?:었)?다)"
            r"|(?:曰|云|言曰|問曰|答曰)"
            r")\s*[.。!?]?$"
        )
        opening_quote = re.compile(r"^[\s\"'“”‘’「『《〈【\(\[]")

        merged: list[str] = []
        i = 0
        while i < len(segs):
            cur = segs[i].strip()
            if not cur:
                i += 1
                continue
            if i < len(segs) - 1:
                nxt = segs[i + 1].strip()
                if nxt and len(cur) <= 60 and speaker_end.search(cur):
                    if opening_quote.match(nxt) or len(nxt) >= 20:
                        merged.append((cur.rstrip() + " " + nxt.lstrip()).strip())
                        i += 2
                        continue
            merged.append(cur)
            i += 1
        return merged

    def _split_long_by_comma_outside_brackets(s: str, limit: int) -> list[str]:
        if len(s) <= limit:
            return [s]
        level = 0
        split_pos = -1
        for i, ch in enumerate(s):
            if ch in "([":
                level += 1
            elif ch in ") ]" and level > 0:
                # ']'는 위 문자열에 공백이 들어갈 수 있으니 아래에서 따로 처리
                pass
            if ch == ")" and level > 0:
                level -= 1
            elif ch == "]" and level > 0:
                level -= 1

            if level == 0 and ch in [",", "，"]:
                split_pos = i
                break

        if split_pos < 0:
            return [s]
        left = s[:split_pos].strip()
        right = s[split_pos + 1 :].strip()
        out: list[str] = []
        if left:
            out.append(left)
        if right:
            out.append(right)
        return out if out else [s]

    def split_target_sentences_rule_based(text: str, max_length: int = 150) -> list[str]:
        if _pa_split_target is not None:
            # PA와 동일한 splitter 사용
            return [s.strip() for s in _pa_split_target(text, max_length=max_length) if str(s).strip()]

        # 폴백(호스트 환경 의존성 이슈용): 구두점 기반 분할 + 인용표지/화자발화 병합 + 콤마 1회 분할
        strong_end_pattern = r"(?<=[。！？.!?])\s+"
        sentences = re.split(strong_end_pattern, text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        sentences = _merge_quotation_markers_in_list(sentences)
        sentences = _merge_speaker_utterance_pairs(sentences)

        final: list[str] = []
        for s in sentences:
            final.extend(_split_long_by_comma_outside_brackets(s, limit=max_length))
        return [s.strip() for s in final if s.strip()]

    out_df = _read_tabular(pa_output)
    src_df = _read_tabular(source_paragraphs)

    # pandas는 CSV에서 빈 셀을 NaN으로 읽는다. 검증은 빈 문자열로 통일한다.
    for col in ("원문", "번역문"):
        if col in out_df.columns:
            out_df[col] = out_df[col].fillna("")
        if col in src_df.columns:
            src_df[col] = src_df[col].fillna("")

    required_out = {"문단식별자", "원문", "번역문"}
    required_src = {"문단식별자", "원문", "번역문"}
    if not required_out.issubset(set(out_df.columns)):
        missing = sorted(required_out - set(out_df.columns))
        raise SystemExit(f"PA 출력 파일에 필수 컬럼이 없습니다: {missing}")
    if not required_src.issubset(set(src_df.columns)):
        missing = sorted(required_src - set(src_df.columns))
        raise SystemExit(f"원본 문단 파일에 필수 컬럼이 없습니다: {missing}")

    # 빈 행 체크
    empty_src_rows = int((out_df["원문"].astype(str).str.strip() == "").sum())
    empty_tgt_rows = int((out_df["번역문"].astype(str).str.strip() == "").sum())

    # 문단별 검증
    out_groups = out_df.groupby("문단식별자", sort=False)
    src_map = {row["문단식별자"]: row for _, row in src_df.iterrows()}

    # 전역(전체 결합) 무결성: 분할/문장번호와 무관하게 전체 텍스트가 보존됐는지 확인
    out_src_global = _norm("".join(out_df["원문"].astype(str).tolist()))
    out_tgt_global = _norm("".join(out_df["번역문"].astype(str).tolist()))
    src_src_global = _norm("".join(src_df["원문"].astype(str).tolist()))
    src_tgt_global = _norm("".join(src_df["번역문"].astype(str).tolist()))

    global_src_ok = (out_src_global == src_src_global)
    global_tgt_ok = (out_tgt_global == src_tgt_global)

    ok_concat_src: list[int] = []
    ok_concat_tgt: list[int] = []
    ok_order_src: list[int] = []
    ok_order_tgt: list[int] = []
    ok_tgt_split_exact: list[int] = []

    fail_missing_para: list[int] = []
    fail_concat_src: list[int] = []
    fail_concat_tgt: list[int] = []
    fail_order_src: list[int] = []
    fail_order_tgt: list[int] = []
    fail_tgt_split_exact: list[int] = []

    for para_id, g in out_groups:
        if para_id not in src_map:
            fail_missing_para.append(int(para_id))
            continue

        src_para = str(src_map[para_id]["원문"])
        tgt_para = str(src_map[para_id]["번역문"])

        src_parts = [str(x).strip() for x in g["원문"].fillna("").tolist()]
        tgt_parts = [str(x).strip() for x in g["번역문"].fillna("").tolist()]

        src_join = _norm("".join(src_parts))
        tgt_join = _norm("".join(tgt_parts))
        src_full = _norm(src_para)
        tgt_full = _norm(tgt_para)

        if src_join == src_full:
            ok_concat_src.append(int(para_id))
        else:
            fail_concat_src.append(int(para_id))

        if tgt_join == tgt_full:
            ok_concat_tgt.append(int(para_id))
        else:
            fail_concat_tgt.append(int(para_id))

        if _check_in_order(src_full, [_norm(p) for p in src_parts]):
            ok_order_src.append(int(para_id))
        else:
            fail_order_src.append(int(para_id))

        if _check_in_order(tgt_full, [_norm(p) for p in tgt_parts]):
            ok_order_tgt.append(int(para_id))
        else:
            fail_order_tgt.append(int(para_id))

        # 번역문 분할(문장 수/순서) 불변: rule-based splitter 결과와 출력이 동일해야 함
        expected_tgt = split_target_sentences_rule_based(tgt_para, max_length=150)

        # 비교는 정규화(공백/개행/탭 제거) 기준으로 수행
        expected_tgt_norm = [_norm(s) for s in expected_tgt]
        tgt_parts_norm = [_norm(s) for s in tgt_parts]

        if expected_tgt_norm == tgt_parts_norm:
            ok_tgt_split_exact.append(int(para_id))
        else:
            fail_tgt_split_exact.append(int(para_id))

    total_paras = len(out_groups)

    print("=" * 120)
    print("📌 PA 결과 무결성 리포트")
    print("=" * 120)
    print(f"PA 출력: {pa_output}")
    print(f"원본 문단: {source_paragraphs}")
    print(f"총 문장쌍(행): {len(out_df)}")
    print(f"총 문단 수: {total_paras}")
    print()
    print(f"빈 원문 행: {empty_src_rows}")
    print(f"빈 번역문 행: {empty_tgt_rows}")
    print()
    print(f"[OK] 전역 결합 무결성(원문, 공백정규화): {'PASS' if global_src_ok else 'FAIL'}")
    print(f"[OK] 전역 결합 무결성(번역문, 공백정규화): {'PASS' if global_tgt_ok else 'FAIL'}")
    if not global_src_ok:
        mi = _first_mismatch(out_src_global, src_src_global)
        print(f"   ↳ 원문 첫 mismatch idx={mi}")
        print(f"   ↳ out: { _snippet(out_src_global, mi) }")
        print(f"   ↳ src: { _snippet(src_src_global, mi) }")
    if not global_tgt_ok:
        mi = _first_mismatch(out_tgt_global, src_tgt_global)
        print(f"   ↳ 번역문 첫 mismatch idx={mi}")
        print(f"   ↳ out: { _snippet(out_tgt_global, mi) }")
        print(f"   ↳ src: { _snippet(src_tgt_global, mi) }")
    print()
    print(f"[OK] 문단별 결합 무결성(원문) 통과: {len(ok_concat_src)}/{total_paras}")
    print(f"[OK] 문단별 결합 무결성(번역문) 통과: {len(ok_concat_tgt)}/{total_paras}")
    print(f"[OK] 문단별 순서 보존(원문) 통과: {len(ok_order_src)}/{total_paras}")
    print(f"[OK] 문단별 순서 보존(번역문) 통과: {len(ok_order_tgt)}/{total_paras}")
    print(f"[OK] 번역문 분할(문장 수/순서) 완전일치: {len(ok_tgt_split_exact)}/{total_paras}")

    def _print_list(title: str, xs: list[int]):
        if xs:
            print(f"\n[FAIL] {title}: {xs}")

    _print_list("원본 문단 매핑 실패(출력 문단식별자 미존재)", fail_missing_para)
    _print_list("결합 무결성 실패(원문)", fail_concat_src)
    _print_list("결합 무결성 실패(번역문)", fail_concat_tgt)
    _print_list("순서 보존 실패(원문)", fail_order_src)
    _print_list("순서 보존 실패(번역문)", fail_order_tgt)
    _print_list("번역문 분할 불일치", fail_tgt_split_exact)

    ok = (
        empty_src_rows == 0
        and empty_tgt_rows == 0
        and global_src_ok
        and global_tgt_ok
        and len(fail_missing_para) == 0
        and len(fail_concat_src) == 0
        and len(fail_concat_tgt) == 0
        and len(fail_order_src) == 0
        and len(fail_order_tgt) == 0
        and len(fail_tgt_split_exact) == 0
    )

    print("\n" + ("[OK] 전체 통과" if ok else "[FAIL] 실패 항목 존재"))
    return 0 if ok else 2


def run_full_43books_report() -> int:
    """기존 43권 무결성(길이) 리포트."""
    results = []

    for book in books:
        # 입력: 문단병렬
        para_file = Path(f"xlsx/{book}/{book}_문단병렬.xlsx")
        # PA 출력
        pa_file = Path(f"xlsx_pipeline_results/{book}/{book}_PA_문장병렬.xlsx")
        # GT: 문장병렬
        gt_file = Path(f"xlsx/{book}/{book}_문장병렬.xlsx")
        # SA 출력
        sa_file = Path(f"xlsx_pipeline_results/{book}/{book}_SA.xlsx")

        record = {"책": book}

        # 입력 (문단병렬)
        if para_file.exists():
            para = pd.read_excel(para_file)
            record["입력_행"] = len(para)
            record["입력_원문_길이"] = para["원문"].astype(str).map(len).sum()
            record["입력_번역_길이"] = para["번역문"].astype(str).map(len).sum()
        else:
            record["입력_행"] = None
            record["입력_원문_길이"] = None
            record["입력_번역_길이"] = None

        # PA 출력
        if pa_file.exists():
            pa = pd.read_excel(pa_file)
            record["PA_행"] = len(pa)
            record["PA_원문_길이"] = pa["원문"].astype(str).map(len).sum()
            record["PA_번역_길이"] = pa["번역문"].astype(str).map(len).sum()
        else:
            record["PA_행"] = None
            record["PA_원문_길이"] = None
            record["PA_번역_길이"] = None

        # GT (문장병렬)
        if gt_file.exists():
            gt = pd.read_excel(gt_file)
            record["GT_행"] = len(gt)
            record["GT_원문_길이"] = gt["원문"].astype(str).map(len).sum()
            record["GT_번역_길이"] = gt["번역문"].astype(str).map(len).sum()
        else:
            record["GT_행"] = None
            record["GT_원문_길이"] = None
            record["GT_번역_길이"] = None

        # SA 출력
        if sa_file.exists():
            try:
                sa = pd.read_excel(sa_file)
                record["SA_행"] = len(sa)
                record["SA_원문_길이"] = sa["원문"].astype(str).map(len).sum()
                record["SA_번역_길이"] = sa["번역문"].astype(str).map(len).sum()
            except Exception:
                record["SA_행"] = None
                record["SA_원문_길이"] = None
                record["SA_번역_길이"] = None
        else:
            record["SA_행"] = None
            record["SA_원문_길이"] = None
            record["SA_번역_길이"] = None

        # 무결성 체크: PA vs 입력
        if record["입력_원문_길이"] and record["PA_원문_길이"]:
            record["PA_원문_Δ"] = record["PA_원문_길이"] - record["입력_원문_길이"]
            record["PA_번역_Δ"] = record["PA_번역_길이"] - record["입력_번역_길이"]
        else:
            record["PA_원문_Δ"] = None
            record["PA_번역_Δ"] = None

        # 무결성 체크: SA vs 입력
        if record["입력_원문_길이"] and record["SA_원문_길이"]:
            record["SA_원문_Δ"] = record["SA_원문_길이"] - record["입력_원문_길이"]
            record["SA_번역_Δ"] = record["SA_번역_길이"] - record["입력_번역_길이"]
        else:
            record["SA_원문_Δ"] = None
            record["SA_번역_Δ"] = None

        results.append(record)

    df = pd.DataFrame(results)
    Path("analytics").mkdir(parents=True, exist_ok=True)
    df.to_csv("analytics/무결성_리포트.csv", index=False, encoding="utf-8-sig")

    print("=" * 120)
    print("📊 전체 43권 무결성 리포트")
    print("=" * 120)
    print(df.to_string(index=False))

    print("\n" + "=" * 120)
    print("📈 요약 통계")
    print("=" * 120)

    pa_issues = df[df["PA_원문_Δ"].notna() & (df["PA_원문_Δ"] != 0)]
    print(f"\n🔴 PA 원문 무결성 문제 (길이 변형):")
    print(f"   문제 책: {len(pa_issues)}권")
    if len(pa_issues) > 0:
        print(pa_issues[["책", "PA_원문_Δ", "PA_번역_Δ"]].to_string(index=False))

    sa_issues = df[df["SA_원문_Δ"].notna() & (df["SA_원문_Δ"] != 0)]
    print(f"\n🔴 SA 원문 무결성 문제 (길이 변형):")
    print(f"   문제 책: {len(sa_issues)}권")
    if len(sa_issues) > 0:
        print(sa_issues[["책", "SA_원문_Δ", "SA_번역_Δ"]].to_string(index=False))

    df["분할_차이_행"] = df["PA_행"] - df["GT_행"]
    split_issues = df[df["분할_차이_행"].notna() & (df["분할_차이_행"] != 0)]
    print(f"\n⚠️  분할 개수 차이 (PA 행 수 vs GT 행 수):")
    print(f"   차이 있는 책: {len(split_issues)}권")
    if len(split_issues) > 0:
        print(split_issues[["책", "PA_행", "GT_행", "분할_차이_행"]].to_string(index=False))

    print("\n[OK] 리포트 저장: analytics/무결성_리포트.csv")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="무결성 리포트")
    parser.add_argument("--input", type=str, help="PA 결과 파일 경로(.csv/.xlsx)")
    parser.add_argument("--source", type=str, help="원본 문단 파일 경로(.csv/.xlsx). (무결성 검증 모드) 기본: datasets/pd/test_10.csv")

    parser.add_argument("--gold", type=str, help="정답(문장 단위) 파일 경로(.csv/.xlsx). columns: 문단식별자,문장식별자,원문,번역문,book_name")
    parser.add_argument("--pids", nargs="*", help="필터: 문단식별자 목록(공백 구분). 예: --pids 10 12")
    parser.add_argument("--book-name", type=str, help="필터: book_name 정확히 일치")
    parser.add_argument("--keys-from", type=str, help="필터: (문단식별자, book_name) 목록이 들어있는 CSV/XLSX 경로")

    parser.add_argument("--extract", action="store_true", help="정답(gold) 파일에서 부분 추출")
    parser.add_argument("--out", type=str, help="(추출 모드) 저장 경로(.csv)")
    parser.add_argument("--out-paragraph", type=str, help="(추출 모드) 문단 단위로 재구성한 파일 저장 경로(.csv)")
    parser.add_argument("--sa-gold", type=str, help="(추출 모드) SA 정답(구병렬) 파일 경로(.csv/.xlsx). 기본: datasets/sa/test_100.csv")
    parser.add_argument("--out-sa", type=str, help="(추출 모드) SA 정답(구병렬) subset 저장 경로(.csv). 기본: <out>_sa_gold.csv")
    args = parser.parse_args()

    pids = _parse_pids(args.pids)

    if args.extract:
        if not args.gold:
            raise SystemExit("--extract 모드에는 --gold가 필요합니다.")
        if not args.out:
            raise SystemExit("--extract 모드에는 --out이 필요합니다.")
        gold_path = Path(args.gold)
        if not gold_path.exists():
            raise SystemExit(f"--gold 파일이 존재하지 않습니다: {gold_path}")

        keys_from = Path(args.keys_from) if args.keys_from else None
        if keys_from is not None and not keys_from.exists():
            raise SystemExit(f"--keys-from 파일이 존재하지 않습니다: {keys_from}")

        # SA gold 기본값 자동 선택
        if args.sa_gold:
            sa_gold_path = Path(args.sa_gold)
        else:
            default_sa = Path("datasets/sa/test_100.csv")
            sa_gold_path = default_sa if default_sa.exists() else None
        if sa_gold_path is not None and not sa_gold_path.exists():
            raise SystemExit(f"--sa-gold 파일이 존재하지 않습니다: {sa_gold_path}")

        return extract_gold_subset(
            gold_path=gold_path,
            out_path=Path(args.out),
            pids=pids,
            book_name=args.book_name,
            out_paragraph_path=(Path(args.out_paragraph) if args.out_paragraph else None),
            keys_from=keys_from,
            sa_gold_path=sa_gold_path,
            out_sa_path=(Path(args.out_sa) if args.out_sa else None),
        )

    if args.input:
        pa_output = Path(args.input)
        if not pa_output.exists():
            raise SystemExit(f"--input 파일이 존재하지 않습니다: {pa_output}")

        if args.gold:
            gold_path = Path(args.gold)
            if not gold_path.exists():
                raise SystemExit(f"--gold 파일이 존재하지 않습니다: {gold_path}")
            keys_from = Path(args.keys_from) if args.keys_from else None
            if keys_from is not None and not keys_from.exists():
                raise SystemExit(f"--keys-from 파일이 존재하지 않습니다: {keys_from}")
            return run_pa_output_vs_gold_report(
                pa_output=pa_output,
                gold_sentences=gold_path,
                pids=pids,
                book_name=args.book_name,
                keys_from=keys_from,
            )

        if args.source:
            source = Path(args.source)
        else:
            # 기본값: 로컬 테스트에서 가장 많이 쓰는 파일
            source = Path("datasets/pd/test_10.csv")

        if not source.exists():
            raise SystemExit(
                "--source를 지정해야 합니다. 기본값 datasets/pd/test_10.csv가 존재하지 않습니다."
            )

        return run_pa_output_integrity_report(pa_output=pa_output, source_paragraphs=source)

    return run_full_43books_report()


if __name__ == "__main__":
    raise SystemExit(main())
