#!/usr/bin/env python3
"""
BGE-M3 토큰별 임베딩 사전계산 스크립트

paragraph_train.xlsx + sentence_train.xlsx를 읽어:
  1) 문단별로 문장 경계 라벨 생성
  2) BGE-M3로 원문/번역문 토큰별 임베딩 추출
  3) kiwipiepy로 원문 POS features 추출
  4) datasets/precomputed/semantic_boundary/ 에 .pt 파일로 저장
"""

import argparse
import os
import sys
import math
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import torch
import numpy as np

# ── 프로젝트 루트 설정 ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.semantic_boundary_model import POS_TAGS, POS_DIM


def normalize_text(text: str) -> str:
    """공백/개행/탭 제거"""
    if pd.isna(text):
        return ""
    return str(text).replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "").strip()


def build_sentence_boundary_labels(
    paragraph_src: str, sentence_srcs: List[str]
) -> List[int]:
    """
    문단 원문 + 소속 문장 원문들로부터 문자별 경계 라벨 생성.

    경계 정의: 각 문장의 마지막 문자 위치에 1 (마지막 문장 제외).
    즉, text[i]=1이면 "i번째 문자 다음에서 분할".

    Args:
        paragraph_src: 정규화된 문단 원문
        sentence_srcs: 정규화된 문장 원문 리스트 (순서대로)

    Returns:
        [len(paragraph_src)] 크기의 binary 라벨 리스트
    """
    labels = [0] * len(paragraph_src)
    cursor = 0

    for i, sent_src in enumerate(sentence_srcs):
        sent_norm = normalize_text(sent_src)
        if not sent_norm:
            continue

        # 문장이 문단 내에서 시작하는 위치 찾기
        pos = paragraph_src.find(sent_norm, cursor)
        if pos == -1:
            # exact match 실패 → 순차 매칭으로 폴백
            pos = cursor

        end_pos = pos + len(sent_norm)

        # 마지막 문장이 아닌 경우에만 경계 표시
        if i < len(sentence_srcs) - 1 and end_pos - 1 < len(paragraph_src):
            labels[end_pos - 1] = 1

        cursor = end_pos

    return labels


def extract_pos_features(text: str, kiwi_tok) -> torch.Tensor:
    """
    kiwipiepy로 원문의 POS features 추출.

    Returns:
        [len(text), POS_DIM] float tensor (binary)
    """
    features = torch.zeros(len(text), POS_DIM)

    try:
        particles = kiwi_tok.extract_particles(text)
        # particles: [(token_text, type_str, position), ...]
        for token_text, type_str, position in particles:
            # position부터 token_text 길이만큼 해당 POS 활성화
            # kiwi_tokenizer에서 POS tag는 내부적으로 사용하므로
            # 여기서는 조사/어미 위치를 모두 마킹
            for j in range(len(token_text)):
                char_idx = position + j
                if 0 <= char_idx < len(text):
                    # 조사면 JK* 계열, 어미면 E* 계열
                    if type_str == "조사":
                        # 모든 조사 태그에 1
                        for k in range(9):  # JKS~JC
                            features[char_idx, k] = 1.0
                    elif type_str == "어미":
                        for k in range(9, 14):  # EP~ETM
                            features[char_idx, k] = 1.0
    except Exception as e:
        print(f"  ⚠️ POS feature 추출 실패: {e}")

    return features


def extract_pos_features_detailed(text: str, kiwi_tok) -> torch.Tensor:
    """
    kiwipiepy의 pos() 결과로 정밀한 POS features 추출.

    Returns:
        [len(text), POS_DIM] float tensor (binary)
    """
    features = torch.zeros(len(text), POS_DIM)

    try:
        pos_results = kiwi_tok.pos(text)
        # pos_results: [(token, pos_tag), ...]
        cursor = 0
        for token_text, pos_tag in pos_results:
            # token이 text에서 시작하는 위치 찾기
            pos = text.find(token_text, cursor)
            if pos == -1:
                pos = cursor

            if pos_tag in POS_TAGS:
                tag_idx = POS_TAGS.index(pos_tag)
                for j in range(len(token_text)):
                    char_idx = pos + j
                    if 0 <= char_idx < len(text):
                        features[char_idx, tag_idx] = 1.0

            cursor = pos + len(token_text)
    except Exception as e:
        print(f"  ⚠️ POS feature 추출 실패: {e}")

    return features


def get_token_embeddings_bgem3(
    texts: List[str],
    model,
    tokenizer,
    device,
    max_length: int = 512,
    batch_size: int = 16,
) -> List[Tuple[torch.Tensor, List[Tuple[int, int]]]]:
    """
    BGE-M3로 텍스트 리스트의 토큰별 임베딩을 추출.

    Returns:
        [(embeddings [L, 1024], offsets [(start, end), ...]), ...]
        - embeddings: 특수 토큰(CLS, SEP) 제외한 토큰 임베딩
        - offsets: 각 토큰의 원문 (start, end) 문자 위치
    """
    results = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        encoded = tokenizer(
            batch_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        offset_mappings = encoded.pop("offset_mapping")  # [B, L, 2]
        inputs = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state.cpu()  # [B, L, 1024]

        for b in range(len(batch_texts)):
            offsets = offset_mappings[b].tolist()  # [[start, end], ...]
            input_ids = encoded["input_ids"][b].tolist()

            # 특수 토큰(CLS=0번, SEP, PAD) 제외
            valid_embs = []
            valid_offsets = []
            for t_idx, (start, end) in enumerate(offsets):
                if start == 0 and end == 0:
                    continue  # 특수 토큰 or 패딩
                valid_embs.append(hidden[b, t_idx])
                valid_offsets.append((start, end))

            if valid_embs:
                emb_tensor = torch.stack(valid_embs)  # [num_tokens, 1024]
            else:
                emb_tensor = torch.zeros(1, hidden.shape[-1])
                valid_offsets = [(0, 1)]

            results.append((emb_tensor, valid_offsets))

    return results


def token_emb_to_char_emb(
    token_emb: torch.Tensor,
    offsets: List[Tuple[int, int]],
    text_len: int,
) -> torch.Tensor:
    """
    토큰 임베딩을 문자 임베딩으로 변환.
    한 토큰이 여러 문자를 커버하면 해당 문자 모두에 같은 임베딩.
    어떤 토큰에도 속하지 않는 문자는 zero 벡터.

    Returns:
        [text_len, hidden_dim] tensor
    """
    hidden_dim = token_emb.shape[-1]
    char_emb = torch.zeros(text_len, hidden_dim)

    for t_idx, (start, end) in enumerate(offsets):
        if t_idx >= token_emb.shape[0]:
            break
        for c_idx in range(start, min(end, text_len)):
            char_emb[c_idx] = token_emb[t_idx]

    return char_emb


def process_chunk_embeddings(
    chunk_samples: List[Dict],
    xlm_model,
    tokenizer,
    device,
    max_length: int = 512,
    batch_size: int = 8,
) -> List[Dict]:
    """
    청크 단위로 BGE-M3 임베딩 추출 + 문자 변환 + fp16 저장.
    GPU 메모리를 아끼기 위해 작은 단위로 처리.
    """
    import gc

    src_texts = [s["src_text"] for s in chunk_samples]
    tgt_texts = [s["tgt_text"] for s in chunk_samples]

    # 원문 임베딩
    src_results = get_token_embeddings_bgem3(
        src_texts, xlm_model, tokenizer, device,
        max_length=max_length, batch_size=batch_size,
    )

    # 번역문 임베딩
    tgt_results = get_token_embeddings_bgem3(
        tgt_texts, xlm_model, tokenizer, device,
        max_length=max_length, batch_size=batch_size,
    )

    # 문자 임베딩으로 변환
    final = []
    for i, sample in enumerate(chunk_samples):
        src_token_emb, src_offsets = src_results[i]
        tgt_token_emb, tgt_offsets = tgt_results[i]

        src_char_emb = token_emb_to_char_emb(
            src_token_emb, src_offsets, len(sample["src_text"])
        )
        tgt_char_emb = token_emb_to_char_emb(
            tgt_token_emb, tgt_offsets, len(sample["tgt_text"])
        )

        final.append({
            "src_emb": src_char_emb.half(),
            "tgt_emb": tgt_char_emb.half(),
            "pos_feat": sample["pos_feat"].half(),
            "labels": torch.tensor(sample["labels"], dtype=torch.float16),
            "book": sample["book"],
            "para_id": sample["para_id"],
        })

    # 중간 결과 해제
    del src_results, tgt_results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final


def main():
    parser = argparse.ArgumentParser(description="BGE-M3 임베딩 사전계산")
    parser.add_argument(
        "--paragraph-xlsx",
        default="datasets/splits/paragraph_train.xlsx",
        help="문단 학습 데이터",
    )
    parser.add_argument(
        "--sentence-xlsx",
        default="datasets/splits/sentence_train.xlsx",
        help="문장 학습 데이터",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/precomputed/semantic_boundary",
        help="출력 디렉토리",
    )
    parser.add_argument("--max-len", type=int, default=512, help="최대 시퀀스 길이")
    parser.add_argument("--batch-size", type=int, default=8, help="BGE-M3 배치 크기")
    parser.add_argument(
        "--chunk-size", type=int, default=500, help="한 번에 처리할 문단 수"
    )
    args = parser.parse_args()

    import gc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 이전 중간 저장 확인 (이어서 진행)
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터 로드 ──
    print("📂 데이터 로드 중...")
    para_df = pd.read_excel(args.paragraph_xlsx, engine="openpyxl")
    sent_df = pd.read_excel(args.sentence_xlsx, engine="openpyxl")

    print(f"  문단: {len(para_df)}행, 문장: {len(sent_df)}행")

    # 문장에서 불필요한 Unnamed 컬럼 제거
    sent_df = sent_df[["책명", "문단식별자", "문장식별자", "원문", "번역문"]].copy()
    sent_df["문단식별자"] = sent_df["문단식별자"].astype(float)

    # 문단식별자 기준으로 문장 그룹화
    sent_groups = sent_df.groupby(["책명", "문단식별자"])

    # ── 1단계: POS 처리 + 라벨 생성 (메모리 부담 없음) ──
    print("🔄 Kiwipiepy 로드 중...")
    from common.tokenizers.kiwi_tokenizer import get_kiwi_tokenizer

    kiwi_tok = get_kiwi_tokenizer()
    kiwi_tok._initialize()
    print("  Kiwipiepy 로드 완료")

    samples = []
    skipped = 0
    processed = 0

    print(f"\n🚀 POS 처리 시작 ({len(para_df)} 문단)...")

    for idx, row in para_df.iterrows():
        book = row["책명"]
        para_id = row["문단식별자"]
        src_raw = str(row["원문"]) if not pd.isna(row["원문"]) else ""
        tgt_raw = str(row["번역문"]) if not pd.isna(row["번역문"]) else ""

        src_norm = normalize_text(src_raw)
        tgt_norm = normalize_text(tgt_raw)

        if not src_norm or not tgt_norm:
            skipped += 1
            continue

        if len(src_norm) > args.max_len or len(tgt_norm) > args.max_len:
            skipped += 1
            continue

        key = (book, float(para_id))
        if key not in sent_groups.groups:
            skipped += 1
            continue

        sent_rows = sent_groups.get_group(key).sort_values("문장식별자")
        sentence_srcs = [
            normalize_text(str(s)) for s in sent_rows["원문"].tolist()
        ]

        if len(sentence_srcs) < 2:
            skipped += 1
            continue

        labels = build_sentence_boundary_labels(src_norm, sentence_srcs)
        pos_feat = extract_pos_features_detailed(src_norm, kiwi_tok)

        samples.append({
            "book": book,
            "para_id": para_id,
            "src_text": src_norm,
            "tgt_text": tgt_norm,
            "labels": labels,
            "pos_feat": pos_feat,
        })

        processed += 1
        if processed % 1000 == 0:
            print(f"  POS 처리: {processed}/{len(para_df)} (스킵: {skipped})")

    print(f"\n✅ POS 처리 완료: {processed}개 문단, {skipped}개 스킵")

    # kiwipiepy 메모리 해제
    del kiwi_tok
    gc.collect()

    # ── 2단계: BGE-M3 임베딩 (청크 단위) ──
    print("🔄 BGE-M3 모델 로드 중...")
    from FlagEmbedding import BGEM3FlagModel

    bgem3 = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    tokenizer = bgem3.tokenizer
    xlm_model = bgem3.model.model

    if torch.cuda.is_available():
        xlm_model = xlm_model.cuda().half()
        print("  GPU로 이동 완료")

    device = next(xlm_model.parameters()).device
    print(f"  BGE-M3 로드 완료 (device: {device})")

    # 이미 처리된 청크 확인
    existing_chunks = sorted(chunk_dir.glob("chunk_*.pt"))
    start_chunk = len(existing_chunks)
    total_chunks = math.ceil(len(samples) / args.chunk_size)

    if start_chunk > 0:
        print(f"  ⏩ {start_chunk}/{total_chunks} 청크 이미 완료, 이어서 진행")

    print(f"\n🔄 임베딩 추출 ({len(samples)} 문단, {total_chunks} 청크, 청크당 {args.chunk_size}개)...")

    for chunk_idx in range(start_chunk, total_chunks):
        chunk_start = chunk_idx * args.chunk_size
        chunk_end = min(chunk_start + args.chunk_size, len(samples))
        chunk_samples = samples[chunk_start:chunk_end]

        chunk_result = process_chunk_embeddings(
            chunk_samples, xlm_model, tokenizer, device,
            max_length=args.max_len, batch_size=args.batch_size,
        )

        # 청크 저장
        chunk_path = chunk_dir / f"chunk_{chunk_idx:04d}.pt"
        torch.save(chunk_result, str(chunk_path))
        del chunk_result
        gc.collect()

        print(f"  ✅ 청크 {chunk_idx+1}/{total_chunks} 완료 ({chunk_end}/{len(samples)})")

    # ── 3단계: 청크 병합 ──
    print(f"\n💾 청크 병합 중...")

    # BGE-M3 메모리 해제
    del xlm_model, bgem3, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_samples = []
    all_chunks = sorted(chunk_dir.glob("chunk_*.pt"))
    for cp in all_chunks:
        chunk_data = torch.load(str(cp), map_location="cpu", weights_only=False)
        final_samples.extend(chunk_data)
        del chunk_data

    save_path = output_dir / "precomputed_all.pt"
    torch.save(final_samples, str(save_path))
    print(f"\n✅ 저장 완료: {save_path}")
    print(f"   총 {len(final_samples)}개 샘플")

    # 통계
    total_boundaries = sum(sum(s["labels"]).item() for s in final_samples)
    total_chars = sum(len(s["labels"]) for s in final_samples)
    print(f"   총 문자: {total_chars:,}, 총 경계: {int(total_boundaries):,}")
    print(f"   경계 비율: {total_boundaries/total_chars*100:.2f}%")

    file_size_mb = save_path.stat().st_size / 1024 / 1024
    print(f"   파일 크기: {file_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
