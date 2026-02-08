"""
Semantic Cross-Lingual Boundary Model

BGE-M3 사전계산 임베딩 + kiwipiepy POS features를 사용하여
원문(한문)의 문장 경계를 번역문(한국어) 의미 대응으로 예측.

학습 시에는 사전계산 텐서만 사용 (BGE-M3 로드 불필요).
"""

import torch
from torch import nn
from typing import List, Tuple

# ── BGE-M3 hidden dim (XLM-RoBERTa-large 기반) ──
BGE_M3_DIM = 1024

# ── kiwipiepy POS feature 정의 ──
# 조사 9종 + 어미 5종 = 14 POS tags (kiwi_tokenizer.py 기준)
POS_TAGS = [
    "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC",  # 조사
    "EP", "EF", "EC", "ETN", "ETM",  # 어미
]
POS_DIM = len(POS_TAGS)  # 14


class SemanticCrossLingualBoundary(nn.Module):
    """
    Cross-lingual semantic boundary 모델.

    입력:
        src_emb: [B, L_src, 1024]  - BGE-M3 원문 토큰 임베딩
        tgt_emb: [B, L_tgt, 1024]  - BGE-M3 번역문 토큰 임베딩
        pos_feat: [B, L_src, 14]   - kiwipiepy POS binary features
        src_mask: [B, L_src]       - 원문 패딩 마스크 (True=pad)
        tgt_mask: [B, L_tgt]       - 번역문 패딩 마스크 (True=pad)

    출력:
        logits: [B, L_src]  - 각 원문 위치의 경계 logit
    """

    def __init__(
        self,
        src_dim: int = BGE_M3_DIM,
        tgt_dim: int = BGE_M3_DIM,
        pos_dim: int = POS_DIM,
        proj_dim: int = 256,
        n_attn_heads: int = 4,
        n_attn_layers: int = 2,
        dropout: float = 0.1,
        lstm_hidden: int = 128,
    ):
        super().__init__()
        self.proj_dim = proj_dim

        # projection layers
        self.src_proj = nn.Linear(src_dim + pos_dim, proj_dim)
        self.tgt_proj = nn.Linear(tgt_dim, proj_dim)

        # BiLSTM: 순차적 문맥 포착 (경계 패턴 학습)
        self.src_bilstm = nn.LSTM(
            proj_dim, lstm_hidden, num_layers=1,
            batch_first=True, bidirectional=True, dropout=0,
        )
        # BiLSTM output = 2*lstm_hidden → proj_dim으로 매핑
        self.lstm_proj = nn.Linear(lstm_hidden * 2, proj_dim)

        # cross-attention: src (query) attends to tgt (key/value)
        encoder_layer = nn.TransformerDecoderLayer(
            d_model=proj_dim,
            nhead=n_attn_heads,
            dim_feedforward=proj_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.cross_attn = nn.TransformerDecoder(
            encoder_layer, num_layers=n_attn_layers
        )

        # boundary classifier: [cross_attn_out || bilstm_out] → logit
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim // 2, 1),
        )

    def forward(
        self,
        src_emb: torch.Tensor,
        tgt_emb: torch.Tensor,
        pos_feat: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # src: [B, L_src, src_dim+pos_dim] → [B, L_src, proj_dim]
        src_input = torch.cat([src_emb, pos_feat], dim=-1)
        src_h = self.src_proj(src_input)

        # BiLSTM: 순차적 문맥 (앞뒤 문자 패턴 포착)
        if src_mask is not None:
            # pack_padded_sequence를 위한 길이 계산
            lengths = (~src_mask).sum(dim=1).cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                src_h, lengths, batch_first=True, enforce_sorted=False,
            )
            lstm_out, _ = self.src_bilstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=src_h.shape[1],
            )
        else:
            lstm_out, _ = self.src_bilstm(src_h)

        lstm_h = self.lstm_proj(lstm_out)  # [B, L_src, proj_dim]

        # tgt: [B, L_tgt, tgt_dim] → [B, L_tgt, proj_dim]
        tgt_h = self.tgt_proj(tgt_emb)

        # cross-attention: lstm_h (query) attends to tgt_h (key/value)
        cross_out = self.cross_attn(
            tgt=lstm_h,
            memory=tgt_h,
            tgt_key_padding_mask=src_mask,
            memory_key_padding_mask=tgt_mask,
        )

        # classifier: [cross_out || lstm_h] → logit
        combined = torch.cat([cross_out, lstm_h], dim=-1)  # [B, L_src, proj_dim*2]
        logits = self.classifier(combined).squeeze(-1)  # [B, L_src]

        return logits


class DiceBCELoss(nn.Module):
    """Dice + BCE 결합 손실 (클래스 불균형 대응)"""

    def __init__(self, dice_weight: float = 0.5, smooth: float = 1.0, pos_weight: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.register_buffer("_pos_weight", torch.tensor([pos_weight]))
        self.smooth = smooth

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, L] raw logits
            targets: [B, L] binary labels
            mask: [B, L] True=valid, False=pad (주의: src_mask와 반대)
        """
        # BCE (pos_weight를 logits device로 이동)
        pw = self._pos_weight.to(logits.device)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none"
        )
        if mask is not None:
            bce_loss = bce_loss * mask.float()
            bce_loss = bce_loss.sum() / mask.float().sum().clamp(min=1)
        else:
            bce_loss = bce_loss.mean()

        # Dice
        probs = torch.sigmoid(logits)
        if mask is not None:
            probs = probs * mask.float()
            targets_m = targets * mask.float()
        else:
            targets_m = targets

        intersection = (probs * targets_m).sum()
        union = probs.sum() + targets_m.sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)

        return (1 - self.dice_weight) * bce_loss + self.dice_weight * dice_loss


class FocalBoundaryLoss(nn.Module):
    """
    Focal Loss + 경계 위치 가중치.

    - Focal Loss (gamma=2): 쉬운 negative 샘플의 기여를 억제
    - 경계 가중치: 경계(label=1)에 pos_weight, 인접(±1)에 near_weight 적용
    """

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: float = 10.0,
        near_weight: float = 5.0,
        dice_weight: float = 0.3,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.near_weight = near_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, L] raw logits
            targets: [B, L] binary labels
            mask: [B, L] True=valid, False=pad
        """
        probs = torch.sigmoid(logits)
        # Focal modulation: (1-p_t)^gamma
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # BCE (element-wise)
        bce = -targets * torch.log(probs.clamp(min=1e-7)) - (1 - targets) * torch.log(
            (1 - probs).clamp(min=1e-7)
        )

        # 경계 위치 가중치: boundary=pos_weight, ±1=near_weight, 나머지=1
        pos_w = torch.ones_like(targets)
        pos_w = pos_w + (self.pos_weight - 1) * targets  # boundary 위치

        # 인접 위치 가중치 (경계 좌우 1칸)
        if self.near_weight > 1.0:
            shifted_left = torch.zeros_like(targets)
            shifted_right = torch.zeros_like(targets)
            shifted_left[:, 1:] = targets[:, :-1]
            shifted_right[:, :-1] = targets[:, 1:]
            near_mask = ((shifted_left + shifted_right) > 0).float() * (1 - targets)
            pos_w = pos_w + (self.near_weight - 1) * near_mask

        focal_loss = focal_weight * bce * pos_w

        if mask is not None:
            focal_loss = focal_loss * mask.float()
            focal_loss = focal_loss.sum() / mask.float().sum().clamp(min=1)
        else:
            focal_loss = focal_loss.mean()

        # Dice (보조)
        if self.dice_weight > 0:
            if mask is not None:
                probs_m = probs * mask.float()
                targets_m = targets * mask.float()
            else:
                probs_m = probs
                targets_m = targets
            intersection = (probs_m * targets_m).sum()
            union = probs_m.sum() + targets_m.sum()
            dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
            return (1 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss

        return focal_loss
