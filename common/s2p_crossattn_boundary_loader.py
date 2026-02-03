"""SA Cross-Attention 경계 모델 로더

학습된 sa_crossattn_boundary.pt를 로드하여 원문+번역문 기반 경계 추론
"""

from pathlib import Path
from typing import List, Dict
import torch
from torch import nn


class CrossAttnBoundaryModel(nn.Module):
    """Cross-Attention 기반 경계 태거 (v3 학습 스크립트와 동일 구조)"""
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, 
                 emb_dim: int = 128, hidden: int = 256, num_heads: int = 4):
        super().__init__()
        
        # v3 모델과 동일한 레이어 이름 사용
        self.src_char_emb = nn.Embedding(src_vocab_size, emb_dim, padding_idx=0)
        self.tgt_char_emb = nn.Embedding(tgt_vocab_size, emb_dim, padding_idx=0)
        
        self.src_char_encoder = nn.LSTM(
            emb_dim, hidden // 2, num_layers=2, 
            bidirectional=True, batch_first=True, dropout=0.2
        )
        
        self.tgt_char_encoder = nn.LSTM(
            emb_dim, hidden // 2, num_layers=2, 
            bidirectional=True, batch_first=True, dropout=0.2
        )
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=num_heads, batch_first=True, dropout=0.1
        )
        
        self.norm = nn.LayerNorm(hidden)
        
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_emb = self.src_char_emb(src)
        src_hidden, _ = self.src_char_encoder(src_emb)
        
        tgt_emb = self.tgt_char_emb(tgt)
        tgt_hidden, _ = self.tgt_char_encoder(tgt_emb)
        
        src_padding_mask = (src == 0)
        
        cross_out, _ = self.cross_attn(
            query=tgt_hidden,
            key=src_hidden,
            value=src_hidden,
            key_padding_mask=src_padding_mask
        )
        
        cross_out = self.norm(cross_out + tgt_hidden)
        combined = torch.cat([tgt_hidden, cross_out], dim=-1)
        logits = self.boundary_head(combined).squeeze(-1)
        
        return logits
    
    def forward_with_attention(self, src: torch.Tensor, tgt: torch.Tensor):
        """어텐션 가중치도 함께 반환하는 forward"""
        src_emb = self.src_char_emb(src)
        src_hidden, _ = self.src_char_encoder(src_emb)
        
        tgt_emb = self.tgt_char_emb(tgt)
        tgt_hidden, _ = self.tgt_char_encoder(tgt_emb)
        
        src_padding_mask = (src == 0)
        
        # 어텐션 가중치 반환
        cross_out, attn_weights = self.cross_attn(
            query=tgt_hidden,
            key=src_hidden,
            value=src_hidden,
            key_padding_mask=src_padding_mask
        )
        
        cross_out = self.norm(cross_out + tgt_hidden)
        combined = torch.cat([tgt_hidden, cross_out], dim=-1)
        logits = self.boundary_head(combined).squeeze(-1)
        
        return logits, attn_weights


class CrossAttnBoundaryTaggerLoader:
    """Cross-Attention 경계 태거 로더"""
    
    def __init__(self, model_path: Path = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        if model_path is None:
            # s2p_crossattn_boundary.pt 파일이 실제 모델이므로 경로 조정
            model_path = Path(__file__).parent.parent / "models" / "s2p_crossattn_boundary.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Cross-Attention 경계 모델 없음: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.src_vocab = checkpoint["src_vocab"]
        self.tgt_vocab = checkpoint["tgt_vocab"]
        self.src_max_len = checkpoint.get("src_max_len", 256)
        self.tgt_max_len = checkpoint.get("tgt_max_len", 512)
        self.hidden = checkpoint.get("hidden", 256)
        self.emb_dim = checkpoint.get("emb_dim", 128)
        self.test_scores = checkpoint.get("test_scores", {})
        
        self.model = CrossAttnBoundaryModel(
            src_vocab_size=len(self.src_vocab) + 1,
            tgt_vocab_size=len(self.tgt_vocab) + 1,
            emb_dim=self.emb_dim,
            hidden=self.hidden,
        ).to(self.device)
        
        # 체크포인트 키 호환성
        state_dict_key = "state_dict" if "state_dict" in checkpoint else "model_state_dict"
        loaded_state = checkpoint[state_dict_key]
        
        # 키 이름 변환 (checkpoint: src_emb/src_encoder → model: src_char_emb/src_char_encoder)
        remapped_state = {}
        for k, v in loaded_state.items():
            # src_emb.* → src_char_emb.*
            if k.startswith("src_emb."):
                remapped_state["src_char_" + k] = v
            # tgt_emb.* → tgt_char_emb.*
            elif k.startswith("tgt_emb."):
                remapped_state["tgt_char_" + k] = v
            # src_encoder.* → src_char_encoder.*
            elif k.startswith("src_encoder."):
                remapped_state["src_char_" + k] = v
            # tgt_encoder.* → tgt_char_encoder.*
            elif k.startswith("tgt_encoder."):
                remapped_state["tgt_char_" + k] = v
            else:
                remapped_state[k] = v
        
        self.model.load_state_dict(remapped_state, strict=False)
        self.model.eval()
        
        print(f"✅ Cross-Attention 경계 태거 로드 완료 (src_vocab={len(self.src_vocab)}, tgt_vocab={len(self.tgt_vocab)}, device={self.device})")
    
    def _extract_huento_positions(self, src_text: str) -> List[int]:
        """원문에서 한국어 현토(토씨) 위치 추출
        
        현토는 한문 사이에 삽입된 한국어 조사/어미로, 구 단위 경계의 강력한 신호입니다.
        
        Returns:
            현토가 끝나는 위치(경계 후보)의 인덱스 리스트
        """
        if not src_text:
            return []
            
        try:
            # 싱글톤 캐싱: Kiwipiepy 인스턴스 재사용
            if not hasattr(self, '_kiwi_instance'):
                from kiwipiepy import Kiwi
                self._kiwi_instance = Kiwi()
            kiwi = self._kiwi_instance
            analysis = kiwi.analyze(src_text, top_n=1)
            if not analysis or not analysis[0]:
                return []
                
            tokens = analysis[0][0]  # 첫 번째 분석 결과
            huento_positions = []
            
            # 현토 패턴: 한자 뒤에 오는 한국어 조사/어미 (EC, EF, JC, JX 등)
            huento_tags = {'EC', 'EF', 'EP', 'JC', 'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX'}
            
            for token in tokens:
                tag = getattr(token, 'tag', '') or ''
                if tag in huento_tags:
                    # 현토 끝나는 위치 = 경계 후보
                    end_pos = getattr(token, 'start', 0) + getattr(token, 'len', 0)
                    if 0 < end_pos < len(src_text):
                        huento_positions.append(end_pos)
                        
            return huento_positions
            
        except Exception:
            return []
    
    def _encode(self, text: str, vocab: Dict, max_len: int) -> torch.Tensor:
        ids = [vocab.get(ch, 0) for ch in text][:max_len]
        ids += [0] * (max_len - len(ids))
        return torch.tensor([ids], dtype=torch.long, device=self.device)
    
    def _encode_batch(self, texts: List[str], vocab: Dict, max_len: int) -> torch.Tensor:
        """🚀 배치 인코딩 - 여러 텍스트를 한 번에 인코딩"""
        batch_ids = []
        for text in texts:
            ids = [vocab.get(ch, 0) for ch in text][:max_len]
            ids += [0] * (max_len - len(ids))
            batch_ids.append(ids)
        return torch.tensor(batch_ids, dtype=torch.long, device=self.device)
    
    def segment_text_batch(self, src_texts: List[str], tgt_texts: List[str], 
                          n_segments_list: List[int] = None,
                          threshold: float = 0.55, huento_bonus: float = 0.15, **kwargs) -> List[List[str]]:
        """🚀 배치 경계 분할 - 여러 (원문, 번역문) 쌍을 한 번에 처리
        
        Args:
            src_texts: 원문 텍스트 리스트
            tgt_texts: 번역문 텍스트 리스트
            n_segments_list: 각 행의 목표 세그먼트 개수 리스트. None이면 threshold 기준
            threshold: 경계 확률 임계값
            huento_bonus: 현토 보너스
            
        Returns:
            각 행의 분할된 번역문 세그먼트 리스트의 리스트
        """
        if not src_texts or not tgt_texts:
            return []
        
        if len(src_texts) != len(tgt_texts):
            raise ValueError("src_texts와 tgt_texts 길이가 다릅니다")
        
        batch_size = len(src_texts)
        
        # 배치 인코딩
        with torch.no_grad():
            src_batch = self._encode_batch(src_texts, self.src_vocab, self.src_max_len)
            tgt_batch = self._encode_batch(tgt_texts, self.tgt_vocab, self.tgt_max_len)
            
            # 배치 순방향 추론
            logits, attn_weights = self.model.forward_with_attention(src_batch, tgt_batch)
            probs_batch = torch.sigmoid(logits).cpu().numpy()  # [batch, tgt_max_len]
            attn_batch = attn_weights.cpu().numpy()  # [batch, tgt_max_len, src_max_len]
        
        results = []
        for i in range(batch_size):
            src_text = src_texts[i]
            tgt_text = tgt_texts[i]
            n_segments = n_segments_list[i] if n_segments_list else None
            
            if not tgt_text.strip():
                results.append([tgt_text] if tgt_text else [])
                continue
            
            probs = probs_batch[i][:len(tgt_text)]
            attn = attn_batch[i][:len(tgt_text), :len(src_text)]
            
            # 현토 보너스 적용
            huento_positions = self._extract_huento_positions(src_text)
            if huento_positions and len(probs) > 0:
                for src_pos in huento_positions:
                    if src_pos >= attn.shape[1]:
                        continue
                    attn_col = attn[:, src_pos]
                    top_tgt_indices = attn_col.argsort()[-3:][::-1]
                    for rank, tgt_idx in enumerate(top_tgt_indices):
                        if 0 < tgt_idx < len(probs):
                            weight = huento_bonus * (1.0 - rank * 0.3) * min(1.0, attn_col[tgt_idx] * 5)
                            probs[tgt_idx] = min(1.0, probs[tgt_idx] + weight)
            
            # 세그먼트 추출
            if n_segments is not None and n_segments > 1:
                prob_positions = [(probs[j], j) for j in range(1, len(probs))]
                prob_positions.sort(reverse=True)
                top_positions = sorted([pos for _, pos in prob_positions[:n_segments - 1]])
                
                segments = []
                start = 0
                for pos in top_positions:
                    segments.append(tgt_text[start:pos])
                    start = pos
                segments.append(tgt_text[start:])
                segments = [s for s in segments if s.strip()]
                results.append(segments if segments else [tgt_text])
            else:
                # threshold 기준 분할
                segments = []
                start = 0
                for j, prob in enumerate(probs):
                    if prob >= threshold and j > start:
                        segments.append(tgt_text[start:j])
                        start = j
                if start < len(tgt_text):
                    segments.append(tgt_text[start:])
                segments = [s for s in segments if s.strip()]
                results.append(segments if segments else [tgt_text])
        
        return results
    
    def segment_text(self, src_text: str, tgt_text: str, n_segments: int = None, 
                      threshold: float = 0.55, huento_bonus: float = 0.15, **kwargs) -> List[str]:
        """원문과 번역문을 기반으로 번역문을 경계에서 분할
        
        Args:
            src_text: 원문 텍스트
            tgt_text: 번역문 텍스트
            n_segments: 목표 세그먼트 개수 (원문 구 개수). None이면 threshold 기준
            threshold: 경계 확률 임계값 (n_segments가 None일 때만 사용)
            huento_bonus: 원문 현토 위치에 부여할 경계 확률 보너스 (기본: 0.15)
            
        Returns:
            분할된 번역문 세그먼트 리스트
        """
        if not tgt_text.strip():
            return [tgt_text] if tgt_text else []
        
        with torch.no_grad():
            src = self._encode(src_text, self.src_vocab, self.src_max_len)
            tgt = self._encode(tgt_text, self.tgt_vocab, self.tgt_max_len)
            
            # 어텐션 가중치도 함께 반환
            logits, attn_weights = self.model.forward_with_attention(src, tgt)
            logits = logits[0][:len(tgt_text)]
            probs = torch.sigmoid(logits).cpu().numpy()
            
            # attn_weights: [1, tgt_len, src_len] -> 각 번역문 위치가 원문의 어디를 보는지
            attn = attn_weights[0][:len(tgt_text), :len(src_text)].cpu().numpy()
        
        # 🆕 어텐션 기반 의미 매핑으로 현토 보너스 적용
        huento_positions = self._extract_huento_positions(src_text)
        if huento_positions and len(probs) > 0 and attn is not None:
            for src_pos in huento_positions:
                if src_pos >= attn.shape[1]:
                    continue
                # 원문의 현토 위치에 높은 어텐션을 주는 번역문 위치 찾기
                # attn[:, src_pos] = 각 번역문 위치가 원문의 src_pos를 얼마나 보는지
                attn_col = attn[:, src_pos]
                # 상위 3개의 번역문 위치에 보너스 부여
                top_tgt_indices = attn_col.argsort()[-3:][::-1]
                for rank, tgt_idx in enumerate(top_tgt_indices):
                    if 0 < tgt_idx < len(probs):
                        # 어텐션 강도에 비례한 보너스 (순위별 감쇠)
                        weight = huento_bonus * (1.0 - rank * 0.3) * min(1.0, attn_col[tgt_idx] * 5)
                        probs[tgt_idx] = min(1.0, probs[tgt_idx] + weight)
        
        # n_segments가 주어지면 상위 n-1개 경계 선택
        if n_segments is not None and n_segments > 1:
            # 첫 문자(i=0)는 경계가 아님, i>0인 위치에서 상위 n-1개 선택
            prob_positions = [(probs[i], i) for i in range(1, len(probs))]
            prob_positions.sort(reverse=True)  # 확률 높은 순
            
            # 상위 n_segments - 1개 경계 선택
            top_positions = sorted([pos for _, pos in prob_positions[:n_segments - 1]])
            
            # 분할
            segments = []
            start = 0
            for pos in top_positions:
                segments.append(tgt_text[start:pos])
                start = pos
            segments.append(tgt_text[start:])
            
            # 빈 세그먼트 제거
            segments = [s for s in segments if s.strip()]
            return segments if segments else [tgt_text]
        
        # threshold 기준 분할 (기존 방식)
        segments = []
        start = 0
        
        for i, prob in enumerate(probs):
            if prob >= threshold and i > start:
                segments.append(tgt_text[start:i])
                start = i
        
        # 마지막 세그먼트
        if start < len(tgt_text):
            segments.append(tgt_text[start:])
        
        # 빈 세그먼트 제거
        segments = [s for s in segments if s.strip()]
        
        return segments if segments else [tgt_text]
    
    def predict_boundary_probs(self, src_text: str, tgt_text: str) -> List[float]:
        """각 번역문 문자 위치의 경계 확률 반환"""
        if not tgt_text:
            return []
        
        with torch.no_grad():
            src = self._encode(src_text, self.src_vocab, self.src_max_len)
            tgt = self._encode(tgt_text, self.tgt_vocab, self.tgt_max_len)
            
            logits = self.model(src, tgt)[0][:len(tgt_text)]
            probs = torch.sigmoid(logits).cpu().numpy()
        
        return probs.tolist()


# 전역 인스턴스 캐싱
_crossattn_tagger_instance = None


def get_crossattn_boundary_tagger(model_path: Path = None, device: str = "cuda") -> CrossAttnBoundaryTaggerLoader:
    """Cross-Attention 경계 태거 싱글톤 반환"""
    global _crossattn_tagger_instance
    
    if _crossattn_tagger_instance is None:
        _crossattn_tagger_instance = CrossAttnBoundaryTaggerLoader(model_path=model_path, device=device)
    
    return _crossattn_tagger_instance
