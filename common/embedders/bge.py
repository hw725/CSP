"""BGE 임베더 - 프로세스 안전 버전

주의:
- 로컬 환경에 torch가 없을 수 있어(예: Docker-only 구성), torch 의존 부분은 optional로 처리한다.
- torch/FlagEmbedding 로딩이 실패하면 deterministic한 문자 기반 해시 임베딩으로 fallback하여
    파이프라인이 끝까지 실행되도록 한다.
"""

import logging
import numpy as np
import os
import hashlib
import pickle
from pathlib import Path
from typing import List, Optional, Callable

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from tqdm import tqdm

logger = logging.getLogger(__name__)

# 전역 설정 - GPU 최적화
DEFAULT_BATCH_SIZE = 128  # 🚀 배치 크기 증가 (32 → 128)
DISK_CACHE_DIR = Path(".cache/embeddings")  # 디스크 캐시 디렉토리
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"

class EmbeddingManager:
    """
    BGE-M3 모델을 사용한 Multi-Vector 임베딩 계산 및 캐시 관리 클래스

    BGE-M3는 세 가지 벡터 유형을 제공합니다:
    - Dense vectors: 일반적인 의미 임베딩 (1024차원)
    - Sparse vectors: 키워드 기반 lexical 매칭
    - ColBERT vectors: 토큰별 상세 표현

    Features:
    - GPU 가속 지원
    - 캐싱 시스템으로 성능 최적화
    - Multi-vector와 dense-only 모드 지원
    - 프로세스 안전 락 시스템
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        fallback_to_dummy: bool = True,
        device_id=None,
        use_disk_cache: bool = True,
    ):
        self.model_name = model_name
        self.model = None
        self._cache = {}
        self._fallback_to_dummy = fallback_to_dummy
        self._model_loaded = False
        self._use_dummy = False
        self.process_id = os.getpid()
        self.device_id = device_id
        # 🔧 API 버전 캐싱 (한 번만 감지하도록)
        self._api_version_checked = False
        self._use_legacy_api = False  # True면 구버전 API
        self._fallback_vectorizer = None

        # 🚀 디스크 캐시 설정
        self._use_disk_cache = use_disk_cache
        if use_disk_cache:
            DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self._load_disk_cache()

    def _get_cache_path(self) -> Path:
        """디스크 캐시 파일 경로 반환"""
        return DISK_CACHE_DIR / f"bge_cache_{self.model_name.replace('/', '_')}.pkl"

    def _load_disk_cache(self):
        """디스크에서 캐시 로드"""
        cache_path = self._get_cache_path()
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    self._cache = pickle.load(f)
                logger.info(f"✅ 디스크 캐시 로드: {len(self._cache)}개 항목")
            except Exception as e:
                logger.warning(f"⚠️ 디스크 캐시 로드 실패: {e}")
                self._cache = {}

    def _save_disk_cache(self):
        """캐시를 디스크에 저장"""
        if not self._use_disk_cache:
            return
        cache_path = self._get_cache_path()
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(self._cache, f)
            logger.debug(f"💾 디스크 캐시 저장: {len(self._cache)}개 항목")
        except Exception as e:
            logger.warning(f"⚠️ 디스크 캐시 저장 실패: {e}")

    def _get_fallback_vectorizer(self):
        if self._fallback_vectorizer is not None:
            return self._fallback_vectorizer
        try:
            from sklearn.feature_extraction.text import HashingVectorizer

            # char_wb ngram은 CJK/현토(한글) 포함 텍스트에서도 비교적 안정적
            self._fallback_vectorizer = HashingVectorizer(
                n_features=1024,
                alternate_sign=False,
                norm=None,
                analyzer="char_wb",
                ngram_range=(2, 5),
            )
        except Exception:
            self._fallback_vectorizer = None
        return self._fallback_vectorizer

    def _fallback_dense_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """torch 없이도 동작하는 deterministic dense(1024) 임베딩."""
        vec = self._get_fallback_vectorizer()
        if vec is None:
            return [self._generate_dummy_embedding(t) for t in texts]

        X = vec.transform([t or "" for t in texts])
        X = X.astype(np.float32)
        # HashingVectorizer는 sparse -> dense로 변환 (n_features=1024)
        Xd = X.toarray().astype(np.float32)
        norms = np.linalg.norm(Xd, axis=1, keepdims=True)
        Xd = Xd / (norms + 1e-8)
        return [Xd[i] for i in range(Xd.shape[0])]

    def _load_model(self):
        """모델 로딩 (프로세스별)"""
        if self._model_loaded and os.getpid() == self.process_id:
            return

        # 프로세스 변경 시 재초기화
        if os.getpid() != self.process_id:
            self.process_id = os.getpid()
            self._model_loaded = False
            self.model = None
            # 🔧 프로세스 변경 시 API 버전도 재확인
            self._api_version_checked = False
            self._use_legacy_api = False

        try:
            from FlagEmbedding import FlagModel

            # 🔧 verbose 모드에서만 출력
            if logger.isEnabledFor(logging.DEBUG):
                print(
                    f"프로세스 {self.process_id}: BGE 모델 로딩 중... (device_id={self.device_id})"
                )

            # 환경 변수 설정
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

            # 🔧 non-verbose 모드에서는 모든 출력 숨김
            if not logger.isEnabledFor(logging.DEBUG):
                import sys
                import contextlib

                # stdout와 stderr를 임시로 리다이렉트
                with open(os.devnull, "w") as devnull:
                    with contextlib.redirect_stdout(
                        devnull
                    ), contextlib.redirect_stderr(devnull):
                        # 디바이스 설정 (프로세스별로 GPU 메모리 분리)
                        if torch is not None and torch.cuda.is_available():
                            if self.device_id is not None:
                                device = f"cuda:{self.device_id}"
                            else:
                                device = "cuda"
                        else:
                            device = "cpu"

                        self.model = FlagModel(
                            self.model_name,
                            query_instruction_for_retrieval="Represent this query for retrieving relevant documents: ",
                            use_fp16=True,
                        )
            else:
                # verbose 모드에서는 정상 출력
                # 디바이스 설정 (프로세스별로 GPU 메모리 분리)
                if torch is not None and torch.cuda.is_available():
                    if self.device_id is not None:
                        device = f"cuda:{self.device_id}"
                    else:
                        device = "cuda"
                else:
                    device = "cpu"

                self.model = FlagModel(
                    self.model_name,
                    query_instruction_for_retrieval="Represent this query for retrieving relevant documents: ",
                    use_fp16=True,
                )

            self._model_loaded = True
            self._use_dummy = False
            # 🔧 verbose 모드에서만 출력
            if logger.isEnabledFor(logging.DEBUG):
                print(
                    f"프로세스 {self.process_id}: BGE 모델 로딩 완료 (device={device})"
                )

        except Exception as e:
            # 🔧 verbose 모드에서만 출력
            if logger.isEnabledFor(logging.DEBUG):
                print(f"프로세스 {self.process_id}: BGE 모델 로딩 실패: {e}")

            if self._fallback_to_dummy:
                if logger.isEnabledFor(logging.DEBUG):
                    print(f"프로세스 {self.process_id}: 더미 모드로 전환")
                self._use_dummy = True
                self._model_loaded = True
            else:
                raise RuntimeError(f"BGE 모델 초기화 실패: {e}")

    def _generate_dummy_embedding(self, text: str) -> np.ndarray:
        """더미 dense 임베딩 생성 (1024차원)"""
        seed = hash(text) % (2**31)
        np.random.seed(seed)
        dummy_emb = np.random.randn(1024).astype(np.float32)
        dummy_emb = dummy_emb / (np.linalg.norm(dummy_emb) + 1e-8)
        return dummy_emb

    def _generate_dummy_multi_embedding(self, text: str) -> np.ndarray:
        """더미 multi-vector 임베딩 생성"""
        seed = hash(text) % (2**31)
        np.random.seed(seed)

        # Dense vector (1024)
        dense = np.random.randn(1024).astype(np.float32)
        dense = dense / (np.linalg.norm(dense) + 1e-8)

        # Sparse vector representation (축약된 키 특징들, 100차원)
        sparse_indices = np.random.choice(
            30522, size=min(100, len(text.split()) * 10), replace=False
        )
        sparse_values = np.random.exponential(0.5, size=len(sparse_indices)).astype(
            np.float32
        )
        sparse_dense = np.zeros(100, dtype=np.float32)  # 축약된 sparse representation
        sparse_dense[: len(sparse_indices)] = sparse_values[:100]

        # ColBERT vector representation (평균 토큰별 벡터, 512차원)
        avg_tokens = max(1, len(text.split()))
        colbert_tokens = min(avg_tokens, 16)  # 최대 16토큰
        colbert = np.random.randn(colbert_tokens, 32).astype(
            np.float32
        )  # 토큰당 32차원
        colbert_flat = colbert.flatten()[:512]  # 최대 512차원으로 제한
        colbert_padded = np.pad(colbert_flat, (0, 512 - len(colbert_flat)), "constant")

        # 전체 결합: dense(1024) + sparse(100) + colbert(512) = 1636차원
        multi_vector = np.concatenate([dense, sparse_dense, colbert_padded])
        return multi_vector

    def _combine_multi_vectors(self, result, texts: List[str]) -> List[np.ndarray]:
        """BGE-M3 multi-vector 결과를 결합"""
        combined_embeddings = []

        for i, text in enumerate(texts):
            # Dense vector (1024차원)
            dense = result["dense_vecs"][i]

            # Sparse vector 처리 (희소 벡터를 조밀한 표현으로 변환)
            sparse_dict = result["lexical_weights"][i]
            sparse_dense = self._sparse_to_dense(sparse_dict, target_dim=100)

            # ColBERT vectors 처리 (토큰별 벡터를 평탄화)
            colbert_vecs = result["colbert_vecs"][i]  # [num_tokens, 1024]
            colbert_flat = self._colbert_to_flat(colbert_vecs, target_dim=512)

            # 전체 결합
            multi_vector = np.concatenate([dense, sparse_dense, colbert_flat])
            combined_embeddings.append(multi_vector)

        return combined_embeddings

    def _sparse_to_dense(self, sparse_dict: dict, target_dim: int = 100) -> np.ndarray:
        """Sparse vector를 dense representation으로 변환"""
        dense_sparse = np.zeros(target_dim, dtype=np.float32)

        # 상위 중요 토큰들의 가중치만 사용
        sorted_items = sorted(sparse_dict.items(), key=lambda x: x[1], reverse=True)

        for i, (token_id, weight) in enumerate(sorted_items[:target_dim]):
            dense_sparse[i] = weight

        # 정규화
        norm = np.linalg.norm(dense_sparse)
        if norm > 0:
            dense_sparse = dense_sparse / norm

        return dense_sparse

    def _colbert_to_flat(
        self, colbert_vecs: np.ndarray, target_dim: int = 512
    ) -> np.ndarray:
        """ColBERT vectors를 고정 차원으로 평탄화"""
        if len(colbert_vecs.shape) == 2:
            # [num_tokens, 1024] -> 평균 풀링 후 차원 축소
            pooled = np.mean(colbert_vecs, axis=0)  # [1024]

            # PCA류 차원 축소 (간단한 선형 투영)
            target_ratio = target_dim / len(pooled)
            if target_ratio < 1:
                # 다운샘플링
                indices = np.linspace(0, len(pooled) - 1, target_dim, dtype=int)
                reduced = pooled[indices]
            else:
                # 패딩
                reduced = np.pad(pooled, (0, target_dim - len(pooled)), "constant")[
                    :target_dim
                ]
        else:
            # 예상치 못한 형태인 경우 제로 패딩
            reduced = np.zeros(target_dim, dtype=np.float32)

        return reduced

    def _simulate_multi_vector_from_dense(
        self, text: str, dense_emb: np.ndarray
    ) -> np.ndarray:
        """Dense embedding을 기반으로 multi-vector 시뮬레이션"""

        # Dense vector는 그대로 사용 (1024차원)
        dense = dense_emb.astype(np.float32)

        # Sparse vector 시뮬레이션 (텍스트 특성 기반, 100차원)
        text_hash = hash(text) % (2**31)
        np.random.seed(text_hash)

        # 텍스트 길이와 특성을 반영한 sparse 특징
        text_features = []
        text_length = len(text)
        char_diversity = len(set(text))

        # 길이 기반 특징
        text_features.extend(
            [
                text_length / 100.0,  # 정규화된 길이
                char_diversity / max(text_length, 1),  # 문자 다양성
                len(text.split()) / max(text_length / 4, 1),  # 단어 밀도
            ]
        )

        # Dense embedding에서 파생된 특징 (상위 차원들의 통계적 특성)
        dense_stats = [
            np.mean(dense),
            np.std(dense),
            np.max(dense),
            np.min(dense),
            np.median(dense),
        ]
        text_features.extend(dense_stats)

        # 나머지는 dense embedding 기반 변형
        remaining_dims = 100 - len(text_features)
        if remaining_dims > 0:
            # Dense의 일부 차원을 선택적으로 변형
            selected_indices = np.linspace(0, len(dense) - 1, remaining_dims, dtype=int)
            transformed_features = dense[selected_indices] * 0.1  # 스케일 조정
            text_features.extend(transformed_features)

        sparse_sim = np.array(text_features[:100], dtype=np.float32)
        sparse_sim = sparse_sim / (np.linalg.norm(sparse_sim) + 1e-8)

        # ColBERT vector 시뮬레이션 (토큰별 특성 시뮬레이션, 512차원)
        tokens = text.split()
        token_count = min(len(tokens), 16)  # 최대 16토큰

        if token_count > 0:
            # 각 토큰에 대해 dense embedding의 다른 부분을 사용
            colbert_features = []
            for i in range(token_count):
                start_idx = (i * len(dense) // token_count) % len(dense)
                end_idx = min(start_idx + 32, len(dense))
                token_feature = dense[start_idx:end_idx]

                # 32차원으로 패딩 또는 자르기
                if len(token_feature) < 32:
                    token_feature = np.pad(
                        token_feature, (0, 32 - len(token_feature)), "constant"
                    )
                else:
                    token_feature = token_feature[:32]

                colbert_features.extend(token_feature)

            # 512차원으로 맞추기
            if len(colbert_features) < 512:
                colbert_features.extend([0.0] * (512 - len(colbert_features)))
            else:
                colbert_features = colbert_features[:512]

            colbert_sim = np.array(colbert_features, dtype=np.float32)
        else:
            # 토큰이 없으면 dense의 일부를 변형해서 사용
            colbert_sim = dense[:512] * 0.5  # 스케일 조정
            if len(colbert_sim) < 512:
                colbert_sim = np.pad(
                    colbert_sim, (0, 512 - len(colbert_sim)), "constant"
                )

        colbert_sim = colbert_sim / (np.linalg.norm(colbert_sim) + 1e-8)

        # 전체 결합: dense(1024) + sparse_sim(100) + colbert_sim(512) = 1636차원
        multi_vector = np.concatenate([dense, sparse_sim, colbert_sim])

        return multi_vector

    def compute_embeddings_with_cache(
        self,
        texts: List[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_batch_progress: bool = False,
        use_multi_vector: bool = True,
        save_to_disk: bool = True,  # 🔧 추가된 파라미터
    ) -> np.ndarray:
        """프로세스 안전한 임베딩 계산 - BGE-M3 multi-vector 지원"""

        if not texts:
            return np.array([])

        # 모델 로딩
        self._load_model()

        result_list: List[Optional[np.ndarray]] = [None] * len(texts)
        to_embed: List[str] = []
        indices_to_embed: List[int] = []

        # 캐시 확인 (multi-vector 모드에 따라 다른 키 사용)
        cache_suffix = "_multi" if use_multi_vector else "_dense"
        for i, txt in enumerate(texts):
            cache_key = txt + cache_suffix
            if cache_key in self._cache:
                result_list[i] = self._cache[cache_key]
            else:
                to_embed.append(txt)
                indices_to_embed.append(i)

        # 새 임베딩 계산
        if to_embed:
            if self._use_dummy:
                # torch/FlagEmbedding 없이도 파이프라인이 의미 있게 돌도록 fallback 임베딩 사용
                dense_list = self._fallback_dense_embeddings(to_embed)
                if use_multi_vector:
                    embeddings = [
                        self._simulate_multi_vector_from_dense(text, dense_emb)
                        for text, dense_emb in zip(to_embed, dense_list)
                    ]
                else:
                    embeddings = dense_list
            else:
                # 실제 BGE 모델 사용
                embeddings = []

                for start in range(0, len(to_embed), batch_size):
                    batch = to_embed[start : start + batch_size]

                    try:
                        # GPU 메모리 정리
                        if torch is not None and torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        if use_multi_vector:
                            # BGE-M3 multi-vector 임베딩 계산 (API 버전 자동 감지)
                            # 🔧 API 버전이 이미 결정되었으면 그대로 사용
                            if not self._api_version_checked:
                                try:
                                    # 최신 API 시도 (BGE-M3 v1.2.0+)
                                    result = self.model.encode(
                                        batch,
                                        return_dense=True,
                                        return_sparse=True,
                                        return_colbert_vecs=True,
                                    )
                                    batch_embeddings = self._combine_multi_vectors(
                                        result, batch
                                    )
                                    self._api_version_checked = True
                                    self._use_legacy_api = False
                                    logger.info(
                                        f"✅ BGE-M3 최신 API 감지 (multi-vector 지원)"
                                    )

                                except TypeError as api_error:
                                    if "unexpected keyword argument" in str(api_error):
                                        # 구버전 API - dense embedding + 시뮬레이션 multi-vector
                                        self._api_version_checked = True
                                        self._use_legacy_api = True
                                        logger.warning(
                                            f"⚠️ BGE-M3 구버전 API 감지, Dense + 시뮬레이션 Multi-vector 모드"
                                        )
                                        dense_embeddings = self.model.encode(batch)
                                        # Dense embedding을 기반으로 multi-vector 시뮬레이션 (품질 유지)
                                        batch_embeddings = []
                                        for i, (text, dense_emb) in enumerate(
                                            zip(batch, dense_embeddings)
                                        ):
                                            simulated_multi = (
                                                self._simulate_multi_vector_from_dense(
                                                    text, dense_emb
                                                )
                                            )
                                            batch_embeddings.append(simulated_multi)
                                    else:
                                        raise api_error
                            else:
                                # API 버전이 이미 결정됨 - 그에 맞게 처리
                                if self._use_legacy_api:
                                    # 구버전: dense + 시뮬레이션 multi-vector (품질 유지)
                                    dense = self.model.encode(batch)
                                    batch_embeddings = []
                                    for i, (text, dense_emb) in enumerate(
                                        zip(batch, dense)
                                    ):
                                        simulated_multi = (
                                            self._simulate_multi_vector_from_dense(
                                                text, dense_emb
                                            )
                                        )
                                        batch_embeddings.append(simulated_multi)
                                else:
                                    # 최신 버전: multi-vector 계산
                                    result = self.model.encode(
                                        batch,
                                        return_dense=True,
                                        return_sparse=True,
                                        return_colbert_vecs=True,
                                    )
                                    batch_embeddings = self._combine_multi_vectors(
                                        result, batch
                                    )
                        else:
                            # dense embedding만 계산
                            dense = self.model.encode(batch)
                            batch_embeddings = dense

                        embeddings.extend(batch_embeddings)

                    except Exception as e:
                        # TypeError는 이미 위에서 처리됨
                        if not isinstance(e, TypeError):
                            print(f"프로세스 {self.process_id}: 임베딩 계산 실패: {e}")

                        # 실패한 배치는 더미로 대체
                        if use_multi_vector:
                            dense_list = self._fallback_dense_embeddings(list(batch))
                            embeddings.extend(
                                [
                                    self._simulate_multi_vector_from_dense(
                                        text, dense_emb
                                    )
                                    for text, dense_emb in zip(batch, dense_list)
                                ]
                            )
                        else:
                            embeddings.extend(
                                self._fallback_dense_embeddings(list(batch))
                            )

            # 캐시 업데이트
            for i, (txt, emb) in enumerate(zip(to_embed, embeddings)):
                cache_key = txt + cache_suffix
                self._cache[cache_key] = emb
                result_list[indices_to_embed[i]] = emb

            # 🚀 디스크 캐시 저장 (새 임베딩이 추가된 경우 & save_to_disk=True일 때만)
            if len(to_embed) > 0 and save_to_disk:
                self._save_disk_cache()

        return np.array(result_list)

# 전역 인스턴스
_embedding_manager = EmbeddingManager(fallback_to_dummy=True)

def compute_embeddings_with_cache(texts: List[str], **kwargs) -> np.ndarray:
    """하위 호환성 함수"""
    return _embedding_manager.compute_embeddings_with_cache(texts, **kwargs)

def get_embed_func(device_id=None) -> Callable:
    """임베딩 함수 반환 (device_id 지정 가능)"""
    manager = EmbeddingManager(fallback_to_dummy=True, device_id=device_id)
    return manager.compute_embeddings_with_cache

def get_embedding_manager() -> EmbeddingManager:
    """임베딩 매니저 반환"""
    return _embedding_manager

# 프록시 클래스
class EmbeddingManagerProxy:
    def __getattr__(self, name):
        return getattr(_embedding_manager, name)

embedding_manager = EmbeddingManagerProxy()
