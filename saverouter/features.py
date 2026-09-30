"""Optional text-plus-dense feature encoder used by the public examples."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer


class TextFeatureEncoder:
    """Build portable sparse features from prompt text and optional embeddings."""

    def __init__(
        self,
        *,
        word_max_features: int = 30_000,
        char_max_features: int = 30_000,
        word_min_df: int = 2,
        char_min_df: int = 3,
        dense_scale: float = 1.0,
    ) -> None:
        self.word = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=int(word_min_df),
            max_features=int(word_max_features),
            sublinear_tf=True,
        )
        self.char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=int(char_min_df),
            max_features=int(char_max_features),
            sublinear_tf=True,
        )
        self.dense_scale = float(dense_scale)
        self.has_text_ = False
        self.has_dense_ = False

    @staticmethod
    def _validate(texts, dense) -> int:
        if texts is None and dense is None:
            raise ValueError("Provide texts, dense features, or both.")
        n_samples = len(texts) if texts is not None else int(dense.shape[0])
        if dense is not None and int(dense.shape[0]) != n_samples:
            raise ValueError("texts and dense features must have equal length.")
        return n_samples

    def fit_transform(
        self,
        *,
        texts: Optional[Sequence[str]] = None,
        dense=None,
    ):
        self._validate(texts, dense)
        parts = []
        self.has_text_ = texts is not None
        self.has_dense_ = dense is not None
        if texts is not None:
            values = [str(text) for text in texts]
            parts.extend([self.word.fit_transform(values), self.char.fit_transform(values)])
        if dense is not None:
            parts.append(csr_matrix(np.asarray(dense, dtype=np.float32) * self.dense_scale))
        return hstack(parts, format="csr") if len(parts) > 1 else parts[0]

    def transform(
        self,
        *,
        texts: Optional[Sequence[str]] = None,
        dense=None,
    ):
        self._validate(texts, dense)
        if (texts is not None) != self.has_text_ or (dense is not None) != self.has_dense_:
            raise ValueError("transform() must use the modalities supplied to fit_transform().")
        parts = []
        if texts is not None:
            values = [str(text) for text in texts]
            parts.extend([self.word.transform(values), self.char.transform(values)])
        if dense is not None:
            parts.append(csr_matrix(np.asarray(dense, dtype=np.float32) * self.dense_scale))
        return hstack(parts, format="csr") if len(parts) > 1 else parts[0]
