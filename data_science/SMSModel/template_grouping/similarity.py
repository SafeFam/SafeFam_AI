"""문자 n-gram cosine similarity 기반 SMS 템플릿 연결"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .config import TemplateGroupingConfig
from .fingerprint import add_text_fingerprints

GROUP_ID_HASH_LENGTH = 12


class UnionFind:
    """유사 메시지 쌍의 연결 요소를 하나의 그룹으로 합침"""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._rank = [0] * size

    def find(self, item: int) -> int:
        if self._parent[item] != item:
            self._parent[item] = self.find(self._parent[item])
        return self._parent[item]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return

        if self._rank[left_root] < self._rank[right_root]:
            self._parent[left_root] = right_root
        elif self._rank[left_root] > self._rank[right_root]:
            self._parent[right_root] = left_root
        else:
            self._parent[right_root] = left_root
            self._rank[left_root] += 1


def _build_vectorizer(config: TemplateGroupingConfig) -> TfidfVectorizer:
    """예측이 아닌 그룹화 전용 문자 TF-IDF 벡터라이저 생성"""
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_features=config.max_features,
        lowercase=False,
        dtype=np.float32,
        norm="l2",
    )


def _find_similar_pairs(
    texts: list[str],
    config: TemplateGroupingConfig,
) -> list[tuple[int, int, float]]:
    """임계값 이상의 cosine similarity를 가진 인덱스 쌍을 반환합니다."""
    if len(texts) < 2 or not any(text.strip() for text in texts):
        return []

    matrix = _build_vectorizer(config).fit_transform(texts)
    radius = 1.0 - config.similarity_threshold

    neighbors = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        radius=radius,
        n_jobs=-1,
    ).fit(matrix)
    distances, indices = neighbors.radius_neighbors(
        matrix,
        return_distance=True,
        sort_results=True,
    )

    pairs: list[tuple[int, int, float]] = []
    for left, (row_distances, row_indices) in enumerate(
        zip(distances, indices, strict=True)
    ):
        for distance, right_value in zip(row_distances, row_indices, strict=True):
            right = int(right_value)
            if left >= right:
                continue

            similarity = 1.0 - float(distance)
            if similarity >= config.similarity_threshold:
                pairs.append((left, right, similarity))
    return pairs


def _create_stable_group_ids(
    fingerprints: list[str],
    union_find: UnionFind,
) -> list[str]:
    """각 연결 요소의 최소 fingerprint로 안정적인 그룹 ID를 만듭니다."""
    members_by_root: dict[int, list[int]] = {}
    for index in range(len(fingerprints)):
        members_by_root.setdefault(union_find.find(index), []).append(index)

    ids_by_index: dict[int, str] = {}
    for member_indices in members_by_root.values():
        representative = min(fingerprints[index] for index in member_indices)
        group_id = f"tpl_{representative[:GROUP_ID_HASH_LENGTH]}"
        for index in member_indices:
            ids_by_index[index] = group_id

    return [ids_by_index[index] for index in range(len(fingerprints))]


def assign_template_groups(
    df: pd.DataFrame,
    *,
    config: TemplateGroupingConfig | None = None,
    text_column: str = "text_norm",
) -> pd.DataFrame:
    """정규화 메시지에 유사도 기반 template_group_id를 부여합니다."""
    config = config or TemplateGroupingConfig()
    if text_column not in df.columns:
        raise ValueError(f"missing text column: {text_column}")

    result = df.copy().reset_index(drop=True)
    if "text_fingerprint" not in result.columns:
        result = add_text_fingerprints(result, text_column=text_column)

    if result.empty:
        result["template_group_id"] = pd.Series(dtype="string")
        return result

    texts = result[text_column].astype(str).tolist()
    fingerprints = result["text_fingerprint"].astype(str).tolist()
    union_find = UnionFind(len(result))

    for left, right, _similarity in _find_similar_pairs(texts, config):
        union_find.union(left, right)

    result["template_group_id"] = _create_stable_group_ids(
        fingerprints,
        union_find,
    )
    return result
