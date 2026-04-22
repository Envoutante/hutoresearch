from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_SIMILARITY_THRESHOLD = 0.8
DEFAULT_HF_MIRROR = "https://hf-mirror.com"


def _ensure_hf_mirror() -> None:
    """Use domestic HF mirror by default; allow explicit user override."""
    os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_MIRROR)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


class DescriptionEmbeddingIndex:
    """Semantic similarity index for experiment descriptions."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self._model: Any | None = None
        self._items: list[dict[str, Any]] = []
        self._embeddings: np.ndarray | None = None

    def _get_model(self):
        if self._model is not None:
            return self._model
        _ensure_hf_mirror()
        try:
            module = importlib.import_module("sentence_transformers")
            model_cls = getattr(module, "SentenceTransformer")
        except Exception:
            return None
        self._model = model_cls(self.model_name)
        return self._model

    @staticmethod
    def _extract_description(item: dict[str, Any]) -> str:
        return str(item.get("description") or "").strip()

    def rebuild(self, items: list[dict[str, Any]]) -> None:
        self._items = [x for x in items if self._extract_description(x)]
        if not self._items:
            self._embeddings = None
            return

        model = self._get_model()
        if model is None:
            self._embeddings = None
            return

        descriptions = [self._extract_description(x) for x in self._items]
        vectors = model.encode(descriptions, normalize_embeddings=True)
        self._embeddings = np.asarray(vectors, dtype=np.float32)

    def add_item(self, item: dict[str, Any]) -> None:
        desc = self._extract_description(item)
        if not desc:
            return
        self._items.append(item)
        if self._embeddings is None:
            self.rebuild(self._items)
            return

        model = self._get_model()
        if model is None:
            self._embeddings = None
            return

        vec = model.encode([desc], normalize_embeddings=True)
        vec_arr = np.asarray(vec, dtype=np.float32)
        self._embeddings = np.vstack([self._embeddings, vec_arr])

    def is_duplicate(
        self,
        new_description: str,
    ) -> tuple[bool, dict[str, Any] | None, float]:
        text = str(new_description or "").strip()
        if not text:
            return False, None, 0.0
        if not self._items:
            return False, None, 0.0
        if self._embeddings is None:
            return False, None, 0.0

        model = self._get_model()
        if model is None:
            return False, None, 0.0

        new_vec = np.asarray(
            model.encode([text], normalize_embeddings=True),
            dtype=np.float32,
        )[0]
        sims = self._embeddings @ new_vec
        idx = int(np.argmax(sims))
        score = float(sims[idx])
        if score >= self.similarity_threshold:
            return True, self._items[idx], score
        return False, None, score

    @classmethod
    def from_failure_file(
        cls,
        path: Path,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> "DescriptionEmbeddingIndex":
        inst = cls(
            model_name=model_name,
            similarity_threshold=similarity_threshold,
        )
        if not path.exists():
            inst.rebuild([])
            return inst

        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            inst.rebuild([])
            return inst

        items = obj.get("items") if isinstance(obj, dict) else []
        if not isinstance(items, list):
            items = []
        filtered = [x for x in items if isinstance(x, dict)]
        inst.rebuild(filtered)
        return inst


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def _compute_pair_similarity(
    s1: str,
    s2: str,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> float | None:
    _ensure_hf_mirror()
    try:
        module = importlib.import_module("sentence_transformers")
        model_cls = getattr(module, "SentenceTransformer")
        model = model_cls(model_name)
    except Exception:
        return None

    vectors = np.asarray(
        model.encode([s1, s2], normalize_embeddings=True),
        dtype=np.float32,
    )
    return _cosine_similarity(vectors[0], vectors[1])


def main() -> int:
    # Directly edit these two lines for quick local testing.
    s1 = "Increase ASPECT_RATIO from 64 to 96 to expand model width to 768 dimensions."
    s2 = "Raise ASPECT_RATIO from 64 to 96, widening the model (embedding 768->896, attention heads 6->7) while keeping depth at 8."
    threshold = DEFAULT_SIMILARITY_THRESHOLD
    model_name = DEFAULT_EMBEDDING_MODEL

    if not s1 or not s2:
        print("Both sentences are required.")
        return 2

    score = _compute_pair_similarity(s1, s2, model_name=model_name)
    if score is None:
        print(
            "Failed to load sentence-transformers model. Please install dependencies."
        )
        return 1

    is_dup = score >= threshold
    print(f"similarity={score:.4f}")
    print(f"threshold={threshold:.4f}")
    print(f"is_duplicate={is_dup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
