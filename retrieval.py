# -*- coding: utf-8 -*-
"""
Koib-V-4.1 — Модуль поиска и переранжирования
=================================================
Гибридный поиск: векторный (FAISS) + BM25 + переранжирование.
Автоматическая маршрутизация запросов к нужному индексу.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from langchain_core.documents import Document

from .indexing import IndexBuilder, DocStore, BM25Index
from .utils import clean_text
from config import (
    INDEX_DIR, DOCSTORE_DIR,
    QUERY_PREFIX, PASSAGE_PREFIX,
    VECTOR_SEARCH_K, BM25_SEARCH_K, FINAL_TOP_K,
    HYBRID_ALPHA, USE_RERANKER, RERANKER_MODEL,
    USE_HYDE, get_device,
)

logger = logging.getLogger("koib.retrieval")


# ═══════════════════════════════════════════════════════════════
# Результат поиска
# ═══════════════════════════════════════════════════════════════

@dataclass
class RetrievalResult:
    """Результат поиска с полным контекстом."""
    chunk_id: str
    content: str                          # Текст чанка / сводка
    full_content: Optional[str] = None    # Полный контент (из docstore)
    score: float = 0.0
    source: str = ""
    page: int = 0
    heading: str = ""
    model: str = "unknown"
    chunk_type: str = "text"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_context_string(self) -> str:
        """Форматировать для подачи в LLM-промпт."""
        source_name = self.source
        parts = [f"[Документ: {source_name}, стр. {self.page}]"]

        if self.heading:
            parts.append(f"Раздел: {self.heading}")

        # Для таблиц/формул — полный контент из docstore
        display_content = self.full_content or self.content

        if self.chunk_type == "table":
            parts.append(f"ТАБЛИЦА:\n{display_content}")
        elif self.chunk_type == "formula":
            parts.append(f"ФОРМУЛА: {display_content}")
        elif self.chunk_type == "figure":
            parts.append(f"РИСУНОК: {display_content}")
        else:
            parts.append(display_content)

        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# Определение типа запроса
# ═══════════════════════════════════════════════════════════════

# Ключевые слова, указывающие на табличные данные
TABLE_KEYWORDS = {
    "таблиц", "табл", "значени", "параметр", "характеристик",
    "спецификаци", "сводк", "данные", "показател", "предел",
    "норм", "допуск", "величин", "измерен",
}

# Ключевые слова для формул
FORMULA_KEYWORDS = {
    "формул", "вычислен", "расчёт", "расчет", "уравнен",
    "выражен", "коэффициент", "зависимост", "математич",
}

# Ключевые слова для схем/рисунков
FIGURE_KEYWORDS = {
    "схем", "рисунок", "рис", "диаграмм", "чертёж", "чертеж",
    "график", "блок-схем", "структур", "компоновк",
}


def _detect_query_intent(query: str) -> Dict[str, float]:
    """
    Определить намерение запроса: нужен ли табличный/формульный контент.

    Returns:
        Словарь {"table": 0.0-1.0, "formula": 0.0-1.0, "figure": 0.0-1.0, "text": 1.0}
    """
    query_lower = query.lower()
    intent = {"table": 0.0, "formula": 0.0, "figure": 0.0, "text": 1.0}

    table_hits = sum(1 for kw in TABLE_KEYWORDS if kw in query_lower)
    formula_hits = sum(1 for kw in FORMULA_KEYWORDS if kw in query_lower)
    figure_hits = sum(1 for kw in FIGURE_KEYWORDS if kw in query_lower)

    total_hits = table_hits + formula_hits + figure_hits
    if total_hits > 0:
        intent["table"] = min(table_hits / 2.0, 1.0)
        intent["formula"] = min(formula_hits / 2.0, 1.0)
        intent["figure"] = min(figure_hits / 2.0, 1.0)
        # Снижаем вес текстового поиска
        intent["text"] = max(0.3, 1.0 - total_hits * 0.2)

    return intent


# ═══════════════════════════════════════════════════════════════
# Гибридный поиск
# ═══════════════════════════════════════════════════════════════

class HybridRetriever:
    """
    Гибридный поисковик: векторный + BM25 + переранжирование.

    Алгоритм:
      1. Векторный поиск по текстовому индексу
      2. Векторный поиск по summary-индексу (таблицы/формулы)
      3. BM25-поиск
      4. Объединение результатов с весами (Reciprocal Rank Fusion)
      5. Переранжирование через cross-encoder (опционально)
      6. Возврат top-k результатов
    """

    def __init__(self, index_builder: Optional[IndexBuilder] = None):
        self.index_builder = index_builder or IndexBuilder()
        self.index_builder.load()
        self._reranker = None

    def _get_reranker(self):
        """Ленивая загрузка переранжировщика."""
        if self._reranker is not None:
            return self._reranker

        if not USE_RERANKER:
            return None

        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANKER_MODEL)
            logger.info(f"Переранжировщик загружен: {RERANKER_MODEL}")
            return self._reranker
        except Exception as exc:
            logger.warning(f"Не удалось загрузить переранжировщик: {exc}")
            self._reranker = None
            return None

    def search(
        self,
        query: str,
        k: int = FINAL_TOP_K,
        model_filter: str = "",
        use_hyde: Optional[bool] = None,
    ) -> List[RetrievalResult]:
        """
        Выполнить гибридный поиск.

        Args:
            query:        Поисковый запрос
            k:            Количество финальных результатов
            model_filter: Фильтр по модели КОИБ
            use_hyde:     Использовать HyDE (None = из конфига)

        Returns:
            Список RetrievalResult, отсортированных по релевантности
        """
        # Определяем намерение запроса
        intent = _detect_query_intent(query)

        # HyDE: генерируем гипотетический ответ для лучшего поиска
        search_query = query
        if (use_hyde if use_hyde is not None else USE_HYDE):
            search_query = self._apply_hyde(query) or query

        # ── 1. Векторный поиск ──────────────────────────────────
        vector_results = self._vector_search(search_query, intent, model_filter)

        # ── 2. BM25-поиск ──────────────────────────────────────
        bm25_results = self._bm25_search(query, model_filter)

        # ── 3. Объединение (Reciprocal Rank Fusion) ────────────
        fused = self._reciprocal_rank_fusion(vector_results, bm25_results)

        # ── 4. Переранжирование ────────────────────────────────
        if USE_RERANKER and len(fused) > k:
            reranker = self._get_reranker()
            if reranker:
                fused = self._rerank(query, fused, reranker)

        # ── 5. Фильтрация и форматирование ─────────────────────
        results = fused[:k]

        # Подгружаем полный контент из docstore
        for r in results:
            if r.chunk_type in ("table", "formula", "figure") and r.full_content is None:
                full = self.index_builder.docstore.get_content(r.chunk_id)
                if full:
                    r.full_content = full

        return results

    def _vector_search(
        self,
        query: str,
        intent: Dict[str, float],
        model_filter: str,
    ) -> List[RetrievalResult]:
        """Векторный поиск по обоим индексам."""
        results: List[RetrievalResult] = []
        query_text = QUERY_PREFIX + query

        # Текстовый индекс
        if self.index_builder.text_vectorstore and intent["text"] > 0:
            k_text = int(VECTOR_SEARCH_K * intent["text"]) + 3
            try:
                docs = self.index_builder.text_vectorstore.similarity_search_with_score(
                    query_text, k=k_text
                )
                for doc, score in docs:
                    if model_filter and doc.metadata.get("model") != model_filter:
                        continue
                    results.append(RetrievalResult(
                        chunk_id=doc.metadata.get("chunk_id", ""),
                        content=doc.page_content,
                        score=float(score),
                        source=doc.metadata.get("source", ""),
                        page=doc.metadata.get("page", 0),
                        heading=doc.metadata.get("heading", ""),
                        model=doc.metadata.get("model", "unknown"),
                        chunk_type=doc.metadata.get("chunk_type", "text"),
                        metadata=doc.metadata,
                    ))
            except Exception as exc:
                logger.warning(f"Ошибка векторного поиска (текст): {exc}")

        # Summary-индекс (таблицы/формулы)
        if self.index_builder.summary_vectorstore:
            k_struct = int(VECTOR_SEARCH_K * max(intent["table"], intent["formula"], 0.3)) + 3
            try:
                docs = self.index_builder.summary_vectorstore.similarity_search_with_score(
                    query_text, k=k_struct
                )
                for doc, score in docs:
                    if model_filter and doc.metadata.get("model") != model_filter:
                        continue
                    results.append(RetrievalResult(
                        chunk_id=doc.metadata.get("chunk_id", ""),
                        content=doc.page_content,
                        score=float(score),
                        source=doc.metadata.get("source", ""),
                        page=doc.metadata.get("page", 0),
                        heading=doc.metadata.get("heading", ""),
                        model=doc.metadata.get("model", "unknown"),
                        chunk_type=doc.metadata.get("chunk_type", "structured"),
                        metadata=doc.metadata,
                    ))
            except Exception as exc:
                logger.warning(f"Ошибка векторного поиска (summary): {exc}")

        return results

    def _bm25_search(
        self,
        query: str,
        model_filter: str,
    ) -> List[RetrievalResult]:
        """BM25-поиск."""
        results: List[RetrievalResult] = []
        try:
            bm25_results = self.index_builder.bm25.search(query, k=BM25_SEARCH_K)
            for meta, score in bm25_results:
                if model_filter and meta.get("model") != model_filter:
                    continue
                results.append(RetrievalResult(
                    chunk_id=meta.get("chunk_id", ""),
                    content=meta.get("content", ""),  # BM25 хранит полный текст
                    score=score,
                    source=meta.get("source", ""),
                    page=meta.get("page", 0),
                    heading=meta.get("heading", ""),
                    model=meta.get("model", "unknown"),
                    chunk_type=meta.get("chunk_type", "text"),
                    metadata=meta,
                ))
        except Exception as exc:
            logger.warning(f"Ошибка BM25-поиска: {exc}")

        return results

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        k_rrf: int = 60,
    ) -> List[RetrievalResult]:
        """
        Reciprocal Rank Fusion для объединения векторного и BM25 поиска.

        RRF_score = Σ 1/(k + rank_i) для каждого списка результатов.
        """
        rrf_scores: Dict[str, float] = {}
        result_map: Dict[str, RetrievalResult] = {}

        # Векторные результаты
        for rank, r in enumerate(vector_results):
            cid = r.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0) + HYBRID_ALPHA / (k_rrf + rank + 1)
            if cid not in result_map:
                result_map[cid] = r

        # BM25 результаты
        for rank, r in enumerate(bm25_results):
            cid = r.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0) + (1 - HYBRID_ALPHA) / (k_rrf + rank + 1)
            if cid not in result_map:
                result_map[cid] = r

        # Сортируем по RRF-скорам
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

        results = []
        for cid in sorted_ids:
            r = result_map[cid]
            r.score = rrf_scores[cid]
            results.append(r)

        return results

    def _rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        reranker,
        top_n: int = FINAL_TOP_K * 2,
    ) -> List[RetrievalResult]:
        """Переранжировать результаты через cross-encoder."""
        if not results:
            return results

        try:
            pairs = [(query, r.content) for r in results[:top_n]]
            scores = reranker.predict(pairs)

            for r, s in zip(results[:top_n], scores):
                r.score = float(s)

            results[:top_n] = sorted(results[:top_n], key=lambda x: x.score, reverse=True)
            logger.info(f"Переранжирование выполнено для {len(pairs)} пар")
        except Exception as exc:
            logger.warning(f"Ошибка переранжирования: {exc}")

        return results

    def _apply_hyde(self, query: str) -> Optional[str]:
        """
        HyDE (Hypothetical Document Embeddings):
        Генерируем гипотетический ответ на запрос и используем его
        для векторного поиска вместо оригинального запроса.
        """
        try:
            from .generation import LLMClient
            client = LLMClient()
            hypothetical = client.generate(
                f"Ответь кратко на вопрос, как если бы ты был экспертом "
                f"по технической документации:\n\n{query}",
                max_tokens=300,
            )
            if hypothetical and len(hypothetical) > 20:
                logger.info(f"HyDE: сгенерирован гипотетический ответ ({len(hypothetical)} символов)")
                return hypothetical
        except Exception as exc:
            logger.debug(f"HyDE не сработал: {exc}")

        return None
