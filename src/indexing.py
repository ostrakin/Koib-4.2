# -*- coding: utf-8 -*-
"""
Koib-V-4.1 — Модуль индексации
=================================
Мультимодальный индекс: текстовый (FAISS) + BM25 + docstore.

Архитектура:
  1. Текстовый индекс (FAISS) — для обычных текстовых чанков
  2. Summary-индекс (FAISS) — сводки таблиц/формул для поиска
  3. BM25-индекс — ключевой поиск (sparse retrieval)
  4. Docstore — хранилище полного контента таблиц/формул

Метаданные каждого чанка: source, page, heading, model, chunk_type.
"""

import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .chunking import Chunk
from .utils import text_hash
from config import (
    INDEX_DIR, DOCSTORE_DIR, METADATA_DIR,
    LOCAL_EMBEDDING_MODEL, OPENAI_EMBEDDING_MODEL,
    EMBEDDING_PROVIDER, OPENAI_API_KEY,
    PASSAGE_PREFIX, QUERY_PREFIX, get_device, ensure_dirs,
)

logger = logging.getLogger("koib.indexing")


# ═══════════════════════════════════════════════════════════════
# Docstore — хранилище полного контента
# ═══════════════════════════════════════════════════════════════

class DocStore:
    """
    Хранилище полного контента чанков (таблицы, формулы).

    Ключ: chunk_id → значение: полный контент + метаданные.
    Реализация: JSON-файл для простоты (можно заменить на SQLite).
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path or DOCSTORE_DIR / "docstore.json"
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Загрузить docstore из файла."""
        if self.path.exists():
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    self._data = json.load(f)
                logger.info(f"DocStore загружен: {len(self._data)} записей")
            except Exception as exc:
                logger.warning(f"Ошибка загрузки DocStore: {exc}")
                self._data = {}

    def save(self) -> None:
        """Сохранить docstore в файл."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        logger.info(f"DocStore сохранён: {len(self._data)} записей")

    def add(self, chunk_id: str, full_content: str, metadata: Dict[str, Any]) -> None:
        """Добавить запись в docstore."""
        self._data[chunk_id] = {
            "full_content": full_content,
            "metadata": metadata,
        }

    def get(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Получить запись из docstore."""
        return self._data.get(chunk_id)

    def get_content(self, chunk_id: str) -> Optional[str]:
        """Получить полный контент по chunk_id."""
        entry = self._data.get(chunk_id)
        return entry["full_content"] if entry else None

    @property
    def size(self) -> int:
        return len(self._data)

    def remove_by_source(self, source: str) -> int:
        """Удалить все записи от указанного источника. Возвращает кол-во удалённых."""
        to_remove = [
            cid for cid, entry in self._data.items()
            if entry.get("metadata", {}).get("source") == source
        ]
        for cid in to_remove:
            del self._data[cid]
        return len(to_remove)


# ═══════════════════════════════════════════════════════════════
# BM25-индекс (sparse retrieval)
# ═══════════════════════════════════════════════════════════════

class BM25Index:
    """
    BM25-индекс для ключевого поиска.

    Использует rank_bm25 библиотеку. Хранит тексты чанков и их метаданные.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path or INDEX_DIR / "bm25_index.pkl"
        self._texts: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._bm25 = None

    def build(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """Построить BM25-индекс."""
        try:
            from rank_bm25 import BM25Okapi
            import re

            # Токенизация (простая, с учётом русского)
            tokenized = []
            for text in texts:
                tokens = re.findall(r'\w+', text.lower())
                tokenized.append(tokens)

            self._texts = texts
            self._metadatas = metadatas
            self._bm25 = BM25Okapi(tokenized)
            logger.info(f"BM25-индекс построен: {len(texts)} документов")
        except ImportError:
            logger.warning("rank_bm25 не установлен. BM25-поиск недоступен.")
            logger.warning("Установите: pip install rank-bm25")

    def search(self, query: str, k: int = 20) -> List[Tuple[Dict[str, Any], float]]:
        """
        Поиск по BM25.

        Returns:
            Список (metadata, score), отсортированный по убыванию score.
        """
        if self._bm25 is None:
            return []

        import re
        query_tokens = re.findall(r'\w+', query.lower())
        scores = self._bm25.get_scores(query_tokens)

        # Сортируем по убыванию score
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed_scores[:k]:
            if score > 0:
                results.append((self._metadatas[idx], float(score)))

        return results

    def save(self) -> None:
        """Сохранить BM25-индекс."""
        if self._bm25 is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "texts": self._texts,
            "metadatas": self._metadatas,
            "bm25_corpus": self._bm25.corpus if hasattr(self._bm25, 'corpus') else None,
        }
        with open(self.path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"BM25-индекс сохранён: {self.path}")

    def load(self) -> bool:
        """Загрузить BM25-индекс. Возвращает True при успехе."""
        if not self.path.exists():
            return False
        try:
            with open(self.path, 'rb') as f:
                data = pickle.load(f)
            self._texts = data["texts"]
            self._metadatas = data["metadatas"]
            # Перестраиваем BM25 из текстов
            self.build(self._texts, self._metadatas)
            return True
        except Exception as exc:
            logger.warning(f"Ошибка загрузки BM25-индекса: {exc}")
            return False


# ═══════════════════════════════════════════════════════════════
# Построитель индексов
# ═══════════════════════════════════════════════════════════════

class IndexBuilder:
    """
    Построитель мультимодального индекса.

    Создаёт:
      - FAISS-индекс для текстовых чанков
      - FAISS-индекс для сводок таблиц/формул
      - BM25-индекс
      - Docstore
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or INDEX_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.text_index_path = self.output_dir / "text_index"
        self.summary_index_path = self.output_dir / "summary_index"

        self.docstore = DocStore()
        self.bm25 = BM25Index(self.output_dir / "bm25_index.pkl")

        self.text_vectorstore: Optional[FAISS] = None
        self.summary_vectorstore: Optional[FAISS] = None
        self.embeddings = None

    def _get_embeddings(self):
        """Инициализировать модель эмбеддингов."""
        if self.embeddings is not None:
            return self.embeddings

        device = get_device()

        if EMBEDDING_PROVIDER == "openai" and OPENAI_API_KEY:
            from langchain_openai import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings(
                model=OPENAI_EMBEDDING_MODEL,
                openai_api_key=OPENAI_API_KEY,
            )
            logger.info(f"Эмбеддинги: OpenAI {OPENAI_EMBEDDING_MODEL}")
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=LOCAL_EMBEDDING_MODEL,
                encode_kwargs={"normalize_embeddings": True},
                model_kwargs={"device": device},
            )
            logger.info(f"Эмбеддинги: {LOCAL_EMBEDDING_MODEL} ({device})")

        return self.embeddings

    def build(self, chunks: List[Chunk]) -> None:
        """
        Построить все индексы из списка чанков.

        Args:
            chunks: Список Chunk от SmartChunker
        """
        ensure_dirs()

        embeddings = self._get_embeddings()

        # Разделяем чанки по типу
        text_chunks = [c for c in chunks if c.chunk_type == "text"]
        structured_chunks = [c for c in chunks if c.chunk_type in ("table", "formula", "figure")]

        logger.info(f"Чанков: {len(text_chunks)} текстовых, {len(structured_chunks)} структурированных")

        # ── 1. Текстовый FAISS-индекс ──────────────────────────
        if text_chunks:
            texts = [PASSAGE_PREFIX + c.content for c in text_chunks]
            metadatas = []
            for c in text_chunks:
                meta = dict(c.metadata)
                meta["chunk_id"] = c.chunk_id
                meta["chunk_type"] = c.chunk_type
                metadatas.append(meta)

            self.text_vectorstore = FAISS.from_texts(
                texts, embeddings, metadatas=metadatas
            )
            self.text_vectorstore.save_local(str(self.text_index_path))
            logger.info(f"Текстовый FAISS сохранён: {self.text_index_path}")

        # ── 2. Summary FAISS-индекс (таблицы + формулы) ────────
        if structured_chunks:
            summary_texts = []
            summary_metadatas = []

            for c in structured_chunks:
                # Добавляем контекст: тип + заголовок + сводка
                context_prefix = f"[{c.chunk_type.upper()}] "
                if c.metadata.get("heading"):
                    context_prefix += f"Раздел: {c.metadata['heading']}. "

                summary_texts.append(PASSAGE_PREFIX + context_prefix + c.content)

                meta = dict(c.metadata)
                meta["chunk_id"] = c.chunk_id
                meta["chunk_type"] = c.chunk_type
                summary_metadatas.append(meta)

                # Сохраняем полный контент в docstore
                if c.full_content:
                    self.docstore.add(c.chunk_id, c.full_content, c.metadata)

            self.summary_vectorstore = FAISS.from_texts(
                summary_texts, embeddings, metadatas=summary_metadatas
            )
            self.summary_vectorstore.save_local(str(self.summary_index_path))
            logger.info(f"Summary FAISS сохранён: {self.summary_index_path}")

        # ── 3. BM25-индекс ─────────────────────────────────────
        all_texts = []
        all_metas = []
        for c in chunks:
            # Для BM25 используем content (сводку) + полный контент, если есть
            bm25_text = c.content
            if c.full_content:
                bm25_text += "\n" + c.full_content
            all_texts.append(bm25_text)
            meta = dict(c.metadata)
            meta["chunk_id"] = c.chunk_id
            meta["chunk_type"] = c.chunk_type
            all_metas.append(meta)

        if all_texts:
            self.bm25.build(all_texts, all_metas)
            self.bm25.save()

        # ── 4. Docstore ────────────────────────────────────────
        self.docstore.save()

        # ── 5. Метаданные индексов ─────────────────────────────
        meta = {
            "num_text_chunks": len(text_chunks),
            "num_structured_chunks": len(structured_chunks),
            "total_chunks": len(chunks),
            "docstore_size": self.docstore.size,
            "embedding_model": (
                OPENAI_EMBEDDING_MODEL
                if EMBEDDING_PROVIDER == "openai"
                else LOCAL_EMBEDDING_MODEL
            ),
        }
        meta_path = METADATA_DIR / "index_meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(f"Индексы построены: {meta}")

    def add_chunks(self, new_chunks: List[Chunk]) -> None:
        """
        Инкрементальное добавление чанков (без перестройки всего индекса).

        Удаляет существующие чанки от того же источника и добавляет новые.
        """
        if not new_chunks:
            return

        # Определяем источники
        sources = set(c.metadata.get("source", "") for c in new_chunks)
        logger.info(f"Инкрементальное обновление: источники = {sources}")

        # Загружаем существующие индексы
        self.load()

        # Удаляем старые чанки из docstore
        for source in sources:
            removed = self.docstore.remove_by_source(source)
            if removed:
                logger.info(f"Удалено {removed} записей из docstore для {source}")

        # FAISS не поддерживает удаление по метаданным, поэтому
        # при инкрементальном обновлении нужно перестроить.
        # Для простоты: загружаем все существующие чанки из метаданных,
        # объединяем с новыми и перестраиваем.
        # TODO: перейти на Chroma для настоящего инкрементального обновления

        # Получаем все существующие чанки из сохранённых файлов
        existing_chunks = self._load_existing_chunks()

        # Фильтруем: удаляем чанки от обновляемых источников
        filtered = [c for c in existing_chunks if c.metadata.get("source", "") not in sources]

        # Добавляем новые
        all_chunks = filtered + new_chunks
        logger.info(f"Перестройка индекса: {len(filtered)} старых + {len(new_chunks)} новых чанков")

        # Перестраиваем
        self.build(all_chunks)

    def _load_existing_chunks(self) -> List[Chunk]:
        """Загрузить существующие чанки из файлов метаданных."""
        chunks_path = METADATA_DIR / "all_chunks.json"
        if not chunks_path.exists():
            return []

        try:
            with open(chunks_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [
                Chunk(
                    chunk_id=d["chunk_id"],
                    content=d["content"],
                    full_content=d.get("full_content"),
                    chunk_type=d["chunk_type"],
                    metadata=d["metadata"],
                )
                for d in data
            ]
        except Exception:
            return []

    def load(self) -> bool:
        """Загрузить существующие индексы."""
        embeddings = self._get_embeddings()

        try:
            if self.text_index_path.exists():
                self.text_vectorstore = FAISS.load_local(
                    str(self.text_index_path),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info("Текстовый FAISS загружен")
        except Exception as exc:
            logger.warning(f"Ошибка загрузки текстового FAISS: {exc}")

        try:
            if self.summary_index_path.exists():
                self.summary_vectorstore = FAISS.load_local(
                    str(self.summary_index_path),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info("Summary FAISS загружен")
        except Exception as exc:
            logger.warning(f"Ошибка загрузки summary FAISS: {exc}")

        self.bm25.load()
        self.docstore._load()

        return (self.text_vectorstore is not None or
                self.summary_vectorstore is not None)

    def save_chunks_snapshot(self, chunks: List[Chunk]) -> None:
        """Сохранить снимок чанков для инкрементального обновления."""
        chunks_path = METADATA_DIR / "all_chunks.json"
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        data = [c.to_dict() for c in chunks]
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Снимок чанков сохранён: {chunks_path} ({len(chunks)} шт.)")
