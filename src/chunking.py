# -*- coding: utf-8 -*-
"""
Koib-V-4.1 — Модуль умного чанкинга
======================================
Разбиение извлечённых элементов на чанки с учётом типа контента:
  - Текст: семантическое разбиение ~800 токенов с перекрытием 10%
  - Таблицы: не дробятся, хранятся целиком в docstore
  - Формулы: не дробятся, хранятся целиком в docstore
  - Рисунки: описание/подпись хранится как отдельный чанк

Для таблиц и формул генерируется LLM-сводка (summary), которая
индексируется векторно, а полный контент хранится в docstore.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from .parsing import DocumentElement
from .utils import clean_text, text_hash, estimate_tokens, truncate_to_tokens
from config import (
    TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP, MIN_CHUNK_LENGTH,
)

logger = logging.getLogger("koib.chunking")


# ═══════════════════════════════════════════════════════════════
# Структура чанка
# ═══════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    """Чанк документа, готовый к индексации."""
    chunk_id: str
    content: str                        # Текст чанка / сводка таблицы
    full_content: Optional[str] = None  # Полный контент (для таблиц/формул в docstore)
    chunk_type: str = "text"            # text | table | formula | figure
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_langchain_doc(self) -> Document:
        """Конвертировать в LangChain Document."""
        return Document(
            page_content=self.content,
            metadata={
                "chunk_id": self.chunk_id,
                "chunk_type": self.chunk_type,
                **self.metadata,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "full_content": self.full_content,
            "chunk_type": self.chunk_type,
            "metadata": self.metadata,
        }


# ═══════════════════════════════════════════════════════════════
# Семантическое разбиение текста
# ═══════════════════════════════════════════════════════════════

def _split_text_semantic(
    text: str,
    max_tokens: int = TEXT_CHUNK_SIZE,
    overlap_tokens: int = TEXT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Семантическое разбиение текста на чанки.

    Стратегия:
      1. Сначала делим по двойным переносам строк (абзацы).
      2. Группируем абзацы, пока не превысим max_tokens.
      3. Добавляем overlap из конца предыдущего чанка.
    """
    if not text or len(text.strip()) < MIN_CHUNK_LENGTH:
        return []

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: List[str] = []
    current_parts: List[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        if current_tokens + para_tokens > max_tokens and current_parts:
            # Сохраняем текущий чанк
            chunk_text = '\n\n'.join(current_parts)
            chunks.append(chunk_text)

            # Overlap: берём последний абзац, если он влезает
            overlap_parts: List[str] = []
            overlap_tok = 0
            for p in reversed(current_parts):
                pt = estimate_tokens(p)
                if overlap_tok + pt > overlap_tokens:
                    break
                overlap_parts.insert(0, p)
                overlap_tok += pt

            current_parts = overlap_parts
            current_tokens = overlap_tok

        current_parts.append(para)
        current_tokens += para_tokens

    # Последний чанк
    if current_parts:
        chunk_text = '\n\n'.join(current_parts)
        if estimate_tokens(chunk_text) >= MIN_CHUNK_LENGTH // 4:
            chunks.append(chunk_text)

    return chunks


# ═══════════════════════════════════════════════════════════════
# Генерация сводки таблицы/формулы (summary)
# ═══════════════════════════════════════════════════════════════

def _generate_table_summary(table_markdown: str, metadata: Dict) -> str:
    """
    Генерация сводки таблицы для индексации.

    Если LLM доступна — используем её, иначе — эвристическая сводка.
    Сводка нужна для векторного поиска: по ней пользователь находит
    таблицу, а полный контент берётся из docstore.
    """
    # Эвристическая сводка (без LLM)
    lines = table_markdown.strip().split('\n')
    header_line = lines[0] if lines else ""
    num_rows = metadata.get("num_rows", 0)
    num_cols = metadata.get("num_cols", 0)

    # Извлекаем заголовки столбцов
    headers = [h.strip() for h in header_line.split('|') if h.strip()]

    summary_parts = [
        f"Таблица ({num_rows} строк, {num_cols} столбцов).",
    ]
    if headers:
        summary_parts.append(f"Столбцы: {', '.join(headers[:10])}.")

    # Добавляем первые 2 строки данных
    data_lines = [l for l in lines[2:] if l.strip() and '---' not in l][:2]
    if data_lines:
        summary_parts.append("Пример данных:")
        for dl in data_lines:
            cells = [c.strip() for c in dl.split('|') if c.strip()]
            summary_parts.append("  " + " | ".join(cells[:5]))

    return " ".join(summary_parts)


def _generate_formula_summary(formula_content: str, metadata: Dict) -> str:
    """Генерация сводки формулы для индексации."""
    formula_type = metadata.get("formula_type", "unknown")
    type_desc = {
        "latex_inline": "Формула (LaTeX, строковая)",
        "latex_block": "Формула (LaTeX, блочная)",
        "suspected_formula": "Подозреваемая формула",
        "unknown": "Формула",
    }.get(formula_type, "Формула")

    # Ограничиваем длину сводки
    content_preview = formula_content[:200]
    return f"{type_desc}: {content_preview}"


# ═══════════════════════════════════════════════════════════════
# Основной класс чанкера
# ═══════════════════════════════════════════════════════════════

class SmartChunker:
    """
    Умный чанкер с разделением по типам контента.

    Текст разбивается на чанки ~800 токенов.
    Таблицы и формулы хранятся целиком с LLM-сводкой для поиска.
    """

    def __init__(
        self,
        chunk_size: int = TEXT_CHUNK_SIZE,
        chunk_overlap: int = TEXT_CHUNK_OVERLAP,
        min_chunk_length: int = MIN_CHUNK_LENGTH,
        llm_summary: bool = False,  # Использовать LLM для сводок таблиц
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self.llm_summary = llm_summary
        self._llm_client = None

    def _get_llm_client(self):
        """Ленивая инициализация LLM-клиента для сводок."""
        if self._llm_client is not None:
            return self._llm_client

        try:
            from .generation import LLMClient
            self._llm_client = LLMClient()
        except Exception:
            logger.warning("LLM недоступна для генерации сводок, используем эвристику")
            self._llm_client = None

        return self._llm_client

    def chunk_elements(self, elements: List[DocumentElement]) -> List[Chunk]:
        """
        Разбить список DocumentElement на чанки.

        Returns:
            Список Chunk, готовых к индексации
        """
        chunks: List[Chunk] = []
        text_buffer: List[DocumentElement] = []  # Буфер для текстовых элементов
        current_heading = ""

        for element in elements:
            # Обновляем заголовок
            if element.element_type == "heading":
                current_heading = element.content

            if element.element_type in ("table", "formula", "figure"):
                # Сначала сбрасываем буфер текста
                if text_buffer:
                    chunks.extend(self._chunk_text_buffer(text_buffer, current_heading))
                    text_buffer = []

                # Структурированный элемент — отдельный чанк
                chunks.append(self._chunk_structured_element(element, current_heading))
            else:
                # Накапливаем текстовые элементы
                text_buffer.append(element)

        # Сброс оставшегося буфера
        if text_buffer:
            chunks.extend(self._chunk_text_buffer(text_buffer, current_heading))

        logger.info(f"Создано {len(chunks)} чанков из {len(elements)} элементов")
        return chunks

    def _chunk_text_buffer(
        self,
        elements: List[DocumentElement],
        heading: str,
    ) -> List[Chunk]:
        """Разбить буфер текстовых элементов на чанки."""
        # Объединяем текст
        combined = '\n\n'.join(e.content for e in elements if e.content.strip())
        if not combined or len(combined.strip()) < self.min_chunk_length:
            return []

        # Разбиваем семантически
        text_chunks = _split_text_semantic(
            combined,
            max_tokens=self.chunk_size,
            overlap_tokens=self.chunk_overlap,
        )

        chunks: List[Chunk] = []
        source = elements[0].source if elements else ""
        page = elements[0].page if elements else 0
        model = elements[0].model if elements else "unknown"

        for i, text in enumerate(text_chunks):
            text = text.strip()
            if len(text) < self.min_chunk_length:
                continue

            chunk_id = text_hash(f"{source}:{page}:{i}:{text[:100]}")

            chunks.append(Chunk(
                chunk_id=chunk_id,
                content=text,
                full_content=None,  # Текст не хранится отдельно
                chunk_type="text",
                metadata={
                    "source": source,
                    "page": page,
                    "heading": heading,
                    "model": model,
                    "chunk_index": i,
                },
            ))

        return chunks

    def _chunk_structured_element(
        self,
        element: DocumentElement,
        heading: str,
    ) -> Chunk:
        """Создать чанк для структурированного элемента (таблица/формула/рисунок)."""
        if element.element_type == "table":
            summary = _generate_table_summary(element.content, element.metadata)
        elif element.element_type == "formula":
            summary = _generate_formula_summary(element.content, element.metadata)
        elif element.element_type == "figure":
            summary = element.content  # Описание/подпись рисунка
        else:
            summary = element.content

        # Попробуем улучшить сводку через LLM
        if self.llm_summary and element.element_type == "table":
            llm = self._get_llm_client()
            if llm:
                try:
                    summary = llm.generate(
                        f"Опиши кратко содержание этой таблицы в 2-3 предложениях "
                        f"для поискового индекса:\n\n{element.content}",
                        max_tokens=200,
                    )
                except Exception:
                    pass  # Оставляем эвристическую сводку

        chunk_id = text_hash(
            f"{element.source}:{element.page}:{element.element_type}:{element.content[:200]}"
        )

        return Chunk(
            chunk_id=chunk_id,
            content=summary,                    # Сводка для векторного поиска
            full_content=element.content,       # Полный контент в docstore
            chunk_type=element.element_type,
            metadata={
                "source": element.source,
                "page": element.page,
                "heading": heading or element.heading,
                "model": element.model,
                "element_id": element.element_id,
                **element.metadata,
            },
        )
