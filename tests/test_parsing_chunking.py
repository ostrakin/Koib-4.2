# -*- coding: utf-8 -*-
"""
Тесты для модулей парсинга и чанкинга RAG-системы Koib-V-4.1

Запуск:
    pytest tests/test_parsing_chunking.py -v
"""

import pytest
from pathlib import Path
import tempfile
import os


class TestParsing:
    """Тесты модуля парсинга."""

    def test_document_element_creation(self):
        """Проверка создания DocumentElement."""
        from src.parsing import DocumentElement
        
        elem = DocumentElement(
            content="Тестовый контент",
            element_type="text",
            source="test.pdf",
            page=1,
            heading="Раздел 1",
        )
        
        assert elem.content == "Тестовый контент"
        assert elem.element_type == "text"
        assert elem.source == "test.pdf"
        assert elem.page == 1
        assert elem.heading == "Раздел 1"
        assert elem.element_id is not None
        assert len(elem.element_id) == 12  # MD5 hash первые 12 символов

    def test_document_element_to_dict(self):
        """Проверка конвертации DocumentElement в dict."""
        from src.parsing import DocumentElement
        
        elem = DocumentElement(
            content="Тест",
            element_type="table",
            source="test.pdf",
        )
        
        d = elem.to_dict()
        assert d["content"] == "Тест"
        assert d["element_type"] == "table"
        assert d["source"] == "test.pdf"

    def test_document_element_is_structured(self):
        """Проверка свойства is_structured."""
        from src.parsing import DocumentElement
        
        text_elem = DocumentElement(content="текст", element_type="text")
        table_elem = DocumentElement(content="таблица", element_type="table")
        formula_elem = DocumentElement(content="формула", element_type="formula")
        figure_elem = DocumentElement(content="рисунок", element_type="figure")
        
        assert text_elem.is_structured is False
        assert table_elem.is_structured is True
        assert formula_elem.is_structured is True
        assert figure_elem.is_structured is True

    def test_clean_text(self):
        """Проверка очистки текста."""
        from src.utils import clean_text
        
        assert clean_text("  много   пробелов  ") == "много пробелов"
        assert clean_text("текст\n\n\nс переносами") == "текст\n\nс переносами"
        assert clean_text("") == ""

    def test_text_hash(self):
        """Проверка хеширования текста."""
        from src.utils import text_hash
        
        h1 = text_hash("одинаковый текст")
        h2 = text_hash("одинаковый текст")
        h3 = text_hash("другой текст")
        
        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 12

    def test_detect_model_from_filename(self):
        """Проверка определения модели по имени файла."""
        from src.utils import detect_model_from_filename
        
        assert detect_model_from_filename("koib-2010_manual.pdf") == "koib2010"
        assert detect_model_from_filename("КОИБ_2017А.docx") == "koib2017a"
        assert detect_model_from_filename("koib_2017б.pdf") == "koib2017b"
        assert detect_model_from_filename("unknown.pdf") == "unknown"

    def test_estimate_tokens(self):
        """Проверка оценки количества токенов."""
        from src.utils import estimate_tokens
        
        assert estimate_tokens("abcd") >= 1  # ~1 токен
        assert estimate_tokens("a" * 100) > 10  # ~25 токенов

    def test_truncate_to_tokens(self):
        """Проверка обрезки текста по токенам."""
        from src.utils import truncate_to_tokens
        
        text = "a" * 100
        truncated = truncate_to_tokens(text, max_tokens=10)
        assert len(truncated) <= 40 + 3  # 10 токенов * 4 + "..."


class TestChunking:
    """Тесты модуля чанкинга."""

    def test_chunk_creation(self):
        """Проверка создания Chunk."""
        from src.chunking import Chunk
        
        chunk = Chunk(
            chunk_id="test-123",
            content="Тестовый контент чанка",
            chunk_type="text",
            metadata={"source": "test.pdf", "page": 1},
        )
        
        assert chunk.chunk_id == "test-123"
        assert chunk.content == "Тестовый контент чанка"
        assert chunk.chunk_type == "text"
        assert chunk.metadata["source"] == "test.pdf"

    def test_chunk_to_langchain_doc(self):
        """Проверка конвертации в LangChain Document."""
        from src.chunking import Chunk
        
        chunk = Chunk(
            chunk_id="test-456",
            content="Контент",
            chunk_type="table",
            metadata={"source": "test.pdf"},
        )
        
        doc = chunk.to_langchain_doc()
        assert doc.page_content == "Контент"
        assert doc.metadata["chunk_id"] == "test-456"
        assert doc.metadata["chunk_type"] == "table"

    def test_chunk_to_dict(self):
        """Проверка конвертации Chunk в dict."""
        from src.chunking import Chunk
        
        chunk = Chunk(
            chunk_id="test-789",
            content="Данные",
            full_content="Полные данные",
            chunk_type="formula",
        )
        
        d = chunk.to_dict()
        assert d["chunk_id"] == "test-789"
        assert d["content"] == "Данные"
        assert d["full_content"] == "Полные данные"
        assert d["chunk_type"] == "formula"

    def test_smart_chunker_initialization(self):
        """Проверка инициализации SmartChunker."""
        from src.chunking import SmartChunker
        
        chunker = SmartChunker(
            chunk_size=500,
            chunk_overlap=50,
            min_chunk_length=30,
            llm_summary=False,
        )
        
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50
        assert chunker.min_chunk_length == 30
        assert chunker.llm_summary is False

    def test_split_text_semantic_empty(self):
        """Проверка разбиения пустого текста."""
        from src.chunking import _split_text_semantic
        
        assert _split_text_semantic("") == []
        assert _split_text_semantic("   ") == []
        assert _split_text_semantic("к") == []  # слишком короткий

    def test_split_text_semantic_short(self):
        """Проверка разбиения короткого текста."""
        from src.chunking import _split_text_semantic
        
        text = "Это короткий текст из одного предложения."
        chunks = _split_text_semantic(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_generate_table_summary(self):
        """Проверка генерации сводки таблицы."""
        from src.chunking import _generate_table_summary
        
        markdown_table = """| Колонка 1 | Колонка 2 |
| --- | --- |
| Значение 1 | Значение 2 |
| Значение 3 | Значение 4 |"""
        
        metadata = {"num_rows": 3, "num_cols": 2}
        summary = _generate_table_summary(markdown_table, metadata)
        
        assert "Таблица" in summary
        assert "3 строк" in summary
        assert "2 столбцов" in summary

    def test_generate_formula_summary(self):
        """Проверка генерации сводки формулы."""
        from src.chunking import _generate_formula_summary
        
        formula = "$E = mc^2$"
        metadata = {"formula_type": "latex_inline"}
        summary = _generate_formula_summary(formula, metadata)
        
        assert "Формула" in summary or "LaTeX" in summary


class TestIntegration:
    """Интеграционные тесты парсинга и чанкинга."""

    def test_parse_and_chunk_mock_document(self):
        """Проверка полного цикла на мок-данных."""
        from src.parsing import DocumentElement
        from src.chunking import SmartChunker
        
        # Создаём мок-элементы
        elements = [
            DocumentElement(content="Заголовок раздела", element_type="heading", source="mock.pdf"),
            DocumentElement(content="Текст абзаца 1. " * 20, element_type="text", source="mock.pdf", page=1),
            DocumentElement(content="Текст абзаца 2. " * 20, element_type="text", source="mock.pdf", page=1),
            DocumentElement(content="| A | B |\n|---|---|\n| 1 | 2 |", element_type="table", source="mock.pdf", page=2),
        ]
        
        # Чанкинг
        chunker = SmartChunker(llm_summary=False)
        chunks = chunker.chunk_elements(elements)
        
        assert len(chunks) > 0
        assert any(c.chunk_type == "text" for c in chunks)
        assert any(c.chunk_type == "table" for c in chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
