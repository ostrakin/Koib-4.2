# -*- coding: utf-8 -*-
"""
Koib-V-4.1 — Конфигурация системы
===================================
Централизованная конфигурация: пути, модели, параметры чанкинга,
поиска, генерации. Всё управляется через переменные окружения
или значения по умолчанию.
"""

import os
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# Пути
# ═══════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = Path(os.getenv("KOIB_DOCS_DIR", str(DATA_DIR / "docs")))
OUTPUT_DIR = Path(os.getenv("KOIB_OUTPUT_DIR", str(BASE_DIR / "output")))

# Поддиректории output
INDEX_DIR = OUTPUT_DIR / "index"
DOCSTORE_DIR = OUTPUT_DIR / "docstore"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = OUTPUT_DIR / "logs"
METADATA_DIR = OUTPUT_DIR / "metadata"


# ═══════════════════════════════════════════════════════════════
# Режим работы: локальный / OpenAI API
# ═══════════════════════════════════════════════════════════════

# "local"  — все модели загружаются локально (HuggingFace)
# "openai" — эмбеддинги и/или LLM через OpenAI API
# "gigachat" — LLM через GigaChat API (Сбер)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gigachat")  # local | openai | gigachat
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")  # local | openai


# ═══════════════════════════════════════════════════════════════
# Эмбеддинг-модели
# ═══════════════════════════════════════════════════════════════

LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "intfloat/multilingual-e5-large"   # отлично работает для русского
)

OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small"
)

PASSAGE_PREFIX = "passage: "   # для instruction-tuned эмбеддингов
QUERY_PREFIX = "query: "

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# ═══════════════════════════════════════════════════════════════
# Параметры чанкинга
# ═══════════════════════════════════════════════════════════════

TEXT_CHUNK_SIZE = int(os.getenv("TEXT_CHUNK_SIZE", "800"))       # токенов (примерно)
TEXT_CHUNK_OVERLAP = int(os.getenv("TEXT_CHUNK_OVERLAP", "80"))  # 10% overlap
MIN_CHUNK_LENGTH = int(os.getenv("MIN_CHUNK_LENGTH", "50"))

# Таблицы и формулы не дробятся — хранятся целиком в docstore,
# в векторный индекс идёт только LLM-сводка (summary).


# ═══════════════════════════════════════════════════════════════
# Параметры поиска
# ═══════════════════════════════════════════════════════════════

# Сколько кандидатов извлекает векторный поиск
VECTOR_SEARCH_K = int(os.getenv("VECTOR_SEARCH_K", "20"))

# Сколько кандидатов извлекает BM25
BM25_SEARCH_K = int(os.getenv("BM25_SEARCH_K", "20"))

# Сколько финальных результатов после переранжирования
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "5"))

# Вес векторного поиска при гибридном объединении (0–1),
# вес BM25 = 1 - HYBRID_ALPHA
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.6"))

# Использовать ли переранжирование
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cointegrated/rubert-tiny2"   # лёгкая модель для русского, можно заменить на bge-reranker
)

# Использовать ли HyDE
USE_HYDE = os.getenv("USE_HYDE", "false").lower() == "true"


# ═══════════════════════════════════════════════════════════════
# LLM-параметры (генерация ответов)
# ═══════════════════════════════════════════════════════════════

# GigaChat
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS", "")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat")
GIGACHAT_TEMPERATURE = float(os.getenv("GIGACHAT_TEMPERATURE", "0.2"))
GIGACHAT_MAX_TOKENS = int(os.getenv("GIGACHAT_MAX_TOKENS", "2048"))

# OpenAI
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "2048"))

# Локальная LLM (Ollama / llama-cpp)
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "IlyaGusev/saiga_mistral_7b")
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:11434")


# ═══════════════════════════════════════════════════════════════
# OCR
# ═══════════════════════════════════════════════════════════════

OCR_DPI = int(os.getenv("OCR_DPI", "300"))
OCR_MIN_TEXT_CHARS = int(os.getenv("OCR_MIN_TEXT_CHARS", "50"))
MIN_IMAGE_WIDTH = int(os.getenv("MIN_IMAGE_WIDTH", "80"))
MIN_IMAGE_HEIGHT = int(os.getenv("MIN_IMAGE_HEIGHT", "80"))


# ═══════════════════════════════════════════════════════════════
# Устройство
# ═══════════════════════════════════════════════════════════════

def get_device() -> str:
    """Определить доступное устройство (cuda / cpu)."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ═══════════════════════════════════════════════════════════════
# Парсинг: выбор движка
# ═══════════════════════════════════════════════════════════════

# "pymupdf" — базовый парсер (всегда доступен)
# "docling" — IBM Docling (лучшее качество, нужен pip install docling)
PARSING_ENGINE = os.getenv("PARSING_ENGINE", "pymupdf")


def ensure_dirs() -> None:
    """Создать все необходимые директории."""
    for d in [DOCS_DIR, OUTPUT_DIR, INDEX_DIR, DOCSTORE_DIR,
              FIGURES_DIR, LOGS_DIR, METADATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)
