# -*- coding: utf-8 -*-
"""
Koib-V-4.1 — Общие утилиты
============================
Очистка текста, хеширование, определение модели КОИБ,
подписи к рисункам и прочие вспомогательные функции.
"""

import os
import re
import hashlib
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("koib")

# ═══════════════════════════════════════════════════════════════
# Модели КОИБ — паттерны для определения
# ═══════════════════════════════════════════════════════════════

KNOWN_MODELS = {"koib2010", "koib2017a", "koib2017b"}

KOIB_MODEL_PATTERNS: Dict[str, List[str]] = {
    "koib2010": [
        r"КОИБ[-\s]?2010", r"КОИБ\s*2010", r"0912054",
        r"PRINT_KOIB2010", r"2010.*руководство",
        r"модель\s*17404049\.438900\.001",
    ],
    "koib2017a": [
        r"КОИБ[-\s]?2017\s*[АA]", r"КОИБ[-\s]?2017А",
        r"модель\s*17404049\.5013009\.008-01",
        r"17404049\.5013009", r"PRINT_KOIB2017[АA]",
    ],
    "koib2017b": [
        r"КОИБ[-\s]?2017\s*[БB]", r"КОИБ[-\s]?2017Б",
        r"БАВУ\.201119", r"0912053", r"PRINT_KOIB2017[БB]",
    ],
}

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "koib2010": "КОИБ-2010",
    "koib2017a": "КОИБ-2017А",
    "koib2017b": "КОИБ-2017Б",
    "unknown": "Неизвестная модель",
}

FIGURE_CAPTION_PATTERNS = [
    re.compile(r"(Рис(?:ун(?:ок|ке))[\s.]?\s*[\d.]+[^\n]*)", re.IGNORECASE),
    re.compile(r"(Рис\.?\s*[\d.]+[^\n]*)", re.IGNORECASE),
    re.compile(r"(Фиг\.?\s*[\d.]+[^\n]*)", re.IGNORECASE),
    re.compile(r"(Схема[\s.]*[\d.]+[^\n]*)", re.IGNORECASE),
]


# ═══════════════════════════════════════════════════════════════
# Текстовые утилиты
# ═══════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Очистить текст от лишних пробелов и спецсимволов, сохранить кириллицу."""
    if not text:
        return ""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Оставляем кириллицу, латиницу, цифры, базовую пунктуацию
    text = re.sub(r'[^\x20-\x7E\u0400-\u04FF\u2116\n\r\t]', '', text)
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines).strip()


def text_hash(text: str) -> str:
    """MD5-хеш текста (первые 12 символов)."""
    return hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()[:12]


def normalize_model_key(key: str) -> str:
    """Нормализовать ключ модели КОИБ."""
    key = str(key).strip().lower()
    return key if key in KNOWN_MODELS else "unknown"


def detect_model_in_text(text: str) -> Tuple[str, float]:
    """Определить модель КОИБ по тексту. Возвращает (ключ, уверенность)."""
    if not text or len(text.strip()) < 5:
        return ("unknown", 0.0)
    scores: Dict[str, float] = {}
    for model_key, patterns in KOIB_MODEL_PATTERNS.items():
        match_count = 0
        for pat in patterns:
            if re.findall(pat, text, re.IGNORECASE):
                match_count += 1
        if match_count > 0:
            scores[model_key] = match_count
    if not scores:
        return ("unknown", 0.0)
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    confidence = min(scores[best] / 3.0, 1.0)
    return (best, round(confidence, 3))


def detect_model_from_filename(filename: str) -> str:
    """Определить модель КОИБ по имени файла."""
    fn = filename.lower()
    for model_key, patterns in KOIB_MODEL_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, fn, re.IGNORECASE):
                return model_key
    return "unknown"


def find_figure_caption(text: str) -> str:
    """Найти подпись к рисунку/схеме в тексте."""
    if not text:
        return ""
    for pat in FIGURE_CAPTION_PATTERNS:
        match = pat.search(text)
        if match:
            caption = match.group(1).strip()
            if len(caption) > 3:
                return caption
    return ""


def extract_headings(text: str, max_count: int = 5) -> List[str]:
    """Извлечь заголовки разделов из текста."""
    patterns = [
        re.compile(r'^(\d+(?:\.\d+)*)\s+([А-ЯЁA-Z][^\n]{3,80})$', re.MULTILINE),
        re.compile(r'^([А-ЯЁ][А-ЯЁ\s]{4,60})$', re.MULTILINE),
    ]
    headings: List[str] = []
    seen: set = set()
    for pat in patterns:
        for m in pat.finditer(text):
            h = m.group(0).strip()
            if h not in seen and len(h) > 4:
                headings.append(h)
                seen.add(h)
            if len(headings) >= max_count:
                break
    return headings[:max_count]


def estimate_tokens(text: str) -> int:
    """Примерная оценка числа токенов (1 токен ≈ 4 символа для русского)."""
    return max(1, len(text) // 4)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Обрезать текст до примерного числа токенов."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(' ', 1)[0] + "..."
