# -*- coding: utf-8 -*-
"""
Koib-V-4.1 — Модуль парсинга документов
==========================================
Извлечение текста, таблиц, формул и изображений из PDF/DOCX
с сохранением логической структуры и метаданных.

Поддерживаемые движки:
  - pymupdf (базовый, всегда доступен)
  - docling (IBM Docling, лучшее качество структуры)

Каждый извлечённый элемент (DocumentElement) содержит:
  - content: текст / Markdown-таблица / LaTeX-формула / описание рисунка
  - element_type: text | table | formula | figure | heading | list
  - metadata: {source, page, heading, element_type, ...}
"""

import io
import re
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

import fitz  # PyMuPDF
from docx import Document as DocxDocument
from PIL import Image

from .utils import (
    clean_text, text_hash, detect_model_in_text,
    detect_model_from_filename, find_figure_caption,
    extract_headings, estimate_tokens,
)
from config import (
    OCR_DPI, OCR_MIN_TEXT_CHARS, MIN_IMAGE_WIDTH, MIN_IMAGE_HEIGHT,
    PARSING_ENGINE, FIGURES_DIR,
)

logger = logging.getLogger("koib.parsing")


# ═══════════════════════════════════════════════════════════════
# Структура извлечённого элемента
# ═══════════════════════════════════════════════════════════════

@dataclass
class DocumentElement:
    """Один логический элемент документа."""
    content: str                           # Текст / Markdown таблица / LaTeX / описание
    element_type: str                      # text | table | formula | figure | heading | list
    source: str = ""                       # Имя файла
    page: int = 0                          # Номер страницы (0 для DOCX)
    heading: str = ""                      # Заголовок раздела, к которому принадлежит
    model: str = "unknown"                 # Модель КОИБ
    element_id: str = ""                   # Уникальный ID элемента
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.element_id:
            self.element_id = text_hash(
                f"{self.source}:{self.page}:{self.element_type}:{self.content[:200]}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def is_structured(self) -> bool:
        """Является ли элемент структурированным (таблица/формула/рисунок)."""
        return self.element_type in ("table", "formula", "figure")


# ═══════════════════════════════════════════════════════════════
# Вспомогательные функции
# ═══════════════════════════════════════════════════════════════

def _expand_rect(rect: fitz.Rect, margin: float) -> fitz.Rect:
    """Расширить прямоугольник на margin (совместимо со всеми версиями PyMuPDF)."""
    return fitz.Rect(
        rect.x0 - margin, rect.y0 - margin,
        rect.x1 + margin, rect.y1 + margin,
    )


def _is_scanned_page(page: fitz.Page, min_chars: int = 50) -> bool:
    """Определить, является ли страница отсканированной."""
    text = page.get_text("text").strip()
    if len(text) >= min_chars:
        return False
    images = page.get_images(full=True)
    if not images:
        return len(text) < min_chars
    page_area = page.rect.width * page.rect.height
    for img_info in images:
        try:
            xref = img_info[0]
            base_image = page.parent.extract_image(xref)
            if not base_image:
                continue
            img = Image.open(io.BytesIO(base_image["image"]))
            if img.width * img.height / page_area > 0.8:
                return True
        except Exception:
            continue
    return True


def _ocr_image(image_pil: Image.Image, lang: str = 'rus+eng') -> str:
    """OCR изображения через Tesseract (fallback) или EasyOCR."""
    if image_pil is None:
        return ""

    # Попытка 1: Tesseract
    try:
        import pytesseract
        text = clean_text(
            pytesseract.image_to_string(image_pil, lang=lang, config='--psm 6')
        )
        if len(text) >= 30:
            return text
    except Exception as exc:
        logger.debug(f"Tesseract OCR error: {exc}")

    # Попытка 2: EasyOCR
    try:
        import easyocr
        import numpy as np
        reader = easyocr.Reader(['ru', 'en'], gpu=False, verbose=False)
        results = reader.readtext(np.array(image_pil), paragraph=True, detail=0)
        text = clean_text('\n'.join(results))
        if len(text) >= 20:
            return text
    except Exception as exc:
        logger.debug(f"EasyOCR error: {exc}")

    return ""


def _extract_tables_from_page(page: fitz.Page) -> List[Dict[str, Any]]:
    """
    Извлечь таблицы со страницы PDF через PyMuPDF.
    Возвращает список словарей с ключами: rows, markdown, bounding_box.
    """
    tables = []
    try:
        # PyMuPDF >= 1.23.0 имеет метод find_tables()
        tab_finder = page.find_tables()
        for tab in tab_finder:
            rows = tab.extract()
            if not rows or len(rows) < 2:
                continue

            # Формируем Markdown-таблицу
            md_lines = []
            for i, row in enumerate(rows):
                cells = [str(c).strip() if c else "" for c in row]
                md_lines.append("| " + " | ".join(cells) + " |")
                if i == 0:
                    md_lines.append("| " + " | ".join(["---"] * len(cells)) + " |")

            markdown_table = "\n".join(md_lines)
            bbox = tab.bbox if hasattr(tab, 'bbox') else [0, 0, 0, 0]

            tables.append({
                "rows": rows,
                "markdown": markdown_table,
                "bounding_box": bbox,
                "num_rows": len(rows),
                "num_cols": len(rows[0]) if rows else 0,
            })
    except AttributeError:
        # Старая версия PyMuPDF — нет find_tables()
        logger.debug("PyMuPDF < 1.23: find_tables() недоступен")
    except Exception as exc:
        logger.warning(f"Ошибка извлечения таблиц: {exc}")

    return tables


def _detect_formulas_in_text(text: str) -> List[Tuple[str, str]]:
    """
    Обнаружить формулы в тексте. Возвращает список (оригинальный_текст, тип).
    Типы: "latex_inline", "latex_block", "疑似_formula"
    """
    formulas = []

    # LaTeX inline: $...$ или \(...\)
    for m in re.finditer(r'\$([^\$]+)\$', text):
        formulas.append((m.group(0), "latex_inline"))
    for m in re.finditer(r'\\\((.+?)\\\)', text):
        formulas.append((m.group(0), "latex_inline"))

    # LaTeX block: $$...$$ или \[...\]
    for m in re.finditer(r'\$\$(.+?)\$\$', text, re.DOTALL):
        formulas.append((m.group(0), "latex_block"))
    for m in re.finditer(r'\\\[(.+?)\\\]', text, re.DOTALL):
        formulas.append((m.group(0), "latex_block"))

    # Паттерны формул в технических документах (без явной разметки LaTeX)
    # Строки с множеством математических символов
    formula_pattern = re.compile(
        r'^[^\n]*[=+\-*/^]\s*[0-9A-Za-zА-Яа-я\u0430-\u044f]'
        r'[^\n]{0,80}[=+\-*/^=][^\n]*$',
        re.MULTILINE,
    )
    for m in formula_pattern.finditer(text):
        candidate = m.group(0).strip()
        # Исключаем обычные предложения
        if any(op in candidate for op in ['≥', '≤', '≈', '±', '×', '÷', '→', '∈', '∑']):
            formulas.append((candidate, "suspected_formula"))
        elif re.search(r'[A-Za-z]\s*[=_]\s*[0-9]', candidate):
            formulas.append((candidate, "suspected_formula"))

    return formulas


# ═══════════════════════════════════════════════════════════════
# Парсер PDF
# ═══════════════════════════════════════════════════════════════

def parse_pdf(
    pdf_path: Path,
    figures_dir: Optional[Path] = None,
) -> List[DocumentElement]:
    """
    Извлечь структурированные элементы из PDF.

    Для каждой страницы:
      1. Определяем, отсканирована ли — если да, OCR.
      2. Извлекаем таблицы через PyMuPDF find_tables().
      3. Извлекаем изображения/схемы.
      4. Извлекаем текст с заголовками.
      5. Обнаруживаем формулы.
    """
    elements: List[DocumentElement] = []
    figures_dir = figures_dir or FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    model = detect_model_from_filename(pdf_path.name)

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.error(f"Не удалось открыть PDF {pdf_path}: {exc}")
        return elements

    # Предварительно определяем модель по всему тексту
    full_text_sample = ""
    for page in doc:
        full_text_sample += page.get_text("text") + "\n"
    detected_model, confidence = detect_model_in_text(full_text_sample)
    if confidence > 0.3:
        model = detected_model

    current_heading = ""

    for page_num, page in enumerate(doc):
        try:
            # ── OCR для отсканированных страниц ─────────────────
            is_scanned = _is_scanned_page(page)
            page_text = page.get_text("text").strip()

            if is_scanned:
                matrix = fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72)
                pix = page.get_pixmap(matrix=matrix)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = _ocr_image(img)
                if len(ocr_text) >= OCR_MIN_TEXT_CHARS:
                    page_text = ocr_text

            # ── Извлечение заголовков ────────────────────────────
            headings = extract_headings(page_text)
            if headings:
                current_heading = headings[0]

            # ── Извлечение таблиц ────────────────────────────────
            tables = _extract_tables_from_page(page)
            table_regions = set()
            for i, tab in enumerate(tables):
                bbox = tab["bounding_box"]
                # Запоминаем регионы таблиц, чтобы исключить их из текста
                table_regions.add((round(bbox[0]), round(bbox[1]), round(bbox[2]), round(bbox[3])))

                elements.append(DocumentElement(
                    content=tab["markdown"],
                    element_type="table",
                    source=pdf_path.name,
                    page=page_num + 1,
                    heading=current_heading,
                    model=model,
                    metadata={
                        "num_rows": tab["num_rows"],
                        "num_cols": tab["num_cols"],
                        "table_index": i,
                    },
                ))

            # ── Извлечение изображений/схем ──────────────────────
            for img_info in page.get_images(full=True):
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    if not base_image:
                        continue
                    img_bytes = base_image["image"]
                    img = Image.open(io.BytesIO(img_bytes))
                    if img.width < MIN_IMAGE_WIDTH or img.height < MIN_IMAGE_HEIGHT:
                        continue

                    # Сохраняем изображение
                    img_hash = hashlib.md5(img_bytes).hexdigest()[:12]
                    ext = base_image.get('ext', 'png')
                    img_fname = f"fig_{pdf_path.stem}_p{page_num+1}_{img_hash}.{ext}"
                    img_path = figures_dir / img_fname
                    try:
                        img.save(img_path)
                    except Exception:
                        img_fname = f"fig_{pdf_path.stem}_p{page_num+1}_{img_hash}.png"
                        img_path = figures_dir / img_fname
                        img.convert("RGB").save(img_path)

                    # Подпись и контекст
                    raw_rect = fitz.Rect(img_info[1:5])
                    clip_rect = _expand_rect(raw_rect, 50)
                    nearby_text = page.get_text("text", clip=clip_rect)
                    caption = find_figure_caption(nearby_text)

                    figure_desc = caption if caption else "[Рисунок: описание отсутствует]"

                    elements.append(DocumentElement(
                        content=figure_desc,
                        element_type="figure",
                        source=pdf_path.name,
                        page=page_num + 1,
                        heading=current_heading,
                        model=model,
                        metadata={
                            "image_path": str(img_path),
                            "caption": caption,
                            "surrounding_text": clean_text(nearby_text)[:300],
                            "width": img.width,
                            "height": img.height,
                        },
                    ))
                except Exception as exc:
                    logger.debug(f"Пропуск изображения стр.{page_num+1}: {exc}")

            # ── Текст (исключая регионы таблиц) ──────────────────
            if page_text:
                # Простой подход: если таблицы найдены, добавляем текст целиком,
                # но помечаем как "text". В идеале нужно вырезать bbox таблиц,
                # но PyMuPDF даёт это через textdict.
                formulas = _detect_formulas_in_text(page_text)

                # Разделяем текст на абзацы
                paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
                para_idx = 0
                for para in paragraphs:
                    if len(para) < 10:
                        continue

                    # Проверяем, является ли абзац заголовком
                    is_heading = False
                    for h in headings:
                        if para.startswith(h[:30]):
                            is_heading = True
                            current_heading = h
                            break

                    # Проверяем, содержит ли абзац формулу
                    para_formulas = _detect_formulas_in_text(para)
                    if para_formulas:
                        # Формула — отдельный элемент
                        for formula_text, formula_type in para_formulas:
                            if formula_type in ("latex_inline", "latex_block"):
                                elements.append(DocumentElement(
                                    content=formula_text,
                                    element_type="formula",
                                    source=pdf_path.name,
                                    page=page_num + 1,
                                    heading=current_heading,
                                    model=model,
                                    metadata={"formula_type": formula_type},
                                ))
                            else:
                                # Подозреваемая формула — сохраняем как текст с пометкой
                                elements.append(DocumentElement(
                                    content=f"[formula] {para}",
                                    element_type="formula",
                                    source=pdf_path.name,
                                    page=page_num + 1,
                                    heading=current_heading,
                                    model=model,
                                    metadata={"formula_type": formula_type},
                                ))

                    if is_heading:
                        elements.append(DocumentElement(
                            content=para,
                            element_type="heading",
                            source=pdf_path.name,
                            page=page_num + 1,
                            heading=current_heading,
                            model=model,
                        ))
                    else:
                        # Обычный текстовый абзац
                        elements.append(DocumentElement(
                            content=para,
                            element_type="text",
                            source=pdf_path.name,
                            page=page_num + 1,
                            heading=current_heading,
                            model=model,
                        ))

        except Exception as exc:
            logger.warning(f"Ошибка стр.{page_num+1} в {pdf_path.name}: {exc}")
            continue

    doc.close()
    logger.info(f"PDF {pdf_path.name}: извлечено {len(elements)} элементов")
    return elements


# ═══════════════════════════════════════════════════════════════
# Парсер DOCX
# ═══════════════════════════════════════════════════════════════

def parse_docx(
    docx_path: Path,
    figures_dir: Optional[Path] = None,
) -> List[DocumentElement]:
    """
    Извлечь структурированные элементы из DOCX.

    python-docx даёт доступ к параграфам (со стилями) и таблицам,
    что позволяет лучше сохранять структуру, чем простой текстовый экспорт.
    """
    elements: List[DocumentElement] = []
    figures_dir = figures_dir or FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    model = detect_model_from_filename(docx_path.name)

    try:
        doc = DocxDocument(docx_path)
    except Exception as exc:
        logger.error(f"Не удалось открыть DOCX {docx_path}: {exc}")
        return elements

    # Определяем модель по тексту
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    detected_model, confidence = detect_model_in_text(full_text)
    if confidence > 0.3:
        model = detected_model

    current_heading = ""

    # ── Обработка тела документа (параграфы и таблицы в порядке) ──
    for element in doc.element.body:
        tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag

        if tag == 'p':
            # Параграф
            para_text = element.text or ""
            # Полный текст параграфа (включая runs)
            from docx.oxml.ns import qn
            runs = element.findall('.//' + qn('w:t'))
            para_text = ''.join(r.text or '' for r in runs).strip()

            if not para_text:
                continue

            # Определяем стиль
            pPr = element.find(qn('w:pPr'))
            style_name = ""
            if pPr is not None:
                pStyle = pPr.find(qn('w:pStyle'))
                if pStyle is not None:
                    style_name = pStyle.get(qn('w:val'), '')

            is_heading = 'Heading' in style_name or 'heading' in style_name.lower()

            if is_heading:
                current_heading = para_text
                elements.append(DocumentElement(
                    content=para_text,
                    element_type="heading",
                    source=docx_path.name,
                    page=0,
                    heading=current_heading,
                    model=model,
                    metadata={"style": style_name},
                ))
            else:
                # Проверяем формулы
                formulas = _detect_formulas_in_text(para_text)
                if formulas:
                    for formula_text, formula_type in formulas:
                        elements.append(DocumentElement(
                            content=formula_text,
                            element_type="formula",
                            source=docx_path.name,
                            page=0,
                            heading=current_heading,
                            model=model,
                            metadata={"formula_type": formula_type},
                        ))

                elements.append(DocumentElement(
                    content=para_text,
                    element_type="text",
                    source=docx_path.name,
                    page=0,
                    heading=current_heading,
                    model=model,
                    metadata={"style": style_name},
                ))

        elif tag == 'tbl':
            # Таблица
            rows = []
            for tr in element.findall('.//' + qn('w:tr')):
                row = []
                for tc in tr.findall(qn('w:tc')):
                    cell_text = ' '.join(
                        t.text or ''
                        for t in tc.findall('.//' + qn('w:t'))
                    ).strip()
                    row.append(cell_text)
                rows.append(row)

            if rows:
                # Формируем Markdown-таблицу
                md_lines = []
                for i, row in enumerate(rows):
                    cells = [str(c).strip() if c else "" for c in row]
                    md_lines.append("| " + " | ".join(cells) + " |")
                    if i == 0:
                        md_lines.append("| " + " | ".join(["---"] * len(cells)) + " |")

                markdown_table = "\n".join(md_lines)

                elements.append(DocumentElement(
                    content=markdown_table,
                    element_type="table",
                    source=docx_path.name,
                    page=0,
                    heading=current_heading,
                    model=model,
                    metadata={
                        "num_rows": len(rows),
                        "num_cols": len(rows[0]) if rows else 0,
                    },
                ))

    # ── Извлечение изображений ────────────────────────────────────
    for rel in doc.part.rels.values():
        if "image" not in rel.target_ref:
            continue
        try:
            img_bytes = rel.target_part.blob
            img = Image.open(io.BytesIO(img_bytes))
            if img.width < MIN_IMAGE_WIDTH or img.height < MIN_IMAGE_HEIGHT:
                continue
            img_hash = hashlib.md5(img_bytes).hexdigest()[:12]
            ext = rel.target_ref.rsplit('.', 1)[-1] if '.' in rel.target_ref else 'png'
            img_fname = f"fig_{docx_path.stem}_{img_hash}.{ext}"
            img_path = figures_dir / img_fname
            try:
                img.save(img_path)
            except Exception:
                img_fname = f"fig_{docx_path.stem}_{img_hash}.png"
                img_path = figures_dir / img_fname
                img.convert("RGB").save(img_path)

            elements.append(DocumentElement(
                content="[Рисунок: описание отсутствует]",
                element_type="figure",
                source=docx_path.name,
                page=0,
                heading=current_heading,
                model=model,
                metadata={
                    "image_path": str(img_path),
                    "caption": "",
                    "width": img.width,
                    "height": img.height,
                },
            ))
        except Exception as exc:
            logger.debug(f"Пропуск изображения {docx_path.name}: {exc}")

    logger.info(f"DOCX {docx_path.name}: извлечено {len(elements)} элементов")
    return elements


# ═══════════════════════════════════════════════════════════════
# Docling-парсер (опционально, лучшее качество)
# ═══════════════════════════════════════════════════════════════

def parse_with_docling(file_path: Path) -> List[DocumentElement]:
    """
    Парсинг через IBM Docling — продвинутый движок с определением
    структуры документа, таблиц и формул.

    Требует: pip install docling
    """
    elements: List[DocumentElement] = []
    model = detect_model_from_filename(file_path.name)

    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(file_path))
        doc = result.document

        for item, _ in doc.iterate_items():
            if hasattr(item, 'text') and item.text:
                content = item.text.strip()
                if not content:
                    continue

                # Определяем тип элемента
                el_type = "text"
                heading = ""
                if hasattr(item, 'label'):
                    label = str(item.label).lower()
                    if 'heading' in label or 'title' in label:
                        el_type = "heading"
                        heading = content
                    elif 'table' in label:
                        el_type = "table"
                        # Docling экспортирует таблицы в Markdown
                        if hasattr(item, 'export_to_markdown'):
                            content = item.export_to_markdown()
                    elif 'formula' in label or 'equation' in label:
                        el_type = "formula"
                    elif 'figure' in label or 'caption' in label:
                        el_type = "figure"
                    elif 'list' in label:
                        el_type = "list"

                page_num = 0
                if hasattr(item, 'prov') and item.prov:
                    prov = item.prov[0] if isinstance(item.prov, list) else item.prov
                    if hasattr(prov, 'page_no'):
                        page_num = prov.page_no

                elements.append(DocumentElement(
                    content=content,
                    element_type=el_type,
                    source=file_path.name,
                    page=page_num,
                    heading=heading,
                    model=model,
                ))
    except ImportError:
        logger.warning("Docling не установлен. Используйте: pip install docling")
    except Exception as exc:
        logger.error(f"Ошибка Docling для {file_path.name}: {exc}")

    # Определяем модель по всему тексту
    if elements:
        full_text = " ".join(e.content for e in elements[:20])
        detected_model, confidence = detect_model_in_text(full_text)
        if confidence > 0.3:
            for e in elements:
                e.model = detected_model

    logger.info(f"Docling {file_path.name}: извлечено {len(elements)} элементов")
    return elements


# ═══════════════════════════════════════════════════════════════
# Единая точка входа
# ═══════════════════════════════════════════════════════════════

def parse_document(
    file_path: Path,
    figures_dir: Optional[Path] = None,
    engine: Optional[str] = None,
) -> List[DocumentElement]:
    """
    Извлечь структурированные элементы из документа.

    Args:
        file_path:  Путь к файлу (PDF или DOCX)
        figures_dir: Директория для сохранения изображений
        engine:     Парсер — "pymupdf" | "docling" (по умолчанию из конфига)

    Returns:
        Список DocumentElement
    """
    engine = engine or PARSING_ENGINE
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    logger.info(f"Парсинг {file_path.name} (движок: {engine})")

    if engine == "docling":
        return parse_with_docling(file_path)

    if suffix == '.pdf':
        return parse_pdf(file_path, figures_dir)
    elif suffix == '.docx':
        return parse_docx(file_path, figures_dir)
    else:
        logger.warning(f"Неподдерживаемый формат: {suffix}")
        return []
