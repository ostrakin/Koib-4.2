# -*- coding: utf-8 -*-
"""
Koib-V-4.1 — Пакетная обработка документов
==============================================
Скрипт для массовой индексации документов:
  - Принимает папку с PDF/DOCX (или список файлов)
  - Прогоняет каждый через парсинг + чанкинг + индексацию
  - Ведёт лог обработки с отметкой об ошибках
  - Поддерживает инкрементальное обновление
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from src.parsing import parse_document, DocumentElement
from src.chunking import SmartChunker, Chunk
from src.indexing import IndexBuilder
from config import DOCS_DIR, OUTPUT_DIR, METADATA_DIR, LOGS_DIR, ensure_dirs

logger = logging.getLogger("koib.batch_ingest")


# ═══════════════════════════════════════════════════════════════
# Результат обработки одного файла
# ═══════════════════════════════════════════════════════════════

@dataclass
class FileProcessingResult:
    """Результат обработки одного файла."""
    filename: str
    file_type: str = ""
    status: str = "pending"        # pending | success | error
    num_elements: int = 0
    num_chunks: int = 0
    num_tables: int = 0
    num_formulas: int = 0
    num_figures: int = 0
    error: Optional[str] = None
    time_sec: float = 0.0
    model: str = "unknown"


# ═══════════════════════════════════════════════════════════════
# Пакетный загрузчик
# ═══════════════════════════════════════════════════════════════

class BatchIngester:
    """
    Пакетный обработчик документов.

    Пайплайн для каждого файла:
      1. parse_document() → List[DocumentElement]
      2. SmartChunker.chunk_elements() → List[Chunk]
      3. IndexBuilder.build() / add_chunks() → индексы

    Поддерживает:
      - Полную перестройку индекса
      - Инкрементальное добавление файлов
      - Логирование результатов
    """

    def __init__(
        self,
        docs_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        llm_summary: bool = False,
        incremental: bool = False,
    ):
        self.docs_dir = docs_dir or DOCS_DIR
        self.output_dir = output_dir or OUTPUT_DIR
        self.llm_summary = llm_summary
        self.incremental = incremental

        self.chunker = SmartChunker(llm_summary=llm_summary)
        self.index_builder = IndexBuilder(output_dir=self.output_dir / "index")

        self.results: List[FileProcessingResult] = []
        self.all_elements: List[DocumentElement] = []
        self.all_chunks: List[Chunk] = []

    def discover_files(self) -> List[Path]:
        """Найти все PDF и DOCX файлы в директории."""
        files = []
        if self.docs_dir.exists():
            files.extend(self.docs_dir.glob("*.pdf"))
            files.extend(self.docs_dir.glob("*.PDF"))
            files.extend(self.docs_dir.glob("*.docx"))
            files.extend(self.docs_dir.glob("*.DOCX"))
        # Убираем дубли (регистр)
        seen = set()
        unique = []
        for f in files:
            key = f.resolve()
            if key not in seen:
                seen.add(key)
                unique.append(f)
        return sorted(unique)

    def process_file(self, file_path: Path) -> FileProcessingResult:
        """Обработать один файл."""
        result = FileProcessingResult(
            filename=file_path.name,
            file_type=file_path.suffix.lower().lstrip('.'),
        )

        t0 = time.time()
        logger.info(f"Обработка: {file_path.name}")

        try:
            # 1. Парсинг
            elements = parse_document(file_path)
            result.num_elements = len(elements)
            result.num_tables = sum(1 for e in elements if e.element_type == "table")
            result.num_formulas = sum(1 for e in elements if e.element_type == "formula")
            result.num_figures = sum(1 for e in elements if e.element_type == "figure")
            result.model = elements[0].model if elements else "unknown"

            if not elements:
                result.status = "error"
                result.error = "Не извлечено ни одного элемента"
                return result

            # 2. Чанкинг
            chunks = self.chunker.chunk_elements(elements)
            result.num_chunks = len(chunks)

            self.all_elements.extend(elements)
            self.all_chunks.extend(chunks)

            result.status = "success"
            logger.info(
                f"  OK: {len(elements)} элементов, {len(chunks)} чанков "
                f"(таблиц: {result.num_tables}, формул: {result.num_formulas}, "
                f"рисунков: {result.num_figures})"
            )

        except Exception as exc:
            result.status = "error"
            result.error = str(exc)
            logger.error(f"  ОШИБКА: {exc}")

        result.time_sec = round(time.time() - t0, 2)
        return result

    def process_all(self, file_list: Optional[List[Path]] = None) -> List[FileProcessingResult]:
        """
        Обработать все файлы.

        Args:
            file_list: Список файлов (если None — автообнаружение)
        """
        ensure_dirs()

        files = file_list or self.discover_files()
        if not files:
            logger.warning(f"Файлы не найдены в {self.docs_dir}")
            return []

        logger.info(f"Найдено {len(files)} файлов для обработки")

        for i, fp in enumerate(files, 1):
            logger.info(f"\n[{i}/{len(files)}] {fp.name}")
            result = self.process_file(fp)
            self.results.append(result)

        # Построение индексов
        if self.all_chunks:
            logger.info(f"\nПостроение индексов из {len(self.all_chunks)} чанков...")

            if self.incremental:
                self.index_builder.add_chunks(self.all_chunks)
            else:
                self.index_builder.build(self.all_chunks)

            # Сохраняем снимок чанков
            self.index_builder.save_chunks_snapshot(self.all_chunks)

        # Сохраняем лог и метаданные
        self._save_results()
        self._print_summary()

        return self.results

    def add_file(self, file_path: Path) -> FileProcessingResult:
        """Инкрементально добавить один файл."""
        result = self.process_file(file_path)

        if self.all_chunks:
            self.index_builder.add_chunks(self.all_chunks)
            self.index_builder.save_chunks_snapshot(self.all_chunks)

        return result

    def _save_results(self) -> None:
        """Сохранить лог обработки и метаданные."""
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        METADATA_DIR.mkdir(parents=True, exist_ok=True)

        # Лог обработки
        log_path = LOGS_DIR / "ingest_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in self.results], f, ensure_ascii=False, indent=2)
        logger.info(f"Лог обработки: {log_path}")

        # Метаданные элементов
        elements_path = METADATA_DIR / "elements.json"
        with open(elements_path, 'w', encoding='utf-8') as f:
            json.dump([e.to_dict() for e in self.all_elements], f, ensure_ascii=False, indent=2)
        logger.info(f"Элементы: {elements_path}")

    def _print_summary(self) -> None:
        """Вывести итоговую сводку."""
        total = len(self.results)
        success = sum(1 for r in self.results if r.status == "success")
        errors = sum(1 for r in self.results if r.status == "error")

        print("\n" + "=" * 70)
        print("  ИТОГИ ПАКЕТНОЙ ОБРАБОТКИ")
        print("=" * 70)
        print(f"  Файлов обработано: {total}")
        print(f"  Успешно:           {success}")
        print(f"  С ошибками:        {errors}")
        print(f"  Элементов всего:   {sum(r.num_elements for r in self.results)}")
        print(f"  Чанков всего:      {sum(r.num_chunks for r in self.results)}")
        print(f"  Таблиц:            {sum(r.num_tables for r in self.results)}")
        print(f"  Формул:            {sum(r.num_formulas for r in self.results)}")
        print(f"  Рисунков:          {sum(r.num_figures for r in self.results)}")

        if errors:
            print(f"\n  Файлы с ошибками:")
            for r in self.results:
                if r.status == "error":
                    print(f"    - {r.filename}: {r.error}")

        total_time = sum(r.time_sec for r in self.results)
        print(f"\n  Общее время: {total_time:.1f}с ({total_time/60:.1f} мин)")
        print("=" * 70)
