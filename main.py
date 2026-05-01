# -*- coding: utf-8 -*-
"""
Koib-V-4.1 — Главная точка входа
===================================
Запуск системы одной командой:

  python main.py --ingest                 # Индексация всех документов
  python main.py --query "Вопрос"         # Поиск и генерация ответа
  python main.py --interactive            # Интерактивный CLI
  python main.py --eval                   # Оценка качества
  python main.py --add-file path/to/file  # Инкрементальное добавление файла

Дополнительные параметры:
  --docs-dir DIR       Директория с документами
  --output-dir DIR     Выходная директория
  --model MODEL        Фильтр модели КОИБ
  --top-k N            Количество результатов поиска
  --hyde               Использовать HyDE
  --llm-summary        Использовать LLM для сводок таблиц
  --parsing-engine ENG Движок парсинга (pymupdf | docling)
"""

import sys
import time
import json
import argparse
import logging
from pathlib import Path

# Добавляем корень проекта в sys.path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import (
    DOCS_DIR, OUTPUT_DIR, FINAL_TOP_K, PARSING_ENGINE, ensure_dirs,
)
from src.utils import logger as koib_logger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("koib.main")


# ═══════════════════════════════════════════════════════════════
# Команда: индексация
# ═══════════════════════════════════════════════════════════════

def cmd_ingest(args) -> None:
    """Запустить индексацию документов."""
    from batch_ingest import BatchIngester

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "  KOIB-V-4.1 — ИНДЕКСАЦИЯ ДОКУМЕНТОВ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"  Документы: {args.docs_dir}")
    print(f"  Выход:     {args.output_dir}")
    print(f"  Парсер:    {args.parsing_engine}")
    print(f"  LLM-сводки: {'да' if args.llm_summary else 'нет'}")

    t0 = time.time()
    ingester = BatchIngester(
        docs_dir=Path(args.docs_dir),
        output_dir=Path(args.output_dir),
        llm_summary=args.llm_summary,
        incremental=args.incremental,
    )
    results = ingester.process_all()
    elapsed = time.time() - t0

    print(f"\nВремя: {elapsed:.1f}с ({elapsed/60:.1f} мин)")
    print("Индексация завершена. Запросы: python main.py --query 'Вопрос'")


# ═══════════════════════════════════════════════════════════════
# Команда: запрос
# ═══════════════════════════════════════════════════════════════

def cmd_query(args) -> None:
    """Выполнить один запрос."""
    from src.retrieval import HybridRetriever
    from src.generation import AnswerGenerator

    print(f"\nВопрос: {args.query}")
    print("─" * 60)

    t0 = time.time()
    generator = AnswerGenerator()
    result = generator.answer(
        query=args.query,
        k=args.top_k,
        model_filter=args.model,
        use_hyde=args.hyde,
    )
    elapsed = time.time() - t0

    # Вывод ответа
    print(f"\nОТВЕТ:\n{result['answer']}")

    # Источники
    if result['sources']:
        print(f"\nИСТОЧНИКИ:")
        for src in result['sources']:
            print(f"  [Документ: {src['document']}, стр. {src['page']}]"
                  + (f" — {src['heading']}" if src['heading'] else ""))

    print(f"\nВремя: {elapsed:.2f}с | Чанков: {len(result.get('results', []))}")


# ═══════════════════════════════════════════════════════════════
# Команда: интерактивный режим
# ═══════════════════════════════════════════════════════════════

def cmd_interactive(args) -> None:
    """Запустить интерактивный CLI."""
    from src.retrieval import HybridRetriever
    from src.generation import AnswerGenerator

    generator = AnswerGenerator()

    print("\n" + "=" * 70)
    print("  ИНТЕРАКТИВНЫЙ РЕЖИМ  |  введите 'q' для выхода")
    print("  Команды: 'model koib2010' — фильтр модели, 'model all' — снять")
    print("           'hyde on/off' — включить/выключить HyDE")
    print("=" * 70)

    model_filter = ""
    use_hyde = args.hyde
    history = []

    while True:
        try:
            print()
            raw = input("Вопрос: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nВыход.")
            break

        if not raw or raw.lower() in ('q', 'quit', 'exit', 'выход'):
            print("До свидания!")
            break

        # Команды
        if raw.lower().startswith("model "):
            arg = raw.split(None, 1)[1].strip().lower()
            if arg == "all":
                model_filter = ""
                print("  Фильтр модели снят")
            else:
                model_filter = arg
                print(f"  Фильтр: {arg}")
            continue

        if raw.lower().startswith("hyde "):
            arg = raw.split(None, 1)[1].strip().lower()
            use_hyde = arg == "on"
            print(f"  HyDE: {'включён' if use_hyde else 'выключен'}")
            continue

        # Запрос
        history.append(raw)
        t0 = time.time()
        result = generator.answer(
            query=raw,
            k=args.top_k,
            model_filter=model_filter,
            use_hyde=use_hyde,
        )
        elapsed = time.time() - t0

        print(f"\n{result['answer']}")

        if result['sources']:
            print(f"\nИсточники:")
            for src in result['sources']:
                print(f"  [Документ: {src['document']}, стр. {src['page']}]")

        print(f"\n  ({elapsed:.2f}с)")

    # Сохранение истории
    if history:
        from config import LOGS_DIR
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOGS_DIR / f"cli_history_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print(f"\nИстория сохранена: {log_path}")


# ═══════════════════════════════════════════════════════════════
# Команда: оценка
# ═══════════════════════════════════════════════════════════════

def cmd_eval(args) -> None:
    """Запустить оценку качества RAG."""
    from src.evaluation import RAGEvaluator, print_report, save_report
    from src.generation import AnswerGenerator

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Файл датасета не найден: {dataset_path}")
        print("Создайте eval_dataset.json или укажите --dataset")
        return

    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)

    if args.ids:
        dataset = [q for q in dataset if q.get("id") in args.ids]

    print(f"\nОценка качества RAG по {len(dataset)} вопросам")

    generator = AnswerGenerator()
    evaluator = RAGEvaluator()

    def answer_fn(question: str, koib_model: str = "") -> Dict:
        result = generator.answer(query=question, k=args.top_k, model_filter=koib_model)
        return {
            "answer": result["answer"],
            "context_text": result.get("context_text", ""),
            "context_chunks": len(result.get("results", [])),
        }

    results = evaluator.evaluate_dataset(dataset, answer_fn)
    print_report(results)

    report_path = args.output or "eval_report.json"
    save_report(results, report_path)


# ═══════════════════════════════════════════════════════════════
# Команда: добавление файла
# ═══════════════════════════════════════════════════════════════

def cmd_add_file(args) -> None:
    """Инкрементально добавить один файл."""
    from batch_ingest import BatchIngester

    file_path = Path(args.add_file)
    if not file_path.exists():
        print(f"Файл не найден: {file_path}")
        return

    print(f"\nДобавление файла: {file_path.name}")
    ingester = BatchIngester(
        docs_dir=Path(args.docs_dir),
        output_dir=Path(args.output_dir),
        incremental=True,
    )
    result = ingester.add_file(file_path)

    if result.status == "success":
        print(f"  OK: {result.num_elements} элементов, {result.num_chunks} чанков")
    else:
        print(f"  ОШИБКА: {result.error}")


# ═══════════════════════════════════════════════════════════════
# Главная функция
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Koib-V-4.1 — Улучшенная RAG-система для технической документации",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Основные команды
    parser.add_argument("--ingest", action="store_true",
                        help="Индексация всех документов из папки")
    parser.add_argument("--query", type=str, default="",
                        help="Задать вопрос системе")
    parser.add_argument("--interactive", action="store_true",
                        help="Интерактивный режим")
    parser.add_argument("--eval", action="store_true",
                        help="Оценка качества RAG")
    parser.add_argument("--add-file", type=str, default="",
                        help="Инкрементально добавить файл")

    # Параметры
    parser.add_argument("--docs-dir", type=str, default=str(DOCS_DIR),
                        help="Директория с документами")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Выходная директория")
    parser.add_argument("--model", type=str, default="",
                        help="Фильтр модели КОИБ")
    parser.add_argument("--top-k", type=int, default=FINAL_TOP_K,
                        help="Количество результатов поиска")
    parser.add_argument("--hyde", action="store_true",
                        help="Использовать HyDE для поиска")
    parser.add_argument("--llm-summary", action="store_true",
                        help="Использовать LLM для генерации сводок таблиц")
    parser.add_argument("--parsing-engine", type=str, default=PARSING_ENGINE,
                        choices=["pymupdf", "docling"],
                        help="Движок парсинга документов")
    parser.add_argument("--incremental", action="store_true",
                        help="Инкрементальное обновление индекса")

    # Параметры оценки
    parser.add_argument("--dataset", type=str, default="eval_dataset.json",
                        help="Файл с вопросами для оценки")
    parser.add_argument("--ids", nargs="*", default=None,
                        help="ID вопросов для оценки (фильтр)")
    parser.add_argument("--output", type=str, default="eval_report.json",
                        help="Файл для сохранения отчёта оценки")

    args = parser.parse_args()

    # Баннер
    print("╔" + "═" * 68 + "╗")
    print("║" + "  KOIB-V-4.1 — RAG для технической документации".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    ensure_dirs()

    # Роутинг команд
    if args.ingest:
        cmd_ingest(args)
    elif args.query:
        cmd_query(args)
    elif args.interactive:
        cmd_interactive(args)
    elif args.eval:
        cmd_eval(args)
    elif args.add_file:
        cmd_add_file(args)
    else:
        parser.print_help()
        print("\nПримеры:")
        print("  python main.py --ingest")
        print('  python main.py --query "Как включить КОИБ-2010?"')
        print("  python main.py --interactive")
        print("  python main.py --eval --dataset eval_dataset.json")


if __name__ == "__main__":
    main()
