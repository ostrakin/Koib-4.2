# -*- coding: utf-8 -*-
"""
Koib-V-4.1 — Модуль оценки качества RAG
==========================================
Автоматическая оценка с помощью LLM-as-Judge (GigaChat/OpenAI)
или библиотек RAGAS / DeepEval.

Метрики:
  1. Faithfulness       — верность ответа контексту (0–1)
  2. Answer Relevancy   — релевантность ответа вопросу (0–1)
  3. Context Precision  — точность: доля полезных чанков (0–1)
  4. Context Recall     — полнота: покрытие нужных фактов контекстом (0–1)
  5. Token F1           — токен-совпадение с эталоном (если задан)
"""

import json
import re
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

from config import GIGACHAT_CREDENTIALS, LLM_PROVIDER, METADATA_DIR

logger = logging.getLogger("koib.evaluation")


# ═══════════════════════════════════════════════════════════════
# Структура результата оценки
# ═══════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    """Результат оценки одного вопроса."""
    question_id: str
    question: str
    category: str = ""
    koib_model: str = ""
    answer: str = ""
    reference_answer: str = ""
    context_chunks: int = 0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    token_f1: float = 0.0
    has_reference: bool = False
    error: Optional[str] = None
    latency_sec: float = 0.0

    @property
    def rag_score(self) -> float:
        """Итоговый RAG-score: среднее 4 LLM-метрик."""
        vals = [self.faithfulness, self.answer_relevancy,
                self.context_precision, self.context_recall]
        return round(sum(vals) / len(vals), 3) if vals else 0.0


# ═══════════════════════════════════════════════════════════════
# Промпты для LLM-судьи
# ═══════════════════════════════════════════════════════════════

PROMPT_FAITHFULNESS = """Ты — строгий судья качества AI-ответов. Оцени ВЕРНОСТЬ ответа относительно контекста.

ВОПРОС: {question}

КОНТЕКСТ (извлечённые фрагменты документации):
{context}

ОТВЕТ СИСТЕМЫ:
{answer}

Критерий — Faithfulness (Верность):
Содержит ли ответ ТОЛЬКО информацию из контекста? Нет ли в нём домыслов?

Оцени по шкале от 0 до 10:
  10 — ответ полностью основан на контексте
   5 — частично из контекста, частично домыслы
   0 — ответ полностью придуман

Ответь ТОЛЬКО одним числом от 0 до 10."""

PROMPT_ANSWER_RELEVANCY = """Ты — строгий судья качества AI-ответов. Оцени РЕЛЕВАНТНОСТЬ ответа вопросу.

ВОПРОС: {question}

ОТВЕТ СИСТЕМЫ:
{answer}

Критерий — Answer Relevancy (Релевантность):
Отвечает ли ответ напрямую на поставленный вопрос?

Оцени по шкале от 0 до 10:
  10 — ответ точно и полно отвечает на вопрос
   5 — частично отвечает, много лишнего
   0 — ответ не по теме

Ответь ТОЛЬКО одним числом от 0 до 10."""

PROMPT_CONTEXT_PRECISION = """Ты — строгий судья качества AI-ответов. Оцени ТОЧНОСТЬ найденного контекста.

ВОПРОС: {question}

НАЙДЕННЫЕ ФРАГМЕНТЫ ДОКУМЕНТАЦИИ:
{context}

Критерий — Context Precision (Точность контекста):
Какая доля фрагментов действительно нужна для ответа на вопрос?

Оцени по шкале от 0 до 10:
  10 — все фрагменты релевантны
   5 — примерно половина по теме
   0 — все нерелевантны

Ответь ТОЛЬКО одним числом от 0 до 10."""

PROMPT_CONTEXT_RECALL = """Ты — строгий судья качества AI-ответов. Оцени ПОЛНОТУ найденного контекста.

ВОПРОС: {question}

ЭТАЛОННЫЙ ОТВЕТ: {reference}

НАЙДЕННЫЕ ФРАГМЕНТЫ ДОКУМЕНТАЦИИ:
{context}

Критерий — Context Recall (Полнота контекста):
Содержит ли найденный контекст достаточно информации для полного ответа?

Оцени по шкале от 0 до 10:
  10 — контекст содержит всё необходимое
   5 — контекст содержит часть нужной информации
   0 — контекст совсем не помогает

Ответь ТОЛЬКО одним числом от 0 до 10."""


# ═══════════════════════════════════════════════════════════════
# Утилиты
# ═══════════════════════════════════════════════════════════════

def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text


def token_f1(prediction: str, reference: str) -> float:
    """F1 по токенам (без LLM)."""
    pred_tokens = set(_normalize_text(prediction).split())
    ref_tokens = set(_normalize_text(reference).split())
    if not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(ref_tokens)
    return round(2 * precision * recall / (precision + recall), 3)


def _extract_score(text: str) -> float:
    """Извлечь число 0–10 из ответа LLM и нормировать до 0–1."""
    nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)
    for n in nums:
        val = float(n)
        if 0 <= val <= 10:
            return round(val / 10.0, 3)
    return 0.0


# ═══════════════════════════════════════════════════════════════
# LLM-судья
# ═══════════════════════════════════════════════════════════════

class LLMJudge:
    """LLM-судья для оценки качества RAG."""

    def __init__(self, provider: Optional[str] = None):
        from .generation import LLMClient
        self.llm = LLMClient(provider=provider or LLM_PROVIDER)

    def score(self, prompt: str) -> float:
        """Получить оценку от 0 до 1 через LLM."""
        try:
            response = self.llm.generate(prompt, max_tokens=50)
            return _extract_score(response)
        except Exception as exc:
            logger.warning(f"Ошибка LLM-судьи: {exc}")
            return 0.0


# ═══════════════════════════════════════════════════════════════
# Основной оценщик
# ═══════════════════════════════════════════════════════════════

class RAGEvaluator:
    """Оценщик качества RAG-системы."""

    def __init__(self, judge_provider: Optional[str] = None):
        self.judge = LLMJudge(provider=judge_provider)

    def evaluate_one(
        self,
        question: str,
        answer: str,
        context: str,
        reference: str = "",
        question_id: str = "",
        category: str = "",
        koib_model: str = "",
    ) -> EvalResult:
        """Оценить один вопрос."""
        result = EvalResult(
            question_id=question_id,
            question=question,
            category=category,
            koib_model=koib_model,
            answer=answer,
            reference_answer=reference,
            has_reference=bool(reference),
        )

        logger.info(f"Оценка [{question_id}]: {question[:80]}...")

        # Faithfulness
        result.faithfulness = self.judge.score(
            PROMPT_FAITHFULNESS.format(question=question, context=context, answer=answer)
        )

        # Answer Relevancy
        result.answer_relevancy = self.judge.score(
            PROMPT_ANSWER_RELEVANCY.format(question=question, answer=answer)
        )

        # Context Precision
        result.context_precision = self.judge.score(
            PROMPT_CONTEXT_PRECISION.format(question=question, context=context)
        )

        # Context Recall
        result.context_recall = self.judge.score(
            PROMPT_CONTEXT_RECALL.format(
                question=question,
                reference=reference or "эталонный ответ не задан",
                context=context,
            )
        )

        # Token F1
        if reference:
            result.token_f1 = token_f1(answer, reference)

        logger.info(
            f"  RAG Score: {result.rag_score:.3f} "
            f"(F={result.faithfulness:.2f} AR={result.answer_relevancy:.2f} "
            f"CP={result.context_precision:.2f} CR={result.context_recall:.2f})"
        )

        return result

    def evaluate_dataset(self, dataset: List[Dict[str, Any]], answer_fn) -> List[EvalResult]:
        """
        Оценить набор вопросов.

        Args:
            dataset:   Список словарей с ключами: id, question, category, koib_model, reference_answer
            answer_fn: Функция (question, koib_model) -> {"answer": str, "context": str, ...}
        """
        results = []
        for item in dataset:
            q_id = item.get("id", "")
            question = item["question"]
            model = item.get("koib_model", "")
            ref = item.get("reference_answer", "")

            logger.info(f"\n{'─' * 60}")
            logger.info(f"[{q_id}] {question}")

            try:
                t0 = time.time()
                rag_result = answer_fn(question, model)
                latency = time.time() - t0

                answer = rag_result.get("answer", "")
                context = rag_result.get("context_text", "")

                result = self.evaluate_one(
                    question=question,
                    answer=answer,
                    context=context,
                    reference=ref,
                    question_id=q_id,
                    category=item.get("category", ""),
                    koib_model=model,
                )
                result.latency_sec = round(latency, 2)
                result.context_chunks = rag_result.get("context_chunks", 0)
            except Exception as exc:
                result = EvalResult(
                    question_id=q_id,
                    question=question,
                    category=item.get("category", ""),
                    koib_model=model,
                    error=str(exc),
                )

            results.append(result)
            time.sleep(1)  # Пауза между запросами

        return results


# ═══════════════════════════════════════════════════════════════
# Отчёт
# ═══════════════════════════════════════════════════════════════

def print_report(results: List[EvalResult]) -> None:
    """Вывести отчёт оценки."""
    ok = [r for r in results if r.error is None]
    if not ok:
        print("\nНет успешных результатов.")
        return

    def avg(attr):
        vals = [getattr(r, attr) for r in ok]
        return round(sum(vals) / len(vals), 3)

    print("\n" + "═" * 65)
    print("  ИТОГОВЫЙ ОТЧЁТ КАЧЕСТВА RAG-СИСТЕМЫ")
    print("═" * 65)
    print(f"  Вопросов обработано : {len(ok)}/{len(results)}")
    print(f"  Среднее время ответа: {avg('latency_sec')} сек")
    print()
    print(f"  Faithfulness       : {avg('faithfulness'):.3f}")
    print(f"  Answer Relevancy   : {avg('answer_relevancy'):.3f}")
    print(f"  Context Precision  : {avg('context_precision'):.3f}")
    print(f"  Context Recall     : {avg('context_recall'):.3f}")
    total_rag = round(sum(r.rag_score for r in ok) / len(ok), 3)
    print(f"  {'─' * 40}")
    print(f"  Итоговый RAG Score : {total_rag:.3f}")

    ref_results = [r for r in ok if r.has_reference]
    if ref_results:
        avg_f1 = round(sum(r.token_f1 for r in ref_results) / len(ref_results), 3)
        print(f"  Token F1 (n={len(ref_results)})   : {avg_f1:.3f}")

    print()
    print("  Детализация:")
    print(f"  {'ID':<8} {'RAG':>6} {'F':>6} {'AR':>6} {'CP':>6} {'CR':>6}")
    for r in results:
        if r.error:
            print(f"  {r.question_id:<8} {'ОШИБКА':>6}  {r.error[:40]}")
        else:
            print(
                f"  {r.question_id:<8} {r.rag_score:>6.3f} "
                f"{r.faithfulness:>6.2f} {r.answer_relevancy:>6.2f} "
                f"{r.context_precision:>6.2f} {r.context_recall:>6.2f}"
            )
    print("═" * 65)


def save_report(results: List[EvalResult], path: str) -> None:
    """Сохранить отчёт в JSON."""
    ok = [r for r in results if r.error is None]
    data = {
        "summary": {
            "total": len(results),
            "successful": len(ok),
            "avg_rag_score": round(sum(r.rag_score for r in ok) / len(ok), 3) if ok else 0,
            "avg_faithfulness": round(sum(r.faithfulness for r in ok) / len(ok), 3) if ok else 0,
            "avg_answer_relevancy": round(sum(r.answer_relevancy for r in ok) / len(ok), 3) if ok else 0,
            "avg_context_precision": round(sum(r.context_precision for r in ok) / len(ok), 3) if ok else 0,
            "avg_context_recall": round(sum(r.context_recall for r in ok) / len(ok), 3) if ok else 0,
        },
        "results": [asdict(r) for r in results],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nОтчёт сохранён: {path}")
