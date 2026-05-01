# Koib-V-4.1 — Улучшенная RAG-система для технической документации

Система Retrieval-Augmented Generation, оптимизированная для работы с русскоязычной технической документацией, содержащей **таблицы, формулы (LaTeX) и схемы/диаграммы**.

## Ключевые улучшения относительно оригинала

| Аспект | Оригинал (v4.1) | Улучшенная версия |
|--------|------------------|-------------------|
| Парсинг | PyPDF2 / python-docx — только текст | PyMuPDF с сохранением структуры: таблицы (Markdown), формулы, изображения |
| Чанкинг | Фиксированный RecursiveCharacterTextSplitter | Умное разбиение по типам контента: текст ~800 токенов, таблицы/формулы — целиком |
| Индекс | Единый плоский FAISS | Мультимодальный: текстовый FAISS + Summary-индекс + BM25 + Docstore |
| Поиск | Только векторный | Гибридный (векторный + BM25) + переранжирование + маршрутизация запросов |
| Генерация | Базовый промпт | Жёсткая привязка к контексту, цитирование [Документ: X, стр. Y], HyDE |
| Метрики | Нет встроенной оценки | LLM-as-Judge: Faithfulness, Answer Relevancy, Context Precision/Recall |
| Обработка | По одному файлу | Пакетная обработка + инкрементальное обновление |

## Архитектура

```
Документ (PDF/DOCX)
       │
       ▼
┌──────────────────────┐
│   parsing.py         │  → DocumentElement (text/table/formula/figure)
│   PyMuPDF / Docling  │     + метаданные (source, page, heading, model)
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│   chunking.py        │  → Chunk (text: ~800 ток, table/formula: целиком)
│   SmartChunker       │     + LLM-сводки таблиц для summary-индекса
└──────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│   indexing.py                                        │
│   ┌─────────────┐  ┌──────────────┐  ┌───────────┐  │
│   │ Text FAISS  │  │ Summary FAISS│  │   BM25    │  │
│   │ (текстовые) │  │ (таблицы/    │  │ (ключевой)│  │
│   │             │  │  формулы)    │  │           │  │
│   └─────────────┘  └──────────────┘  └───────────┘  │
│   ┌─────────────┐                                    │
│   │  Docstore   │  (полный контент таблиц/формул)    │
│   └─────────────┘                                    │
└──────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│   retrieval.py                                       │
│   1. Векторный поиск (текст + summary)               │
│   2. BM25 поиск                                      │
│   3. Reciprocal Rank Fusion (объединение)            │
│   4. Cross-Encoder переранжирование                  │
│   5. Маршрутизация (определение intent запроса)      │
└──────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│   generation.py                                      │
│   • Системный промпт с жёсткой привязкой к контексту│
│   • Цитирование: [Документ: имя, стр. X]            │
│   • Воспроизведение таблиц (Markdown) и формул (LaTeX)│
│   • Поддержка: GigaChat / OpenAI / Локальная LLM    │
│   • HyDE: генерация гипотетического ответа для поиска│
└──────────────────────────────────────────────────────┘
```

## Установка

```bash
# 1. Клонируйте репозиторий
git clone https://github.com/ostrakin/Koib-V-4.1.git
cd Koib-V-4.1

# 2. Создайте виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Установите зависимости
pip install -r requirements.txt

# 4. (Опционально) Для Docling-парсера:
pip install docling

# 5. (Опционально) Для OpenAI:
pip install openai langchain-openai
```

## Использование

### Индексация документов

```bash
# Положите PDF/DOCX файлы в data/docs/
# Затем запустите индексацию:
python main.py --ingest

# С указанием директории:
python main.py --ingest --docs-dir /path/to/docs

# С LLM-сводками таблиц (лучшее качество поиска):
python main.py --ingest --llm-summary

# С Docling-парсером (лучшее качество структуры):
python main.py --ingest --parsing-engine docling
```

### Запросы

```bash
# Один запрос:
python main.py --query "Как включить КОИБ-2010?"

# С фильтром модели:
python main.py --query "Требования к электропитанию" --model koib2017a

# С HyDE (гипотетический ответ для лучшего поиска):
python main.py --query "Параметры сканера" --hyde

# Интерактивный режим:
python main.py --interactive
```

### Инкрементальное обновление

```bash
# Добавить один файл без перестройки всего индекса:
python main.py --add-file /path/to/new_document.pdf
```

### Оценка качества

```bash
# По умолчанию используется eval_dataset.json:
python main.py --eval

# С указанием датасета:
python main.py --eval --dataset my_questions.json --output report.json

# Только конкретные вопросы:
python main.py --eval --ids q001 q005 q010
```

## Настройка

Все параметры настраиваются через переменные окружения:

```bash
# Режим LLM
export LLM_PROVIDER=gigachat      # gigachat | openai | local
export EMBEDDING_PROVIDER=local    # local | openai

# GigaChat
export GIGACHAT_CREDENTIALS="your_base64_token"

# OpenAI
export OPENAI_API_KEY="sk-..."

# Эмбеддинги
export LOCAL_EMBEDDING_MODEL="intfloat/multilingual-e5-large"
export OPENAI_EMBEDDING_MODEL="text-embedding-3-small"

# Чанкинг
export TEXT_CHUNK_SIZE=800
export TEXT_CHUNK_OVERLAP=80

# Поиск
export VECTOR_SEARCH_K=20
export BM25_SEARCH_K=20
export FINAL_TOP_K=5
export USE_RERANKER=true
export USE_HYDE=false

# Парсинг
export PARSING_ENGINE=pymupdf      # pymupdf | docling
```

## Структура файлов

```
Koib-V-4.1/
├── main.py                    # Точка входа (--ingest, --query, --interactive, --eval)
├── batch_ingest.py            # Пакетная обработка документов
├── config.py                  # Конфигурация (пути, модели, параметры)
├── gigachat_client.py         # GigaChat API клиент
├── eval_dataset.json          # 20 тестовых вопросов
├── requirements.txt           # Зависимости
├── src/
│   ├── __init__.py
│   ├── parsing.py             # Парсинг PDF/DOCX (таблицы, формулы, изображения)
│   ├── chunking.py            # Умный чанкинг по типам контента
│   ├── indexing.py            # Мультимодальный индекс (FAISS + BM25 + Docstore)
│   ├── retrieval.py           # Гибридный поиск + переранжирование
│   ├── generation.py          # Генерация с цитированием + HyDE
│   ├── evaluation.py          # Оценка качества RAG
│   └── utils.py               # Общие утилиты
├── data/
│   └── docs/                  # Директория для документов
└── output/                    # Индексы, логи, метаданные
    ├── index/                 # FAISS + BM25 индексы
    ├── docstore/              # Полный контент таблиц/формул
    ├── figures/               # Извлечённые изображения
    ├── logs/                  # Логи обработки
    └── metadata/              # Метаданные и чанки
```

## Расширенные возможности

### Docling-парсер

Docling (IBM) обеспечивает значительно лучшее определение структуры документа: точное извлечение таблиц, формул, заголовков и списков. Рекомендуется для сложных документов.

```bash
pip install docling
python main.py --ingest --parsing-engine docling
```

### HyDE (Hypothetical Document Embeddings)

При включении HyDE система генерирует гипотетический ответ на вопрос пользователя и использует его для векторного поиска вместо оригинального запроса. Это улучшает поиск для коротких или абстрактных вопросов.

```bash
python main.py --query "Параметры сканера" --hyde
```

### Переранжирование

По умолчанию включено переранжирование через Cross-Encoder. После получения кандидатов из векторного и BM25 поиска, они пропускаются через модель переранжирования для более точного отбора.

Модель переранжировщика можно изменить:
```bash
export RERANKER_MODEL="BAAI/bge-reranker-large"  # для английского
export RERANKER_MODEL="cointegrated/rubert-tiny2" # для русского (лёгкая)
```

### Локальный режим (без OpenAI/GigaChat)

Система полностью работоспособна локально:
```bash
export LLM_PROVIDER=local      # Ollama / llama-cpp
export EMBEDDING_PROVIDER=local # HuggingFace эмбеддинги
```

Для локальной LLM установите [Ollama](https://ollama.ai/) и запустите модель:
```bash
ollama pull saiga_mistral_7b
```
