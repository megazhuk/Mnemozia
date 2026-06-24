# Mnemozia

> 🇬🇧 [English version](README.md)

**Семантическая база знаний для [Hermes Agent](https://github.com/NousResearch/hermes-agent).**

Названа в честь Мнемозины (Μνημοσύνη) — греческой богини памяти. Хранит факты, находит их
по смыслу (а не по точным ключевым словам) и отслеживает эволюцию знаний с полной историей версий.

**Стек:**
- **Хранилище:** [pg0](https://github.com/vectorize-io/pg0) (PostgreSQL 18 + pgvector)
- **Эмбеддинги:** [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) через [llama.cpp](https://github.com/ggml-org/llama.cpp) — 1024-dim, 100+ языков, 4-bit квантование (378 МБ)
- **Инференс:** `llama-server` как systemd-сервис, HTTP API — без SIGILL на старых CPU

Создана при помощи [Hermes Agent](https://github.com/NousResearch/hermes-agent) на модели DeepSeek.

## Возможности

- **14 операций:** add, search, update, merge, split, deactivate, reactivate, history, review, stats, export, relate, unrelate, vacuum
- **3-этапный дедупликатор:** точное совпадение → семантический near-duplicate → флаг противоречия — дубликаты не проскальзывают
- **Семантический поиск:** векторный поиск через Qwen3-Embedding-0.6B — находит по смыслу на 100+ языках
- **Полное версионирование:** каждое обновление создаёт новую версию; `history` показывает всю цепочку изменений
- **Слияние и разделение:** объединяй пересекающиеся факты или разбивай сложные на атомарные
- **Оценка достоверности:** 0.0 (гипотеза) → 1.0 (проверенный факт) — агент знает, когда доверять
- **Вывод, оптимизированный для LLM:** результаты поиска включают дистанции, флаги достоверности, подсказки к слиянию
- **Эмбеддинги через systemd:** `llama-server` работает как отдельный демон — ноль RAM в Python-процессе Hermes
- **Нативная интеграция с Hermes:** установка как плагин (git clone в `~/.hermes/plugins/`) или standalone

## Установка

```bash
# Вариант А: Hermes-плагин (рекомендуется)
git clone https://github.com/megazhuk/Mnemozia.git ~/.hermes/plugins/mnemozia
pip install -r ~/.hermes/plugins/mnemozia/requirements.txt
hermes tools enable mnemozia
# Затем введите /reset в чате Hermes, или перезапустите Hermes

# Вариант Б: Standalone Python (установка из git)
pip install git+https://github.com/megazhuk/Mnemozia.git
```

## Быстрый старт

```python
from mnemozia import MnemoziaKB

kb = MnemoziaKB()  # по умолчанию postgresql://postgres:postgres@127.0.0.1:5432/postgres
# или: MnemoziaKB("postgresql://user:pass@host:5432/db")

# Сохраняем факты
kb.execute({"action": "add", "text": "OpenRouter требует socks5h-прокси 199.68.196.14:31149", "category": "devops/networking", "tags": "openrouter,proxy,socks5", "confidence": 1.0})

# Ищем по смыслу
kb.execute({"action": "search", "query": "как подключиться к OpenRouter", "limit": 3})

# Обновляем с сохранением истории
kb.execute({"action": "update", "id": "a1b2c3d4e5f6", "text": "OpenRouter прокси: socks5h://199.68.196.14:31149 (схема обязательна)", "confidence": 1.0})

# Объединяем дубликаты
kb.execute({"action": "merge", "id": "a1b2c3", "with": "d4e5f6", "text": "объединённый текст"})
```

## Операции

| Действие | Описание |
|----------|----------|
| `add` | Сохранить факт (авто-дедупликация: точное → семантическое → флаг противоречия) |
| `search` | Семантический векторный поиск (pgvector: semantic, keyword и hybrid) |
| `update` | Новая версия существующего факта (история сохраняется) |
| `merge` | Объединить два факта в один (оригиналы архивируются) |
| `split` | Разбить сложный факт на атомарные части |
| `deactivate` | Мягкое удаление (архивируется, можно восстановить) |
| `reactivate` | Восстановить архивированный факт |
| `history` | Полная история версий факта |
| `review` | Факты, требующие внимания (низкая достоверность, устаревшие) |
| `stats` | Статистика: всего, по категориям, распределение достоверности |
| `export` | Экспорт в Markdown или JSON |
| `relate` | Связать два факта |
| `unrelate` | Убрать связь между фактами |
| `vacuum` | Полное удаление старых архивированных записей |

## Категории

`general`, `work`, `personal`, `finance`, `credentials`, `ideas`, `tech`, `devops`, `programming`, `schedule`, `contacts`, `health`, `travel`, `learning` — плюс иерархические через `/` (например `devops/networking/proxy`).

## Уровни достоверности

| Уровень | Значение |
|---------|----------|
| 1.0 | Проверенный факт |
| 0.7–0.9 | Надёжный |
| 0.5–0.7 | Вероятный |
| 0.3–0.5 | Гипотеза |
| 0.0–0.3 | Предположение |

## Лицензия

MIT
