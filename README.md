# 🤖 TG AI Agents

Telegram-бот с несколькими специализированными AI-агентами на базе Google Gemini.

## Агенты

| Агент | Что умеет |
|-------|-----------|
| 👨‍💻 Программист | Пишет код, запускает Python, ищет документацию |
| 🔍 Исследователь | Ищет в интернете, читает сайты, сохраняет отчёты |
| ✍️ Писатель | Статьи, посты, письма, рефераты |
| 📊 Аналитик | Анализирует данные, строит логику, делает выводы |
| 🤖 Ассистент | Универсальный помощник |

## Установка

```bash
cd tg-ai-agents
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Настройка

```bash
cp .env.example .env
nano .env  # Вставь токены
```

Нужны два токена:
- **TELEGRAM_BOT_TOKEN** — получить у [@BotFather](https://t.me/BotFather) в Telegram
- **GEMINI_API_KEY** — получить на [aistudio.google.com](https://aistudio.google.com/apikey)

## Запуск

```bash
source venv/bin/activate
python bot.py
```

## Команды в Telegram

| Команда | Действие |
|---------|----------|
| `/start` | Начать работу, показать меню |
| `/agents` | Выбрать агента |
| `/status` | Текущий агент и его инструменты |
| `/reset` | Очистить историю диалога |
| `/help` | Справка |

## Структура проекта

```
tg-ai-agents/
├── bot.py              # Telegram бот
├── requirements.txt
├── .env.example        # Шаблон конфигурации
├── agents/
│   ├── definitions.py  # Описание агентов
│   └── engine.py       # Движок (Gemini + tool calling)
├── tools/
│   └── tools.py        # Инструменты агентов
└── files/              # Файлы, созданные агентами
```

## Инструменты агентов

- **web_search** — поиск через DuckDuckGo
- **fetch_url** — чтение веб-страниц
- **run_python** — выполнение Python кода
- **save_file / read_file** — работа с файлами
- **list_files** — список сохранённых файлов

## Добавление своего агента

В файле `agents/definitions.py` добавь новый агент:

```python
"my_agent": AgentDefinition(
    name="my_agent",
    display_name="Мой агент",
    emoji="🎯",
    description="Описание агента",
    tools=["web_search", "save_file"],
    system_prompt="Ты — ...",
),
```
