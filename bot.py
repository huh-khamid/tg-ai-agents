"""
Telegram бот с webhook-режимом для деплоя на Render (бесплатный план).
Запуск: python bot.py
"""

import asyncio
import logging
import os
import aiohttp
from aiohttp import web as aiohttp_web

from dotenv import load_dotenv
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ParseMode, ChatAction

from agents.definitions import AGENTS
from agents.engine import AgentManager

load_dotenv()
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─── Конфигурация ─────────────────────────────────────────────────────────────

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324:free")
ALLOWED_USERS_RAW = os.getenv("ALLOWED_USERS", "")
ALLOWED_USERS = set(
    int(u.strip()) for u in ALLOWED_USERS_RAW.split(",") if u.strip().isdigit()
)

# Render автоматически даёт переменную PORT и публичный URL
PORT = int(os.getenv("PORT", 8443))
RENDER_URL = os.getenv("RENDER_EXTERNAL_URL", "")  # Render сам заполняет это

# user_id -> AgentManager
_managers: dict[int, AgentManager] = {}
# user_id -> текущий агент
_active_agent: dict[int, str] = {}


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def get_manager(user_id: int) -> AgentManager:
    if user_id not in _managers:
        _managers[user_id] = AgentManager(OPENROUTER_API_KEY, OPENROUTER_MODEL)
    return _managers[user_id]


def is_allowed(user_id: int) -> bool:
    return not ALLOWED_USERS or user_id in ALLOWED_USERS


def agents_keyboard() -> InlineKeyboardMarkup:
    buttons = []
    row = []
    for i, (name, agent) in enumerate(AGENTS.items()):
        row.append(InlineKeyboardButton(
            f"{agent.emoji} {agent.display_name}",
            callback_data=f"agent:{name}"
        ))
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    return InlineKeyboardMarkup(buttons)


# ─── Хендлеры команд ──────────────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not is_allowed(user.id):
        await update.message.reply_text("⛔ Доступ запрещён.")
        return

    text = (
        f"👋 Привет, *{user.first_name}*!\n\n"
        "Я — менеджер AI-агентов. Выбери агента для работы:\n\n"
    )
    for agent in AGENTS.values():
        text += f"{agent.emoji} *{agent.display_name}* — {agent.description}\n"

    text += "\nИли используй /agents чтобы выбрать агента."
    await update.message.reply_text(
        text, parse_mode=ParseMode.MARKDOWN, reply_markup=agents_keyboard()
    )


async def cmd_agents(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    await update.message.reply_text(
        "🤖 *Выбери агента:*",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=agents_keyboard(),
    )


async def cmd_reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        return
    manager = get_manager(user_id)
    active = _active_agent.get(user_id)
    if active:
        manager.reset_agent(active)
        await update.message.reply_text(
            f"🔄 История агента *{AGENTS[active].display_name}* очищена.",
            parse_mode=ParseMode.MARKDOWN,
        )
    else:
        manager.reset_all()
        await update.message.reply_text("🔄 История всех агентов очищена.")


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        return
    active = _active_agent.get(user_id)
    if active and active in AGENTS:
        a = AGENTS[active]
        tools_list = ", ".join(f"`{t}`" for t in a.tools)
        text = (
            f"📍 Текущий агент: *{a.emoji} {a.display_name}*\n"
            f"📝 {a.description}\n"
            f"🔧 Инструменты: {tools_list}"
        )
    else:
        text = "❌ Агент не выбран. Используй /agents"
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    text = (
        "📖 *Команды:*\n\n"
        "/start — Начать работу\n"
        "/agents — Выбрать агента\n"
        "/status — Текущий агент\n"
        "/reset — Очистить историю\n"
        "/help — Эта справка\n\n"
        "💡 *Как пользоваться:*\n"
        "1. Выбери агента через /agents\n"
        "2. Пиши задачи обычным текстом\n"
        "3. Агент сам использует нужные инструменты\n\n"
        "🤖 *Агенты:*\n"
    )
    for a in AGENTS.values():
        text += f"{a.emoji} *{a.display_name}* — {a.description}\n"
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


# ─── Callback: выбор агента ───────────────────────────────────────────────────

async def on_agent_select(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    if not is_allowed(user_id):
        return

    agent_name = query.data.split(":", 1)[1]
    if agent_name not in AGENTS:
        await query.edit_message_text("❌ Агент не найден.")
        return

    _active_agent[user_id] = agent_name
    agent = AGENTS[agent_name]

    await query.edit_message_text(
        f"{agent.emoji} *{agent.display_name}* выбран!\n\n"
        f"_{agent.description}_\n\n"
        f"Пиши задачу, и я приступлю к работе.",
        parse_mode=ParseMode.MARKDOWN,
    )


# ─── Хендлер сообщений ────────────────────────────────────────────────────────

async def on_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        await update.message.reply_text("⛔ Доступ запрещён.")
        return

    active = _active_agent.get(user_id)
    if not active:
        await update.message.reply_text(
            "⚠️ Сначала выбери агента через /agents или кнопку ниже.",
            reply_markup=agents_keyboard(),
        )
        return

    agent = AGENTS[active]
    manager = get_manager(user_id)
    engine = manager.get_engine(active)

    await update.message.chat.send_action(ChatAction.TYPING)

    user_text = update.message.text.strip()
    logger.info(f"User {user_id} -> [{agent.name}]: {user_text[:80]}")

    try:
        answer = await engine.process(user_text)

        header = f"{agent.emoji} *{agent.display_name}:*\n\n"
        full = header + answer

        if len(full) <= 4096:
            await update.message.reply_text(full, parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(header[:-2], parse_mode=ParseMode.MARKDOWN)
            for i in range(0, len(answer), 4000):
                await update.message.reply_text(answer[i:i+4000])

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        await update.message.reply_text(
            f"❌ Ошибка агента: `{str(e)[:200]}`\n\nПопробуй снова или /reset",
            parse_mode=ParseMode.MARKDOWN,
        )


# ─── Регистрация команд ───────────────────────────────────────────────────────

async def post_init(app: Application):
    await app.bot.set_my_commands([
        BotCommand("start", "Начать работу"),
        BotCommand("agents", "Выбрать агента"),
        BotCommand("status", "Текущий агент"),
        BotCommand("reset", "Очистить историю"),
        BotCommand("help", "Справка"),
    ])


# ─── Запуск ───────────────────────────────────────────────────────────────────

def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в .env")
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY не задан в .env")

    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("agents", cmd_agents))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CallbackQueryHandler(on_agent_select, pattern=r"^agent:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    if RENDER_URL:
        # ── Webhook режим (Render, production) ──
        webhook_url = f"{RENDER_URL}/{TELEGRAM_TOKEN}"
        logger.info(f"🚀 Запуск в webhook-режиме на порту {PORT}")

        # Keepalive: пингуем себя каждые 10 минут чтобы не засыпать
        async def keepalive():
            await asyncio.sleep(60)  # ждём старта
            async with aiohttp.ClientSession() as session:
                while True:
                    try:
                        await session.get(f"{RENDER_URL}/health", timeout=aiohttp.ClientTimeout(total=10))
                        logger.info("💓 Keepalive ping sent")
                    except Exception:
                        pass
                    await asyncio.sleep(600)  # каждые 10 минут

        # /health endpoint для ping-а
        async def health_handler(request):
            return aiohttp_web.Response(text="OK")

        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=TELEGRAM_TOKEN,
            webhook_url=webhook_url,
        )
    else:
        # ── Polling режим (локальная разработка) ──
        logger.info("🚀 Запуск в polling-режиме (локально)")
        app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
