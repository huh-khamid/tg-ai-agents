"""
Движок агента: вызывает Groq API напрямую через aiohttp (без SDK).
При превышении лимита автоматически переключается на следующую модель.
"""

import asyncio
import json
import logging
from typing import Optional

import aiohttp

from agents.definitions import AgentDefinition, AGENTS
from tools.tools import TOOLS_REGISTRY

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ─── Цепочка фоллбэк-моделей ─────────────────────────────────────────────────

FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",  # 1. Llama 3.3 70B (основная)
    "llama3-70b-8192",          # 2. Llama 3 70B (стабильная)
    "gemma2-9b-it",             # 3. Gemma 2 9B
    "llama-3.1-8b-instant",     # 4. Llama 8B (быстрая)
    "llama3-8b-8192",           # 5. Llama 3 8B (резервная)
]

DEFAULT_MODEL = FALLBACK_MODELS[0]


# ─── Схемы инструментов ───────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Поиск информации в интернете через DuckDuckGo",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Поисковый запрос"},
                    "max_results": {"type": "integer", "description": "Количество результатов (1-10)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Загрузить и прочитать содержимое веб-страницы по URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL страницы"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Выполнить Python-код и получить результат",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python код для выполнения"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_file",
            "description": "Сохранить текст в файл",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Имя файла"},
                    "content": {"type": "string", "description": "Содержимое файла"},
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Прочитать содержимое файла",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Имя файла"},
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Показать список всех сохранённых файлов",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


def get_tools_for_agent(agent_tools: list[str]) -> list[dict]:
    return [t for t in TOOL_SCHEMAS if t["function"]["name"] in agent_tools]


# ─── Движок агента ────────────────────────────────────────────────────────────

class AgentEngine:
    """Движок одного агента: управляет диалогом и вызовами инструментов."""

    def __init__(self, definition: AgentDefinition, api_key: str, model: str = DEFAULT_MODEL):
        self.definition = definition
        self.api_key = api_key
        self.tools = get_tools_for_agent(definition.tools)
        self.history: list[dict] = [
            {"role": "system", "content": definition.system_prompt}
        ]
        self._max_tool_calls = 8

        if model in FALLBACK_MODELS:
            start = FALLBACK_MODELS.index(model)
            self._model_chain = FALLBACK_MODELS[start:] + FALLBACK_MODELS[:start]
        else:
            self._model_chain = [model] + FALLBACK_MODELS

        self._model_index = 0

    @property
    def current_model(self) -> str:
        return self._model_chain[self._model_index]

    def _next_model(self) -> Optional[str]:
        self._model_index += 1
        if self._model_index >= len(self._model_chain):
            self._model_index = len(self._model_chain) - 1
            return None
        return self._model_chain[self._model_index]

    async def _call_tool(self, name: str, args: dict) -> str:
        fn = TOOLS_REGISTRY.get(name)
        if fn is None:
            return f"❌ Инструмент `{name}` не найден"
        try:
            if asyncio.iscoroutinefunction(fn):
                return str(await fn(**args))
            else:
                return str(fn(**args))
        except Exception as e:
            return f"❌ Ошибка инструмента {name}: {e}"

    async def _groq_request(self, model: str) -> dict:
        """Делает запрос к Groq API через aiohttp."""
        payload = {
            "model": model,
            "messages": self.history,
        }
        if self.tools:
            payload["tools"] = self.tools
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                GROQ_API_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    status = resp.status
                    msg = data.get("error", {}).get("message", str(data))
                    raise RuntimeError(f"HTTP {status}: {msg}")
                return data

    async def _run_with_tools(self, model: str) -> str:
        """Один цикл запрос → инструменты → ответ."""
        for _ in range(self._max_tool_calls):
            data = await self._groq_request(model)
            choice = data["choices"][0]
            msg = choice["message"]

            # Добавляем в историю
            self.history.append(msg)

            # Финальный ответ
            if not msg.get("tool_calls"):
                return msg.get("content") or "🤔 Агент не дал ответа."

            # Выполняем инструменты
            for tool_call in msg["tool_calls"]:
                fn_name = tool_call["function"]["name"]
                try:
                    fn_args = json.loads(tool_call["function"]["arguments"])
                except (json.JSONDecodeError, KeyError):
                    fn_args = {}

                result = await self._call_tool(fn_name, fn_args)
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result,
                })

        return "⚠️ Агент достиг лимита вызовов инструментов."

    async def process(self, user_message: str) -> tuple[str, str]:
        """Обрабатывает сообщение, возвращает (ответ, модель)."""
        self.history.append({"role": "user", "content": user_message})
        history_checkpoint = len(self.history) - 1

        for _ in range(len(self._model_chain)):
            model = self.current_model
            try:
                result = await self._run_with_tools(model)
                return result, model

            except RuntimeError as e:
                err = str(e)
                # Лимит или перегрузка — переключаемся
                if any(code in err for code in ("HTTP 400", "HTTP 429", "HTTP 503", "HTTP 529", "HTTP 404", "rate_limit", "decommissioned")):
                    logger.warning(f"⚠️ Модель {model} недоступна: {err[:80]}, переключаюсь...")
                    next_m = self._next_model()
                    if next_m is None:
                        return "❌ Все модели недоступны. Попробуй позже.", model
                    self.history = self.history[:history_checkpoint + 1]
                else:
                    return f"❌ Ошибка: {err[:300]}", model

        return "❌ Не удалось получить ответ.", self.current_model

    def reset(self):
        self.history = [{"role": "system", "content": self.definition.system_prompt}]
        self._model_index = 0


# ─── Менеджер агентов ─────────────────────────────────────────────────────────

class AgentManager:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model = model
        self._engines: dict[str, AgentEngine] = {}

    def get_engine(self, agent_name: str) -> Optional[AgentEngine]:
        if agent_name not in AGENTS:
            return None
        if agent_name not in self._engines:
            self._engines[agent_name] = AgentEngine(
                AGENTS[agent_name], self.api_key, self.model
            )
        return self._engines[agent_name]

    def reset_agent(self, agent_name: str):
        if agent_name in self._engines:
            self._engines[agent_name].reset()

    def reset_all(self):
        for engine in self._engines.values():
            engine.reset()
