"""
Движок агента: общается с Groq API,
вызывает инструменты, хранит историю диалога.
При превышении лимита автоматически переключается на следующую модель.
"""

import asyncio
import json
import logging
from typing import Optional

from groq import AsyncGroq, RateLimitError, APIStatusError

from agents.definitions import AgentDefinition, AGENTS
from tools.tools import TOOLS_REGISTRY

logger = logging.getLogger(__name__)

# ─── Цепочка фоллбэк-моделей (Groq) ─────────────────────────────────────────
# Бесплатно, быстро, без карты — просто зарегистрируйся на console.groq.com

FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",   # 1. Llama 3.3 70B (основная, мощная)
    "llama-3.1-70b-versatile",   # 2. Llama 3.1 70B (запасная)
    "mixtral-8x7b-32768",        # 3. Mixtral 8x7B (большой контекст)
    "llama-3.1-8b-instant",      # 4. Llama 3.1 8B (быстрая)
    "gemma2-9b-it",              # 5. Gemma 2 9B (резервная)
]

DEFAULT_MODEL = FALLBACK_MODELS[0]


# ─── Схемы инструментов (OpenAI-совместимый формат) ──────────────────────────

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
        self.client = AsyncGroq(api_key=api_key)
        self.tools = get_tools_for_agent(definition.tools)
        self.history: list[dict] = [
            {"role": "system", "content": definition.system_prompt}
        ]
        self._max_tool_calls = 8

        # Строим цепочку фоллбэка начиная с выбранной модели
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

    async def process(self, user_message: str) -> tuple[str, str]:
        """Обрабатывает сообщение, возвращает (ответ, модель)."""
        self.history.append({"role": "user", "content": user_message})
        history_checkpoint = len(self.history) - 1

        for _ in range(len(self._model_chain)):
            model = self.current_model
            try:
                result = await self._run_with_tools(model)
                return result, model

            except RateLimitError:
                logger.warning(f"⚠️ Лимит модели {model}, переключаюсь...")
                next_m = self._next_model()
                if next_m is None:
                    return "❌ Все модели достигли лимита. Попробуй позже.", model
                self.history = self.history[:history_checkpoint + 1]

            except APIStatusError as e:
                if e.status_code in (404, 429, 503, 529):
                    logger.warning(f"⚠️ Модель {model} недоступна (код {e.status_code}), переключаюсь...")
                    next_m = self._next_model()
                    if next_m is None:
                        return "❌ Все модели недоступны. Попробуй позже.", model
                    self.history = self.history[:history_checkpoint + 1]
                else:
                    raise

        return "❌ Не удалось получить ответ.", self.current_model

    async def _run_with_tools(self, model: str) -> str:
        """Один цикл запрос → инструменты → ответ."""
        for _ in range(self._max_tool_calls):
            response = await self.client.chat.completions.create(
                model=model,
                messages=self.history,
                tools=self.tools if self.tools else None,
                tool_choice="auto" if self.tools else None,
            )
            msg = response.choices[0].message
            self.history.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                return msg.content or "🤔 Агент не дал ответа."

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                result = await self._call_tool(fn_name, fn_args)
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        return "⚠️ Агент достиг лимита вызовов инструментов."

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
