"""
Движок агента: общается с OpenRouter (OpenAI-совместимый API),
вызывает инструменты, хранит историю диалога.
"""

import asyncio
import json
from typing import Optional

from openai import AsyncOpenAI

from agents.definitions import AgentDefinition, AGENTS
from tools.tools import TOOLS_REGISTRY


# ─── Схемы инструментов (OpenAI function calling формат) ──────────────────────

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

# Фильтруем схемы по списку инструментов агента
def get_tools_for_agent(agent_tools: list[str]) -> list[dict]:
    return [t for t in TOOL_SCHEMAS if t["function"]["name"] in agent_tools]


# ─── Движок агента ────────────────────────────────────────────────────────────

# Бесплатные модели OpenRouter (хорошее соотношение цена/качество)
DEFAULT_MODEL = "deepseek/deepseek-chat-v3-0324:free"

class AgentEngine:
    """Движок одного агента: управляет диалогом и вызовами инструментов."""

    def __init__(self, definition: AgentDefinition, api_key: str, model: str = DEFAULT_MODEL):
        self.definition = definition
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.tools = get_tools_for_agent(definition.tools)
        # История диалога
        self.history: list[dict] = [
            {"role": "system", "content": definition.system_prompt}
        ]
        self._max_tool_calls = 8

    async def _call_tool(self, name: str, args: dict) -> str:
        """Вызывает инструмент по имени."""
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

    async def process(self, user_message: str) -> str:
        """Обрабатывает сообщение и возвращает финальный ответ."""
        self.history.append({"role": "user", "content": user_message})

        for _ in range(self._max_tool_calls):
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                tools=self.tools if self.tools else None,
                tool_choice="auto" if self.tools else None,
            )
            msg = response.choices[0].message

            # Добавляем ответ модели в историю
            self.history.append(msg.model_dump(exclude_none=True))

            # Если нет tool_calls — финальный ответ
            if not msg.tool_calls:
                return msg.content or "🤔 Агент не дал ответа."

            # Выполняем все вызванные инструменты
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
        """Сбрасывает историю диалога."""
        self.history = [{"role": "system", "content": self.definition.system_prompt}]


# ─── Менеджер агентов ─────────────────────────────────────────────────────────

class AgentManager:
    """Менеджер всех агентов для одного пользователя."""

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
