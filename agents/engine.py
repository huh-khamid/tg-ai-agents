"""
Движок агента: общается с Gemini, вызывает инструменты, хранит историю.
"""

import asyncio
import json
import os
import re
from typing import Optional

import google.generativeai as genai

from agents.definitions import AgentDefinition, AGENTS
from tools.tools import TOOLS_REGISTRY, fetch_url


# Gemini tool schemas для function calling
TOOL_SCHEMAS = {
    "web_search": {
        "name": "web_search",
        "description": "Поиск информации в интернете через DuckDuckGo",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Поисковый запрос"},
                "max_results": {"type": "integer", "description": "Количество результатов (1-10)", "default": 5}
            },
            "required": ["query"]
        }
    },
    "fetch_url": {
        "name": "fetch_url",
        "description": "Загрузить и прочитать содержимое веб-страницы по URL",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL страницы для загрузки"}
            },
            "required": ["url"]
        }
    },
    "run_python": {
        "name": "run_python",
        "description": "Выполнить Python-код и получить результат",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python код для выполнения"}
            },
            "required": ["code"]
        }
    },
    "save_file": {
        "name": "save_file",
        "description": "Сохранить текст в файл",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Имя файла"},
                "content": {"type": "string", "description": "Содержимое файла"}
            },
            "required": ["filename", "content"]
        }
    },
    "read_file": {
        "name": "read_file",
        "description": "Прочитать содержимое файла",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Имя файла"}
            },
            "required": ["filename"]
        }
    },
    "list_files": {
        "name": "list_files",
        "description": "Показать список всех сохранённых файлов",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
}


class AgentEngine:
    """Движок одного агента: управляет диалогом и вызовами инструментов."""

    def __init__(self, definition: AgentDefinition, api_key: str):
        genai.configure(api_key=api_key)

        # Собираем только инструменты, доступные этому агенту
        tools = [TOOL_SCHEMAS[t] for t in definition.tools if t in TOOL_SCHEMAS]

        self.definition = definition
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=definition.system_prompt,
            tools=tools if tools else None,
        )
        self.chat = self.model.start_chat(history=[])
        self._max_tool_calls = 5  # Максимум вызовов инструментов за один ответ

    async def _call_tool(self, name: str, args: dict) -> str:
        """Вызывает инструмент по имени."""
        fn = TOOLS_REGISTRY.get(name)
        if fn is None:
            return f"❌ Инструмент `{name}` не найден"
        try:
            if asyncio.iscoroutinefunction(fn):
                return await fn(**args)
            else:
                return fn(**args)
        except Exception as e:
            return f"❌ Ошибка инструмента {name}: {e}"

    async def process(self, user_message: str) -> str:
        """
        Обрабатывает сообщение пользователя.
        Возвращает финальный текстовый ответ агента.
        """
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.chat.send_message(user_message)
        )

        tool_calls_count = 0
        while tool_calls_count < self._max_tool_calls:
            # Проверяем, есть ли function calls в ответе
            has_tool_call = False
            tool_results = []

            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        has_tool_call = True
                        fc = part.function_call
                        tool_name = fc.name
                        tool_args = dict(fc.args)

                        result = await self._call_tool(tool_name, tool_args)
                        tool_results.append({
                            "tool_name": tool_name,
                            "result": str(result)
                        })
                        tool_calls_count += 1

            if not has_tool_call:
                break

            # Отправляем результаты инструментов обратно
            from google.generativeai.types import content_types
            tool_response_parts = []
            for tr in tool_results:
                tool_response_parts.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=tr["tool_name"],
                            response={"result": tr["result"]},
                        )
                    )
                )

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.chat.send_message(tool_response_parts),
            )

        # Извлекаем финальный текст
        text_parts = []
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)

        return "\n".join(text_parts).strip() or "🤔 Агент не дал ответа."

    def reset(self):
        """Сбрасывает историю диалога."""
        self.chat = self.model.start_chat(history=[])


class AgentManager:
    """Менеджер всех агентов для одного пользователя."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._engines: dict[str, AgentEngine] = {}

    def get_engine(self, agent_name: str) -> Optional[AgentEngine]:
        if agent_name not in AGENTS:
            return None
        if agent_name not in self._engines:
            self._engines[agent_name] = AgentEngine(AGENTS[agent_name], self.api_key)
        return self._engines[agent_name]

    def reset_agent(self, agent_name: str):
        if agent_name in self._engines:
            self._engines[agent_name].reset()

    def reset_all(self):
        for engine in self._engines.values():
            engine.reset()
