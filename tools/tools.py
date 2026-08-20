"""
Инструменты для агентов: поиск в интернете, работа с файлами, выполнение кода.
"""

import subprocess
import sys
import tempfile
import os
import re
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS


# ─── Веб-поиск ────────────────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 5) -> str:
    """Ищет информацию в интернете через DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "Ничего не найдено."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. **{r['title']}**\n   {r['body']}\n   🔗 {r['href']}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Ошибка поиска: {e}"


async def fetch_url(url: str) -> str:
    """Загружает и читает содержимое веб-страницы."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                html = await resp.text()
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Обрезаем до 3000 символов
        return text[:3000] + ("…" if len(text) > 3000 else "")
    except Exception as e:
        return f"Ошибка загрузки страницы: {e}"


# ─── Выполнение Python-кода ────────────────────────────────────────────────────

def run_python(code: str) -> str:
    """Безопасно выполняет Python-код и возвращает вывод."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = ""
        if result.stdout:
            output += f"📤 Вывод:\n{result.stdout}"
        if result.stderr:
            output += f"\n⚠️ Ошибки:\n{result.stderr}"
        return output.strip() or "✅ Код выполнен (без вывода)"
    except subprocess.TimeoutExpired:
        return "⏱️ Время выполнения истекло (30 сек)"
    except Exception as e:
        return f"❌ Ошибка: {e}"
    finally:
        os.unlink(fname)


# ─── Работа с файлами ─────────────────────────────────────────────────────────

FILES_DIR = os.path.join(os.path.dirname(__file__), "files")
os.makedirs(FILES_DIR, exist_ok=True)


def save_file(filename: str, content: str) -> str:
    """Сохраняет текст в файл."""
    # Безопасное имя файла
    safe_name = re.sub(r"[^\w\-_\. ]", "_", filename)
    path = os.path.join(FILES_DIR, safe_name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"✅ Файл сохранён: `{safe_name}`"


def read_file(filename: str) -> str:
    """Читает содержимое файла."""
    safe_name = re.sub(r"[^\w\-_\. ]", "_", filename)
    path = os.path.join(FILES_DIR, safe_name)
    if not os.path.exists(path):
        return f"❌ Файл `{safe_name}` не найден."
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return content[:4000] + ("…" if len(content) > 4000 else "")


def list_files() -> str:
    """Показывает список сохранённых файлов."""
    files = os.listdir(FILES_DIR)
    if not files:
        return "📂 Файлов нет."
    return "📂 Файлы:\n" + "\n".join(f"  • {f}" for f in sorted(files))


# ─── Реестр инструментов ──────────────────────────────────────────────────────

TOOLS_REGISTRY = {
    "web_search": web_search,
    "fetch_url": fetch_url,
    "run_python": run_python,
    "save_file": save_file,
    "read_file": read_file,
    "list_files": list_files,
}
