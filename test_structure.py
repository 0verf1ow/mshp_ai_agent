import ast
import importlib
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent

# Проверки структуры не обращаются к API и могут запускаться до установки
# внешних библиотек. Для импорта модулей достаточно этих безопасных заглушек.
try:
    import dotenv  # noqa: F401
except ModuleNotFoundError:
    sys.modules["dotenv"] = SimpleNamespace(load_dotenv=lambda: None)

try:
    import openai  # noqa: F401
except ModuleNotFoundError:
    sys.modules["openai"] = SimpleNamespace(OpenAI=object)


class FakeCompletions:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.calls = []

    def create(self, *, model, messages):
        self.calls.append({"model": model, "messages": messages})
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class FakeClient:
    def __init__(self, content=None, error=None):
        completions = FakeCompletions(content=content, error=error)
        self.completions = completions
        self.chat = SimpleNamespace(completions=completions)


def imported_project_modules(filename):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module.split(".")[0])
    return result & {"main", "agent", "llm_client", "config"}


def defined_functions(filename):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


class StructureTests(unittest.TestCase):
    def test_01_config_module(self):
        config = importlib.import_module("config")
        self.assertTrue(callable(getattr(config, "load_settings", None)))
        with patch.object(config.dotenv, "load_dotenv"), patch.dict(
            os.environ,
            {"MODEL": "model", "BASE_URL": "https://example.test", "API_KEY": "key"},
            clear=True,
        ):
            self.assertEqual(
                config.load_settings(),
                ("model", "https://example.test", "key"),
            )

    def test_02_llm_client_module(self):
        llm_client = importlib.import_module("llm_client")
        self.assertTrue(callable(getattr(llm_client, "parse_decision", None)))
        self.assertTrue(callable(getattr(llm_client, "request_decision", None)))
        raw = json.dumps({"next_step": "final_answer", "message": "  Готово  "})
        self.assertEqual(
            llm_client.parse_decision(raw),
            {"next_step": "final_answer", "message": "Готово"},
        )
        client = FakeClient(raw)
        messages = [{"role": "user", "content": "Запрос"}]
        self.assertEqual(
            llm_client.request_decision(client, "model", messages),
            {"next_step": "final_answer", "message": "Готово"},
        )
        self.assertEqual(client.completions.calls, [{"model": "model", "messages": messages}])

    def test_03_agent_module(self):
        agent = importlib.import_module("agent")
        self.assertTrue(callable(getattr(agent, "build_messages", None)))
        self.assertTrue(callable(getattr(agent, "route_decision", None)))
        self.assertTrue(callable(getattr(agent, "handle_request", None)))
        with patch.object(
            agent,
            "request_decision",
            return_value={"next_step": "final_answer", "message": "Ответ"},
        ):
            self.assertEqual(agent.handle_request(object(), "model", "Запрос"), "Ответ")

    def test_04_main_contains_only_entry_point(self):
        main = importlib.import_module("main")
        self.assertTrue(callable(getattr(main, "main", None)))
        self.assertEqual(defined_functions("main.py"), {"main"})

    def test_05_functions_are_not_duplicated(self):
        expected = {
            "config.py": {"load_settings"},
            "llm_client.py": {"parse_decision", "request_decision"},
            "agent.py": {"build_messages", "route_decision", "handle_request"},
            "main.py": {"main"},
        }
        for filename, functions in expected.items():
            with self.subTest(filename=filename):
                self.assertEqual(defined_functions(filename), functions)

    def test_06_dependency_direction(self):
        self.assertEqual(imported_project_modules("config.py"), set())
        self.assertEqual(imported_project_modules("llm_client.py"), set())
        self.assertEqual(imported_project_modules("agent.py"), {"llm_client"})
        self.assertEqual(imported_project_modules("main.py"), {"agent", "config"})

    def test_07_gitignore(self):
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        patterns = {line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")}
        required = {".env", ".venv/", "__pycache__/", "*.py[cod]", "*.log"}
        self.assertTrue(required <= patterns, f"Не хватает шаблонов: {sorted(required - patterns)}")


if __name__ == "__main__":
    unittest.main()
