import importlib.util
import ast
from collections import deque
import os
from pathlib import Path
from threading import Lock, RLock, Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]


class CredentialsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("speech_credentials", ROOT / "src/speech_to_speech/utils/speech_credentials.py")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_dedicated_keys_and_explicit_keys_do_not_use_llm_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "llm", "S2S_STT_API_KEY": "asr", "S2S_TTS_API_KEY": "tts"}, clear=True):
            self.assertEqual(self.module.resolve_speech_api_key(None, "S2S_STT_API_KEY"), "asr")
            self.assertEqual(self.module.resolve_speech_api_key(None, "S2S_TTS_API_KEY"), "tts")
            self.assertEqual(self.module.resolve_speech_api_key("explicit", "S2S_TTS_API_KEY"), "explicit")

    def test_empty_service_key_disables_auth_and_unset_preserves_legacy_behavior(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "llm", "S2S_TTS_API_KEY": ""}, clear=True):
            self.assertEqual(self.module.resolve_speech_api_key(None, "S2S_TTS_API_KEY"), "")
            self.assertIsNone(self.module.resolve_speech_api_key(None, "S2S_STT_API_KEY"))

    def test_actual_http_handler_setup_resolves_the_correct_service_key(self):
        # Exercise the production setup methods without importing model stacks.
        for relative, class_name, key_name in (
            ("STT/openai_compatible_handler.py", "OpenAICompatibleSTTHandler", "S2S_STT_API_KEY"),
            ("TTS/openai_compatible_handler.py", "OpenAICompatibleTTSHandler", "S2S_TTS_API_KEY"),
        ):
            path = ROOT / "src/speech_to_speech" / relative
            tree = ast.parse(path.read_text(encoding="utf-8"))
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
            setup = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "setup")
            future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
            namespace = {"os": os, "Lock": Lock, "RLock": RLock, "deque": deque,
                         "OPENAI_BASE_URL": "https://api.openai.com/v1"}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[future, setup], type_ignores=[])), str(path), "exec"), namespace)
            with patch.dict("sys.modules", {"speech_to_speech.utils.speech_credentials": self.module}):
                for key in ("dedicated-key", ""):
                    with self.subTest(handler=class_name, key=bool(key)), patch.dict(os.environ, {key_name: key, "OPENAI_API_KEY": "llm-secret"}, clear=True):
                        handler = SimpleNamespace(warmup=MagicMock())
                        kwargs = {"base_url": "http://service/v1"}
                        if relative.startswith("TTS/"):
                            kwargs["should_listen"] = Event()
                        namespace["setup"](handler, **kwargs)
                        self.assertEqual(handler.api_key, key)
