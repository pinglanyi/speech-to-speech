"""Run the cleanup method without importing GPU/model dependencies."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]


class CleanupTests(unittest.TestCase):
    def test_partial_setup_and_repeated_cleanup_do_not_mask_original_error(self):
        source = ROOT / "src/speech_to_speech/TTS/qwen3_tts_handler.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "cleanup")
        logger = MagicMock()
        namespace = {"Path": Path, "logger": logger}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        torch = MagicMock()
        torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": torch}):
            for handler in (SimpleNamespace(), SimpleNamespace(backend="torch", model=object())):
                namespace["cleanup"](handler)
                namespace["cleanup"](handler)
        logger.warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
