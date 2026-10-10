import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class RopeCompatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("rope_compat", ROOT / "src/speech_to_speech/TTS/qwen_rope_compat.py")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_new_mimi_schema_uses_actual_theta_without_mutating_config(self):
        def original(config, device=None, seq_len=None, layer_type=None):
            return config.rope_theta, config.head_dim, config.partial_rotary_factor
        original.__module__ = "qwen_tts._transformers_compat"
        registry = {"default": original}
        mimi = SimpleNamespace(rope_parameters={"rope_theta": 10000.0}, head_dim=64,
                               hidden_size=512, num_attention_heads=8)
        with self.assertRaises(AttributeError):
            registry["default"](mimi)
        self.module.install_qwen_rope_compat(registry)
        self.assertEqual(registry["default"](mimi), (10000.0, 64, 1.0))
        self.assertFalse(hasattr(mimi, "rope_theta"))
        installed = registry["default"]
        self.module.install_qwen_rope_compat(registry)
        self.assertIs(registry["default"], installed)

    def test_lazy_qwen_shim_is_registered_before_adaptation(self):
        def original(config, **kwargs):
            return config.rope_theta
        original.__module__ = "qwen_tts._transformers_compat"
        registry = {}
        def register():
            registry.setdefault("default", original)
        modules = {
            "qwen_tts": SimpleNamespace(),
            "qwen_tts._transformers_compat": SimpleNamespace(patch_transformers_rope_registry=register),
            "transformers": SimpleNamespace(),
            "transformers.modeling_rope_utils": SimpleNamespace(ROPE_INIT_FUNCTIONS=registry),
        }
        with patch.dict("sys.modules", modules):
            self.assertTrue(self.module.install_qwen_rope_compat())
        config = SimpleNamespace(rope_parameters={"rope_theta": 10000}, hidden_size=512, num_attention_heads=8)
        self.assertEqual(registry["default"](config), 10000)

    def test_legacy_config_passes_through_unchanged_and_other_initializer_is_preserved(self):
        def original(config, **kwargs):
            return config
        original.__module__ = "qwen_tts._transformers_compat"
        registry = {"default": original}
        self.module.install_qwen_rope_compat(registry)
        legacy = SimpleNamespace(rope_theta=1000000)
        self.assertIs(registry["default"](legacy), legacy)
        unrelated = lambda *args: None
        other_registry = {"default": unrelated}
        self.module.install_qwen_rope_compat(other_registry)
        self.assertIs(other_registry["default"], unrelated)

    def test_missing_theta_is_rejected_instead_of_inventing_model_parameters(self):
        def original(config, **kwargs):
            return config.rope_theta
        original.__module__ = "qwen_tts._transformers_compat"
        registry = {"default": original}
        self.module.install_qwen_rope_compat(registry)
        with self.assertRaisesRegex(ValueError, "rope_theta"):
            registry["default"](SimpleNamespace(rope_parameters={}))
