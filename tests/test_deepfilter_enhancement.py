"""DeepFilterNet adapter checks with a buffered native-runtime substitute."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class BufferedProcessor:
    instances = []

    def __init__(self, **kwargs):
        self.sample_rate = 48000
        self.closed = False
        self.instances.append(self)

    def process_chunk(self, audio):
        self.audio = audio.copy()
        return audio[:0]

    def finalize(self):
        return self.audio * 0.5

    def close(self):
        self.closed = True


class EnhancementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("deepfilter_adapter", ROOT / "src/speech_to_speech/VAD/deepfilter_enhancement.py")
        cls.adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.adapter)

    def test_flushes_tail_preserves_length_and_creates_fresh_state_per_utterance(self):
        BufferedProcessor.instances = []
        with patch.object(self.adapter, "DeepFilterNetRealtime", BufferedProcessor):
            model = self.adapter.DeepFilterEnhancer(16000)
            for length in (16000, 1237, 1):
                audio = np.ones(length, dtype=np.float32)
                result = model.enhance(audio)
                self.assertEqual(result.shape, audio.shape)
                self.assertEqual(result.dtype, np.float32)
                self.assertTrue(np.isfinite(result).all())
            self.assertEqual(len(BufferedProcessor.instances), 3)
            self.assertTrue(all(p.closed for p in BufferedProcessor.instances))

    def test_empty_input_does_not_initialize_model(self):
        with patch.object(self.adapter, "DeepFilterNetRealtime") as factory:
            result = self.adapter.DeepFilterEnhancer(16000).enhance(np.zeros(0, dtype=np.float32))
            self.assertEqual(result.size, 0)
            factory.assert_not_called()

    def test_custom_model_path_reaches_native_runtime(self):
        with patch.object(self.adapter, "DeepFilterNetRealtime", BufferedProcessor):
            model = self.adapter.DeepFilterEnhancer(48000, model_path="custom-model.tar.gz")
            self.assertEqual(model.model_path, "custom-model.tar.gz")
            result = model.enhance(np.ones(480, dtype=np.float32))
            self.assertEqual(result.size, 480)

    def test_native_failure_closes_processor_and_is_not_silenced(self):
        with patch.object(self.adapter, "DeepFilterNetRealtime") as factory:
            factory.return_value.sample_rate = 48000
            factory.return_value.process_chunk.side_effect = RuntimeError("native failure")
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                self.adapter.DeepFilterEnhancer(16000).enhance(np.ones(100, dtype=np.float32))
            factory.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
