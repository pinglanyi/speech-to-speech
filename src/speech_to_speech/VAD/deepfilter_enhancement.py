"""DeepFilterNet's Rust runtime with bundled weights and mono segment I/O.

Each utterance has independent recurrent state. No model download, CUDA,
or legacy torchaudio I/O API is needed by this adapter.
"""
from math import gcd

import numpy as np
from scipy.signal import resample_poly

try:
    from deepfilternet_rs import DeepFilterNetRealtime
except (ImportError, OSError):
    DeepFilterNetRealtime = None

HAS_DEEPFILTER = DeepFilterNetRealtime is not None


class DeepFilterEnhancer:
    def __init__(self, sample_rate: int):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.sample_rate = sample_rate

    def enhance(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("DeepFilterNet expects mono audio")
        if not audio.size:
            return audio.copy()
        if DeepFilterNetRealtime is None:
            raise RuntimeError("Install deepfilternet-rs==0.1.1 with bash run.sh before enabling enhancement")
        processor = DeepFilterNetRealtime(compensate_delay=True, log_level="warn")
        try:
            rate = processor.sample_rate
            factor = gcd(self.sample_rate, rate)
            samples = audio if rate == self.sample_rate else resample_poly(
                audio, rate // factor, self.sample_rate // factor).astype(np.float32)
            # The final tail contains buffered samples and delay compensation.
            result = np.concatenate([processor.process_chunk(np.ascontiguousarray(samples)), processor.finalize()])
            if rate != self.sample_rate:
                result = resample_poly(result, self.sample_rate // factor, rate // factor)
            if result.size < audio.size:
                result = np.pad(result, (0, audio.size - result.size))
            return np.asarray(result[:audio.size], dtype=np.float32)
        finally:
            processor.close()
