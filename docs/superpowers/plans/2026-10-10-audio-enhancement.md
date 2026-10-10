# DeepFilterNet and Qwen TTS compatibility

User requirements: always install usable DeepFilterNet with the Ubuntu venv launcher; toggle it only through `.env`. Keep it disabled by default. Also address the reported Qwen TTS startup traceback.

Implementation:
1. Install pinned `deepfilternet-rs==0.1.1` for both profiles, including its bundled weights. This wraps the upstream Rust runtime without the legacy Python package's NumPy/torchaudio constraints. Validate actual native model inference after installation before caching success.
2. Map `S2S_AUDIO_ENHANCEMENT=0/1` to `audio_enhancement`. Resample mono speech to the native rate, flush buffered output, preserve segment length, release native resources and isolate state between utterances. Disabled enhancement never constructs a processor.
3. Adapt only qwen-tts-hf's known default RoPE registry shim to `rope_parameters` while retaining configured theta and dimensions; preserve legacy configs and unrelated initializers. Make partial TTS setup cleanup idempotent.
4. Verify dependency/config tests, buffered-audio edge cases, native bundled-weight inference and RoPE schema/cleanup regressions. Actual Ubuntu GPU TTS remains a server validation step.
