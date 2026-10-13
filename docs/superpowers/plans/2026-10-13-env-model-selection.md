# Environment model selection

User request: configure and change pipeline models through one `.env` file.

1. Retain existing GPU/CPU presets when selectors are auto or empty. Map explicit STT backend/model/language/precision and TTS backend/model/language/voice/reference parameters to upstream argument fields. Reject unsupported backend names and CPU Qwen3-TTS before dependency installation.
2. Derive extra dependencies from the selected backends, retaining the existing venv fingerprint installation mechanism. Keep API credentials in environment variables.
3. Expose VAD selection, optional local Silero/FireRed weights, Smart Turn switch/local ONNX model and optional DeepFilterNet native model. Native model formats must match their loaders.
4. Supply the configured speaker and system prompt to the browser's bootstrap config so stale local storage does not override deployment settings on page load. Keep standalone demo behavior when overrides are absent.
5. Add regression coverage for default preservation, alternate backend fields/dependencies, errors, custom weights and browser defaults. Update the environment example and usage instructions, then push a new commit to test.
