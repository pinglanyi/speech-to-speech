# Remote speech services through .env

User request: expose the existing remote ASR/TTS adapters through the one-click environment configuration.

1. Support `openai` HTTP STT, `vllm-realtime` WebSocket STT and `openai` HTTP TTS. Map base URLs, served model names, timeouts, supported formats and optional vLLM-Omni streaming/task settings. Reject invalid URL/format/stream combinations before model loading.
2. Resolve STT and TTS credentials from dedicated environment variables; retain explicit CLI credential precedence and old fallback behavior outside the launcher. Never serialize credentials to the pipeline config or forward them to the demo process.
3. Skip local CUDA preflight when both speech stages use remote services. Retain local VAD, turn detection and optional enhancement. Remote servers must be deployed independently.
4. Add a generic compatible LLM provider for local vLLM HTTP endpoints, separate served model name and optional key, keeping DeepSeek/Bailian defaults.
5. Verify config mappings, credentials (including production HTTP setup methods), missing/invalid options and remote GPU independence. Document complete examples and limits. Real vLLM/vLLM-Omni integration needs the user's running endpoints.
