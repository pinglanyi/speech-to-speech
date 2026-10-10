# Ubuntu Bailian Launcher Implementation Plan

**Goal:** Provide a venv-based Ubuntu launcher for NVIDIA 48GB and CPU Chinese voice testing, using Bailian for text generation, on the requested test branch.

**Architecture:** A small Bash entrypoint delegates to a standard-library Python bootstrap/supervisor. Credentials come from environment or an ignored .env file; ephemeral pipeline JSON contains no credentials. Both services bind loopback and are accessed through two SSH forwards.

**Tech Stack:** Python 3.11/3.12, venv, pip, existing speech-to-speech CLI, Node.js 20+, existing browser demo.

- [x] Confirm requested remote main matches the downloaded commit and create test branch.
- [x] Write launcher tests before implementation and observe missing-feature failure.
- [x] Implement GPU Qwen3-ASR/Qwen3-TTS and CPU Faster Whisper/OmniVoice configurations, Bailian preflight, isolated venv bootstrap, dependency reuse, process cleanup, and .env parsing.
- [x] Document system packages, provider endpoint selection, startup, SSH browser testing, CPU limitations, and verification scope.
- [x] Run launcher tests, validate configs against upstream definitions, verify shell syntax and inspect diff for credentials.
- [ ] Commit changes and push test without rewriting existing history.
