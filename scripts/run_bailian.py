"""Ubuntu venv bootstrap, Bailian preflight, and supervised browser demo.

Uses only the standard library until the isolated environment is ready.
Credentials remain in environment variables and never enter CLI arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def venv_path(profile: str) -> Path:
    return ROOT / f".venv-{profile}"


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid .env assignment at line {number}; use NAME=value.")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(name, value)


def require_key(env: dict[str, str]) -> str:
    key = env.get("DASHSCOPE_API_KEY", "").strip()
    if not key:
        raise ValueError("Set DASHSCOPE_API_KEY in .env or export it before starting.")
    return key


def pipeline_config(profile: str, env: dict[str, str]) -> dict:
    config = {
        "host": "127.0.0.1",
        "port": int(env.get("S2S_PORT", "8765")),
        "num_pipelines": 1,
        "llm_backend": "chat-completions",
        "model_name": env.get("DASHSCOPE_MODEL", "qwen-plus"),
        "responses_api_base_url": env.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        # Other providers' reasoning_effort/chat_template_kwargs are not Bailian flags.
        "responses_api_reasoning_effort": None,
        "responses_api_disable_thinking": False,
        "init_chat_prompt": "你是中文语音助手。请用简洁自然的中文回答，通常不超过三句话。",
        "stream_batch_sentences": 1,
        "log_transcripts": False,
    }
    if profile == "gpu":
        config.update({
            "device": "cuda", "stt": "qwen3-asr",
            "qwen3_asr_model_name": "Qwen/Qwen3-ASR-1.7B-hf",
            "qwen3_asr_torch_dtype": "float16",
            "qwen3_asr_language": "zh", "tts": "qwen3",
            "qwen3_tts_backend": "torch", "qwen3_tts_language": "chinese",
            "qwen3_tts_dtype": "float16",
            "qwen3_tts_speaker": "Vivian",
        })
    elif profile == "cpu":
        config.update({
            "device": "cpu", "stt": "faster-whisper",
            "faster_whisper_stt_model_name": "small",
            "faster_whisper_stt_compute_type": "int8",
            "faster_whisper_stt_gen_language": "zh", "tts": "omnivoice",
            "omnivoice_device": "cpu", "omnivoice_dtype": "float32",
            "omnivoice_language": "Chinese",
            "enable_live_transcription": False,
        })
    else:
        raise ValueError("Profile must be gpu or cpu.")
    return config


def install_requirements(profile: str) -> list[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    # This launcher deliberately uses the torch backend, avoiding the upstream
    # Linux GGML wheel's glibc 2.39 / CUDA 12.8 requirement on Ubuntu 22.04.
    deps = [dep.replace("faster-qwen3-tts[ggml]", "faster-qwen3-tts") for dep in project["dependencies"]]
    if profile == "cpu":
        for extra in ("faster-whisper", "omnivoice"):
            deps.extend(project["optional-dependencies"][extra])
    # Browser HTTP adapter imports auth and limiter (optional HF OAuth).
    deps.append("huggingface-hub[oauth]>=0.34.0")
    return deps


def run(command: list[str], **kwargs) -> None:
    subprocess.run(command, check=True, cwd=ROOT, **kwargs)


def bootstrap(profile: str, reinstall: bool, demo: bool) -> Path:
    environment = venv_path(profile)
    python = environment / "bin/python"
    if not python.exists():
        print(f"Creating {environment.name} ...", flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    deps = install_requirements(profile)
    fingerprint = hashlib.sha256(json.dumps([deps, sys.version_info[:2]]).encode()).hexdigest()
    marker = environment / ".bailian.installed"
    if reinstall or not marker.exists() or marker.read_text() != fingerprint:
        print("Installing Python dependencies (first run may take several minutes) ...", flush=True)
        run([str(python), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])
        if profile == "cpu":
            run([str(python), "-m", "pip", "install", "torch>=2.4.0", "torchaudio>=2.4.0",
                 "--index-url", "https://download.pytorch.org/whl/cpu"])
        run([str(python), "-m", "pip", "install", *deps])
        run([str(python), "-m", "pip", "install", "--no-deps", "-e", "."])
        run([str(python), "-m", "pip", "check"])
        marker.write_text(fingerprint)
    if demo and (reinstall or not (ROOT / "demo/node_modules/@openai/agents/package.json").exists()):
        run(["npm", "ci", "--prefix", "demo"])
    return python


def check_api(env: dict[str, str]) -> None:
    key = require_key(env)
    base = env.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
    if not base.startswith("https://"):
        raise ValueError("DASHSCOPE_BASE_URL must use https://.")
    payload = {"model": env.get("DASHSCOPE_MODEL", "qwen-plus"),
               "messages": [{"role": "user", "content": "请只回复：连接成功"}], "max_tokens": 32}
    request = urllib.request.Request(base + "/chat/completions", json.dumps(payload).encode(),
                                     {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            result = json.load(response)
        if not result.get("choices"):
            raise ValueError("Bailian response has no choices; verify the endpoint/model.")
    except urllib.error.HTTPError as error:
        # Never print raw provider payloads, which may contain request details.
        raise ValueError(f"Bailian HTTP {error.code}: check key, region/base URL, model access and balance.") from None
    except (urllib.error.URLError, TimeoutError):
        raise ValueError("Bailian connection failed: check DNS, network and DASHSCOPE_BASE_URL.") from None
    print("Bailian API connection OK.", flush=True)


def ensure_free(port: int) -> None:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            raise ValueError(f"Local port {port} is occupied; stop the old process or change .env.") from None


def supervise(python: Path, config: dict, demo: bool) -> int:
    folder = ROOT / ".runtime"
    folder.mkdir(mode=0o700, exist_ok=True)
    path = folder / "bailian.json"
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    path.chmod(0o600)
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = require_key(env)
    processes: list[subprocess.Popen] = []

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        backend = subprocess.Popen([str(python.parent / "speech-to-speech"), "serve", str(path)], cwd=ROOT, env=env)
        processes.append(backend)
        deadline = time.monotonic() + float(env.get("S2S_STARTUP_TIMEOUT", "1800"))
        print("Loading speech models; first run downloads weights. Waiting for backend ...", flush=True)
        while True:
            if backend.poll() is not None:
                raise ValueError("Speech backend exited during startup; inspect the model/dependency messages above.")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{config['port']}/v1/pool", timeout=1) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError):
                pass
            if time.monotonic() > deadline:
                raise ValueError("Model startup timed out; increase S2S_STARTUP_TIMEOUT or check model downloads.")
            time.sleep(1)
        if demo:
            demo_env = env.copy()
            demo_env["SPEECH_TO_SPEECH_URL"] = f"ws://localhost:{config['port']}/v1/realtime"
            demo_env["STARTUP_GREETING"] = "请用中文简短地打招呼。"
            # The browser adapter only needs the public backend URL, not the LLM key.
            for key in ("DASHSCOPE_API_KEY", "OPENAI_API_KEY"):
                demo_env.pop(key, None)
            port = env.get("S2S_DEMO_PORT", "7860")
            processes.append(subprocess.Popen([str(python), "-m", "uvicorn", "--app-dir", "demo", "server:app",
                                               "--host", "127.0.0.1", "--port", port], cwd=ROOT, env=demo_env))
            print(f"Browser: http://localhost:{port} (forward both ports with SSH). Ctrl+C stops both services.", flush=True)
        while all(process.poll() is None for process in processes):
            time.sleep(1)
        return next((process.returncode or 1 for process in processes if process.poll() is not None), 1)
    except KeyboardInterrupt:
        return 0
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        path.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Ubuntu venv launcher: local Chinese speech + Bailian LLM")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gpu", action="store_true", help="Qwen3-ASR + Qwen3-TTS on NVIDIA")
    group.add_argument("--cpu", action="store_true", help="Faster Whisper int8 + OmniVoice on CPU (slower)")
    parser.add_argument("--check-api", action="store_true", help="Check Bailian only, without installing or loading models")
    parser.add_argument("--install-only", action="store_true", help="Prepare .venv without starting services")
    parser.add_argument("--reinstall", action="store_true", help="Refresh dependencies")
    parser.add_argument("--no-demo", action="store_true", help="Start only the speech backend; no Node.js needed")
    parser.add_argument("--dry-run", action="store_true", help="Print the credential-free pipeline configuration")
    args = parser.parse_args(argv)
    load_env(ROOT / ".env")
    profile = "gpu" if args.gpu else "cpu" if args.cpu else os.environ.get("S2S_PROFILE", "auto")
    if profile == "auto":
        profile = "gpu" if shutil.which("nvidia-smi") and subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True).returncode == 0 else "cpu"
    config = pipeline_config(profile, dict(os.environ))
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return 0
    if args.check_api:
        check_api(dict(os.environ))
        return 0
    if sys.platform != "linux" or not ((3, 11) <= sys.version_info[:2] < (3, 13)):
        raise ValueError("Use Ubuntu with Python 3.11 or 3.12 (python3-venv installed).")
    demo = not args.no_demo
    if demo and not shutil.which("npm"):
        raise ValueError("Install Node.js 20+ and npm, or pass --no-demo.")
    if demo:
        version = subprocess.check_output(["node", "--version"], text=True).strip()
        if int(version.lstrip("v").split(".")[0]) < 20:
            raise ValueError("The browser demo requires Node.js 20+; upgrade Node.js or use --no-demo.")
    if not args.install_only:
        check_api(dict(os.environ))
        ensure_free(config["port"])
        if demo:
            ensure_free(int(os.environ.get("S2S_DEMO_PORT", "7860")))
    python = bootstrap(profile, args.reinstall, demo)
    if args.install_only:
        print("venv installation complete; run bash run.sh again to start.")
        return 0
    if profile == "gpu":
        run([str(python), "-c", "import torch; assert torch.cuda.is_available(), 'PyTorch cannot use CUDA; check NVIDIA driver/PyTorch installation.'"])
    return supervise(python, config, demo)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Startup failed: {error}", file=sys.stderr)
        raise SystemExit(1)
