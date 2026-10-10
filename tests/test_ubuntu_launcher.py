"""Launcher checks without installing models or making network requests."""
import ast
import builtins
import importlib.util
import io
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("launcher", ROOT / "scripts/run_bailian.py")
        cls.launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.launcher)

    def test_launcher_import_does_not_require_tomllib(self):
        original_import = builtins.__import__

        def without_tomllib(name, *args, **kwargs):
            if name == "tomllib":
                raise ModuleNotFoundError("No module named 'tomllib'")
            return original_import(name, *args, **kwargs)

        spec = importlib.util.spec_from_file_location("launcher_without_tomllib", ROOT / "scripts/run_bailian.py")
        module = importlib.util.module_from_spec(spec)
        with patch.object(builtins, "__import__", side_effect=without_tomllib):
            spec.loader.exec_module(module)

    def test_api_check_can_run_on_python_310(self):
        with patch.object(self.launcher.sys, "version_info", (3, 10, 12)), \
             patch.object(self.launcher, "check_api") as check, \
             patch.object(self.launcher, "load_env"), \
             patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.launcher.main(["--cpu", "--check-api"]), 0)
            check.assert_called_once()

    def test_old_python_gets_actionable_install_message(self):
        with patch.object(self.launcher.sys, "version_info", (3, 10, 12)), \
             patch.object(self.launcher.sys, "platform", "linux"), \
             patch.object(self.launcher, "load_env"), \
             patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "PYTHON_BIN=python3.11"):
                self.launcher.main(["--cpu", "--install-only"])

    def test_gpu_configuration_is_chinese_and_uses_deepseek(self):
        cfg = self.launcher.pipeline_config("gpu", {})
        self.assertEqual(cfg["stt"], "qwen3-asr")
        self.assertEqual(cfg["qwen3_asr_language"], "zh")
        self.assertEqual(cfg["qwen3_tts_backend"], "torch")
        self.assertEqual(cfg["llm_backend"], "chat-completions")
        self.assertEqual(cfg["responses_api_reasoning_effort"], "none")
        self.assertFalse(cfg["responses_api_disable_thinking"])

    def test_enhancement_is_installed_for_both_profiles_but_disabled_by_default(self):
        for profile in ("gpu", "cpu"):
            self.assertIn("deepfilternet-rs==0.1.1", self.launcher.install_requirements(profile))
            self.assertFalse(self.launcher.pipeline_config(profile, {})["audio_enhancement"])
            self.assertTrue(self.launcher.pipeline_config(profile, {"S2S_AUDIO_ENHANCEMENT": "1"})["audio_enhancement"])
            self.assertFalse(self.launcher.pipeline_config(profile, {"S2S_AUDIO_ENHANCEMENT": "0"})["audio_enhancement"])
        with self.assertRaisesRegex(ValueError, "S2S_AUDIO_ENHANCEMENT"):
            self.launcher.pipeline_config("gpu", {"S2S_AUDIO_ENHANCEMENT": "typo"})

    def test_python_missing_required_sys_api_stops_before_install(self):
        with patch.object(self.launcher.sys, "platform", "linux"), \
             patch.object(self.launcher.sys, "get_int_max_str_digits", None), \
             patch.object(self.launcher, "load_env"), \
             patch.object(self.launcher, "bootstrap") as install, \
             patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "get_int_max_str_digits"):
                self.launcher.main(["--gpu", "--install-only"])
            install.assert_not_called()

    def test_cached_venv_runtime_is_checked_even_with_install_marker(self):
        import json
        broken = {"version": [3, 11, 0], "releaselevel": "candidate",
                  "get_int_max_str_digits": False, "executable": "/old/bin/python"}
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            environment = Path(folder) / ".venv-gpu"
            (environment / "bin").mkdir(parents=True)
            (environment / "bin/python").touch()
            with patch.object(self.launcher, "venv_path", return_value=environment), \
                 patch.object(self.launcher.subprocess, "check_output", return_value=json.dumps(broken)), \
                 patch.object(self.launcher, "run") as install:
                with self.assertRaisesRegex(ValueError, "get_int_max_str_digits"):
                    self.launcher.bootstrap("gpu", False, False)
                install.assert_not_called()

    def test_nltk_uses_created_private_directory_and_preserves_override(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            with patch.object(self.launcher, "ROOT", Path(folder)):
                env = {}
                self.launcher.configure_nltk_data(env)
                default = Path(env["NLTK_DATA"])
                self.assertTrue(default.is_dir())
                self.assertEqual(default, Path(folder) / ".runtime/nltk_data")
                custom = Path(folder) / "custom"
                env = {"NLTK_DATA": str(custom)}
                self.launcher.configure_nltk_data(env)
                self.assertEqual(env["NLTK_DATA"], str(custom))
                self.assertTrue(custom.is_dir())

    def test_gpu_libraries_precede_system_cuda_and_preserve_other_paths(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            site = Path(folder) / "site-packages"
            cudnn = site / "nvidia/cudnn/lib"
            runtime = site / "nvidia/cuda_runtime/lib"
            unused = site / "nvidia/unused/lib"
            for directory in (cudnn, runtime, unused):
                directory.mkdir(parents=True)
            (cudnn / "libcudnn_engines_runtime_compiled.so.9").touch()
            (runtime / "libcudart.so.13").touch()
            env = {"LD_LIBRARY_PATH": "/opt/openfoam/lib:/usr/local/cuda-12.6/lib64:"}
            original = dict(env)
            with patch.object(self.launcher.subprocess, "check_output", return_value=str(site)):
                self.launcher.configure_gpu_libraries(Path(folder) / "bin/python", env)
            expected = ":".join([str(runtime), str(cudnn), "/opt/openfoam/lib", "/usr/local/cuda-12.6/lib64"])
            self.assertEqual(env["LD_LIBRARY_PATH"], expected)
            self.assertNotIn(str(unused), env["LD_LIBRARY_PATH"])
            self.assertNotEqual(env, original)

    def test_no_packaged_gpu_libraries_preserves_existing_environment(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            env = {"LD_LIBRARY_PATH": "/custom/lib"}
            with patch.object(self.launcher.subprocess, "check_output", return_value=folder):
                self.launcher.configure_gpu_libraries(Path(folder) / "bin/python", env)
            self.assertEqual(env, {"LD_LIBRARY_PATH": "/custom/lib"})

    def test_gpu_preflight_and_backend_share_paths_cpu_does_not_configure_them(self):
        def configure(python, env):
            env["LD_LIBRARY_PATH"] = "/venv/nvidia/cudnn/lib"

        for profile in ("gpu", "cpu"):
            with self.subTest(profile=profile), \
                 patch.object(self.launcher.sys, "platform", "linux"), \
                 patch.object(self.launcher, "load_env"), \
                 patch.object(self.launcher, "check_api"), \
                 patch.object(self.launcher, "ensure_free"), \
                 patch.object(self.launcher, "bootstrap", return_value=Path("/venv/bin/python")), \
                 patch.object(self.launcher, "configure_gpu_libraries", side_effect=configure) as libraries, \
                 patch.object(self.launcher, "run") as preflight, \
                 patch.object(self.launcher, "supervise", return_value=0) as backend, \
                 patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-value"}, clear=True):
                self.assertEqual(self.launcher.main(["--" + profile, "--no-demo"]), 0)
                if profile == "gpu":
                    libraries.assert_called_once()
                    preflight.assert_called_once()
                    self.assertEqual(preflight.call_args.kwargs["env"], backend.call_args.kwargs["runtime_env"])
                    self.assertEqual(backend.call_args.kwargs["runtime_env"]["LD_LIBRARY_PATH"], "/venv/nvidia/cudnn/lib")
                else:
                    libraries.assert_not_called()
                    preflight.assert_not_called()
                self.assertNotIn("LD_LIBRARY_PATH", os.environ)

    def test_default_provider_is_deepseek_official(self):
        cfg = self.launcher.pipeline_config("gpu", {"DASHSCOPE_API_KEY": "old-key"})
        self.assertEqual(cfg["responses_api_base_url"], "https://api.deepseek.com")
        self.assertEqual(cfg["model_name"], "deepseek-flash")
        with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
            self.launcher.require_key({"DASHSCOPE_API_KEY": "old-key"})

    def test_bailian_is_still_explicitly_selectable(self):
        env = {"S2S_LLM_PROVIDER": "bailian", "DASHSCOPE_API_KEY": "test-key"}
        cfg = self.launcher.pipeline_config("cpu", env)
        self.assertEqual(cfg["responses_api_base_url"], "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(cfg["model_name"], "qwen-plus")
        self.assertEqual(self.launcher.require_key(env), "test-key")

    def test_deepseek_custom_model_and_endpoint(self):
        env = {"DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1", "DEEPSEEK_MODEL": "deepseek-v4-pro"}
        cfg = self.launcher.pipeline_config("cpu", env)
        self.assertEqual(cfg["model_name"], "deepseek-v4-pro")
        self.assertEqual(cfg["responses_api_base_url"], "https://api.deepseek.com/v1")

    def test_cpu_configuration_never_selects_cuda(self):
        cfg = self.launcher.pipeline_config("cpu", {})
        self.assertEqual(cfg["device"], "cpu")
        self.assertEqual(cfg["faster_whisper_stt_compute_type"], "int8")
        self.assertEqual(cfg["faster_whisper_stt_gen_language"], "zh")
        self.assertEqual(cfg["tts"], "omnivoice")
        self.assertEqual(cfg["omnivoice_dtype"], "float32")

    def test_secret_is_not_in_configuration(self):
        cfg = self.launcher.pipeline_config("gpu", {"DEEPSEEK_API_KEY": "secret-canary"})
        self.assertNotIn("secret-canary", str(cfg))

    def test_endpoint_and_model_can_be_overridden(self):
        cfg = self.launcher.pipeline_config("cpu", {"DEEPSEEK_BASE_URL": "https://example.org/v1", "DEEPSEEK_MODEL": "custom"})
        self.assertEqual(cfg["responses_api_base_url"], "https://example.org/v1")
        self.assertEqual(cfg["model_name"], "custom")

    def test_env_file_does_not_execute_shell(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / ".env"
            path.write_text('DASHSCOPE_API_KEY="$(touch forbidden)"\n# comment\nDASHSCOPE_MODEL=qwen-plus\n', encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.launcher.load_env(path)
                self.assertEqual(os.environ["DASHSCOPE_API_KEY"], "$(touch forbidden)")

    def test_existing_environment_takes_precedence(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / ".env"
            path.write_text("DASHSCOPE_MODEL=file-model\n", encoding="utf-8")
            with patch.dict(os.environ, {"DASHSCOPE_MODEL": "exported-model"}, clear=True):
                self.launcher.load_env(path)
                self.assertEqual(os.environ["DASHSCOPE_MODEL"], "exported-model")

    def test_missing_key_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
            self.launcher.require_key({})

    def test_base_requirements_do_not_require_native_ggml(self):
        deps = self.launcher.install_requirements("gpu")
        self.assertFalse(any("[ggml]" in dep for dep in deps))
        self.assertTrue(any(dep.startswith("faster-qwen3-tts") for dep in deps))

    def test_profiles_have_separate_virtual_environments(self):
        self.assertNotEqual(self.launcher.venv_path("cpu"), self.launcher.venv_path("gpu"))

    def test_model_downloads_default_to_mirror(self):
        env = {}
        self.launcher.configure_model_downloads(env)
        self.assertEqual(env["HF_ENDPOINT"], "https://hf-mirror.com")
        self.assertEqual(env["HF_HUB_DISABLE_XET"], "1")
        self.assertEqual(env["S2S_PACKAGED_SILERO"], "1")

    def test_model_mirror_can_be_overridden(self):
        env = {"HF_ENDPOINT": "https://huggingface.co", "HF_HUB_DISABLE_XET": "0"}
        self.launcher.configure_model_downloads(env)
        self.assertEqual(env["HF_ENDPOINT"], "https://huggingface.co")
        self.assertEqual(env["HF_HUB_DISABLE_XET"], "0")

    def test_packaged_vad_does_not_download_from_github(self):
        spec = importlib.util.spec_from_file_location("vad_loader", ROOT / "src/speech_to_speech/VAD/model_loader.py")
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        packaged = MagicMock()
        torch = MagicMock()
        with patch.dict(sys.modules, {"silero_vad": packaged, "torch": torch}), \
             patch.dict(os.environ, {"S2S_PACKAGED_SILERO": "1"}):
            self.assertIs(loader.load_silero_model(), packaged.load_silero_vad.return_value)
            torch.hub.load.assert_not_called()

    def test_api_failure_does_not_echo_provider_payload(self):
        error = urllib.error.HTTPError("https://example.org", 401, "Unauthorized", {}, io.BytesIO(b"secret-canary"))
        with patch.object(self.launcher.urllib.request, "urlopen", side_effect=error):
            with self.assertRaises(ValueError) as result:
                self.launcher.check_api({"DEEPSEEK_API_KEY": "secret-canary"})
        self.assertIn("401", str(result.exception))
        self.assertNotIn("secret-canary", str(result.exception))

    def test_api_success_validates_provider_response(self):
        response = io.BytesIO(b'{"choices":[{"message":{"content":"OK"}}]}')
        with patch.object(self.launcher.urllib.request, "urlopen", return_value=response):
            self.launcher.check_api({"DEEPSEEK_API_KEY": "test-value"})

    def test_api_check_targets_official_deepseek_with_non_thinking_payload(self):
        response = io.BytesIO(b'{"choices":[{"message":{"content":"OK"}}]}')
        with patch.object(self.launcher.urllib.request, "urlopen", return_value=response) as call:
            self.launcher.check_api({"DEEPSEEK_API_KEY": "test-value"})
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
        import json
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "deepseek-flash")
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-value")

    def test_unknown_provider_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "S2S_LLM_PROVIDER"):
            self.launcher.pipeline_config("cpu", {"S2S_LLM_PROVIDER": "typo"})

    def test_config_keys_exist_in_upstream_argument_classes(self):
        fields = set()
        for path in (ROOT / "src/speech_to_speech/arguments_classes").glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    fields.add(node.target.id)
        for profile in ("gpu", "cpu"):
            self.assertFalse(set(self.launcher.pipeline_config(profile, {})) - fields)

    def test_demo_start_failure_stops_backend_and_removes_config(self):
        backend = MagicMock()
        backend.poll.return_value = None
        response = MagicMock()
        response.__enter__.return_value.status = 200
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            with patch.object(self.launcher, "ROOT", Path(folder)), \
                 patch.object(self.launcher.subprocess, "Popen", side_effect=[backend, OSError("demo failed")]), \
                 patch.object(self.launcher.urllib.request, "urlopen", return_value=response), \
                 patch.object(self.launcher.signal, "signal"), \
                 patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-value"}):
                with self.assertRaisesRegex(OSError, "demo failed"):
                    self.launcher.supervise(Path(folder) / "bin/python", {"port": 8765}, True,
                                            runtime_env={"DEEPSEEK_API_KEY": "test-value", "LD_LIBRARY_PATH": "/venv/nvidia/cudnn/lib"})
                backend.terminate.assert_called_once()
                calls = self.launcher.subprocess.Popen.call_args_list
                self.assertEqual(calls[0].kwargs["env"]["LD_LIBRARY_PATH"], "/venv/nvidia/cudnn/lib")
                self.assertNotIn("DEEPSEEK_API_KEY", calls[1].kwargs["env"])
                self.assertFalse((Path(folder) / ".runtime/bailian.json").exists())


if __name__ == "__main__":
    unittest.main()
