# Ubuntu 一键启动：DeepSeek 官方 API + 中文语音

此脚本使用 **venv**，不使用 Conda 或 Docker。默认由 DeepSeek 官方 API 负责文字回答；语音识别和合成在服务器运行。网页在个人电脑上使用麦克风，不需要服务器声卡。所有环境配置集中在根目录 `.env.example`；复制为 `.env` 后填写 Key 即可。百炼保留为可选服务。

## 1. 系统准备

推荐 **Ubuntu 24.04 + Python 3.12**。也支持已安装 Python 3.11 的 Ubuntu 22.04。脚本要求 Python 3.11/3.12；22.04 自带 Python 3.10 不适用，需要先自行安装 3.11 及对应 venv 包，再使用 `PYTHON_BIN=python3.11 bash run.sh`。

入口会优先寻找 `python3.12`、其次 `python3.11`、最后 `python3`；可用 `PYTHON_BIN` 指定解释器。`--check-api`、`--dry-run` 和 `--help` 可以在 Python 3.10 运行，不需要 `tomllib`，但完整模型安装仍要求 3.11/3.12。若旧版报 `ModuleNotFoundError: No module named 'tomllib'`，先 `git pull --ff-only origin test` 更新，再运行 API 检查。不要改动系统 `/usr/bin/python3` 的链接。

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-dev build-essential \
  ffmpeg libportaudio2 libsndfile1 curl ca-certificates
python3 --version
```

网页需要 **Node.js 20+ 和 npm**。如果尚未安装，可通过 NodeSource 的 Node.js 22 源安装（会添加第三方软件源；也可以按组织规范安装 Node.js）：

```bash
curl -fsSL https://deb.nodesource.com/setup_22.x -o /tmp/nodesource_setup.sh
sudo bash /tmp/nodesource_setup.sh
sudo apt install -y nodejs
node --version
npm --version
```

GPU 版还需要可用的 NVIDIA 驱动，先确认 `nvidia-smi` 正常。脚本不安装或改动驱动。48GB 显存使用 GPU 配置；文字大模型不占用本地显存。首次联网安装依赖并下载模型，需要可访问 PyPI、npm 和模型镜像，预留充足磁盘空间。

### 模型下载镜像

启动器默认在导入模型库前设置 `HF_ENDPOINT=https://hf-mirror.com`，GPU 和 CPU 两种配置的 Hugging Face Hub 模型下载均使用该镜像。同时默认设置 `HF_HUB_DISABLE_XET=1`，使用普通 HTTP 下载路径。环境变量优先于 `.env`，可以按需覆盖：

```dotenv
HF_ENDPOINT=https://hf-mirror.com
HF_HUB_DISABLE_XET=1
```

如果镜像缺少某个模型或临时不可用，可更换 `HF_ENDPOINT`，例如改为 `https://huggingface.co`。模型会保存在 Hugging Face 默认缓存目录，重复启动复用缓存。为避免 Silero VAD 在启动时另行从 GitHub 下载模型，一键启动器安装 `silero-vad==5.1.2` 并加载包内自带的 VAD 权重（`S2S_PACKAGED_SILERO=1`）。识别、合成和 Smart Turn 的 Hub 模型使用镜像。

镜像设置不改变 DeepSeek/百炼 API 地址，也不替代系统软件包、pip、npm、Git 仓库或 NLTK 资源的下载地址。使用方法参考：https://hf-mirror.com/ 。已有旧版虚拟环境会因新增依赖自动补装。

## 2. 下载 test 分支和配置 Key

```bash
git clone --branch test git@github.com:pinglanyi/speech-to-speech.git
cd speech-to-speech
cp .env.example .env
chmod 600 .env
nano .env
```

服务器没有 GitHub SSH Key 时，clone 地址改为 `https://github.com/pinglanyi/speech-to-speech.git`。

把 DeepSeek 官方 Key 填入 `.env` 的 `DEEPSEEK_API_KEY=` 后面。**不要把真实 Key 放进脚本、命令行参数或提交记录。** `.env` 已被 Git 忽略；更换 Key 只需修改此文件。环境中已 export 的变量优先于 `.env`。

默认：

```dotenv
S2S_LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=在此填写DeepSeek官方Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
DEEPSEEK_REASONING_EFFORT=none
```

默认使用 DeepSeek 官方 Chat Completions 地址，基础地址不要包含 `/chat/completions`。`deepseek-flash` 是当前官方文档中的模型名，也可在 `.env` 改为 `deepseek-v4-pro`。默认 `DEEPSEEK_REASONING_EFFORT=none` 关闭思考以减少语音等待；可改为 `low`、`high` 或 `max` 启用思考，但首次回答通常更慢。

从旧版百炼配置升级：`git pull --ff-only origin test` 不会覆盖已有 `.env`。保留现有文件，用 `nano .env` 加入上面的五项。旧 `DASHSCOPE_*` 变量在默认 DeepSeek 模式下不会使用；百炼 Key 不能当作 DeepSeek 官方 Key。不要为了更新配置覆盖自己保存的 Key。

需要切回百炼时，设置 `S2S_LLM_PROVIDER=bailian`，并填写 `.env.example` 中注释的 `DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL`、`DASHSCOPE_MODEL` 三项。百炼 Key、模型、地域/业务空间地址必须匹配。所有可选配置都在 `.env.example`。

先检查 API（会产生一次很短的模型调用，不安装模型）：

```bash
bash run.sh --check-api
```

显示 `DeepSeek API connection OK.` 后继续（百炼模式显示 `Bailian`）。401/403 检查 Key 和权限；404 检查地址和模型；429 检查额度、余额和限流。脚本不会打印 Key 或原始错误响应。

## 3. 一键启动

**48GB NVIDIA GPU：**

```bash
bash run.sh --gpu
```

使用 Qwen3-ASR 1.7B 识别中文，Qwen3-TTS CustomVoice 的 Vivian 音色合成中文。采用 PyTorch 后端，不要求上游 GGML 默认轮子的 Ubuntu 24.04/glibc 2.39/CUDA 12.8 组合。仍需可用的 PyTorch CUDA 和匹配驱动；必要时按 PyTorch 官方安装指引在 `.venv-gpu` 中安装匹配的 torch/torchaudio。

**没有 GPU：**

```bash
bash run.sh --cpu
```

使用多语言 Faster Whisper `small` + CPU int8 识别中文，OmniVoice 在 CPU 用 float32 合成中文。OmniVoice 与当前项目的 Transformers 5 依赖兼容，不依赖 CUDA 图执行。此配置可用于功能测试；必须完成一段语音合成后才开始播放，可能明显慢于实时，延迟取决于 CPU。建议先预留 16GB 系统内存并观察实际占用。这是规划建议，未在目标服务器上测量。OmniVoice 预训练权重为 CC-BY-NC，不能用于商业用途。

不指定选项会检测 `nvidia-smi`，成功则选择 GPU，否则选择 CPU。首次自动创建 `.venv-gpu` 或 `.venv-cpu`、安装依赖及网页组件，再启动后台语音服务和网页服务。两种环境相互隔离，CPU 环境优先安装 PyTorch CPU 轮子。后续启动复用虚拟环境和已下载模型。依赖跟随所下载提交的 pyproject.toml；不是跨时间的完整锁定环境。

脚本仅在此虚拟环境安装模型依赖：把 `faster-qwen3-tts[ggml]` 替换为基础 `faster-qwen3-tts`，再用 `pip install --no-deps -e .` 注册当前源码，避免安装未使用的原生 GGML 扩展。CPU 版额外安装 `faster-whisper`、`omnivoice`，不修改上游默认安装方式。

首次模型下载和预热可能耗时较长，默认等待 30 分钟，可修改 `.env` 的 `S2S_STARTUP_TIMEOUT`。启动失败会停止同一脚本管理的另一个服务；按 `Ctrl+C` 停止两个服务。

其他命令：

```bash
bash run.sh --gpu --install-only  # 只准备环境，不需要 Key，不启动服务
bash run.sh --cpu --no-demo       # 只启动语音后端，不需要 Node.js
bash run.sh --gpu --reinstall    # 重装/更新依赖后启动
bash run.sh --cpu --dry-run       # 查看不含 Key 的配置
bash run.sh --help
```

## 4. 从个人电脑测试服务器

两个服务均监听服务器回环地址。在**个人电脑**开一个终端建立隧道，保持连接：

```bash
ssh -N -L 7860:127.0.0.1:7860 -L 8765:127.0.0.1:8765 用户名@服务器IP
```

个人电脑浏览器打开 **http://localhost:7860**，允许麦克风，点击开始，用中文问问题。默认使用 WebSocket；本分支网页默认中文提示词和 Vivian 音色。如果浏览器以前保存过设置，会优先使用保存值，请在 Settings 中改回中文提示词和 Vivian。隧道必须同时转发两个端口：7860 是网页，8765 是浏览器直接连接的语音后端。浏览器访问 localhost 能使用麦克风；直接访问服务器普通 HTTP IP 地址通常不允许麦克风。

建议戴耳机。依次测试：中文提问、中文播报、继续追问、播报中插话。网页地址显示不代表语音端到端通过，以实际听到回答为准。CPU 模式需要等待合成，不应按 GPU 延迟预期判断失败。

如 `.env` 修改了端口，SSH 隧道和浏览器地址也要同步修改。端口已被占用时脚本会报错，不会自动结束其他进程。不要同时在同一工作目录启动多个实例。

## 5. 验证范围

### 启动报错排查

`AttributeError: module 'sys' has no attribute 'get_int_max_str_digits'` 是 Python 运行时问题。正式版 Python 3.11/3.12 应提供该接口；仅凭安装目录中的 `python3.11` 无法判断是否为早期预发布版本或异常安装。先检查：

```bash
python3.11 -c 'import sys; print(sys.version); print(sys.executable); print(hasattr(sys, "get_int_max_str_digits"))'
.venv-gpu/bin/python -c 'import sys; print(sys.version); print(sys.executable); print(hasattr(sys, "get_int_max_str_digits"))'
```

更新/重新安装来自可信来源的稳定版 Python 3.11（及对应的 venv 包），确认检查结果为 `True` 后重建环境。Ubuntu 24.04 可以改用系统 Python 3.12。不要通过给 `sys` 添加假函数绕过，也不要只运行 `--reinstall`，它只更新包。保留旧环境作为备份：

```bash
git pull origin test
mv .venv-gpu ".venv-gpu.backup-$(date +%Y%m%d-%H%M%S)"
PYTHON_BIN=python3.11 bash run.sh --gpu
# Ubuntu 24.04 也可用：PYTHON_BIN=python3.12 bash run.sh --gpu
```

启动器现在在安装前检查 Python 接口和正式版本状态，也会检查已有 venv，避免加载模型时才失败。CPU 使用同样步骤，将目录和选项改为 `.venv-cpu` / `--cpu`。模型缓存位于用户 Hugging Face 缓存目录，重建 venv 可继续复用它们。

`DeepFilterNet not available` 表示缺少可选增强组件，默认配置不启用它。HF_TOKEN 警告通常不阻止公开模型下载；镜像可能重定向到模型 CDN，最终以下载是否成功为准。

NLTK 的目录权限警告和 `Connection timed out` 是两个不同问题。启动器默认创建私有 `.runtime/nltk_data`，也支持在 `.env` 设置单个绝对路径 `NLTK_DATA`。自定义目录及路径中父目录不能让其他用户写入；脚本不会修改用户指定目录或父目录的权限。可使用 `namei -l /实际目录` 检查并由目录所有者处理。

NLTK 的 `punkt_tab` / `averaged_perceptron_tagger_eng` 通过其独立下载器获取，`HF_ENDPOINT` 不会代理这些资源。服务器可以联网时预下载：

```bash
mkdir -p .runtime/nltk_data
chmod 700 .runtime .runtime/nltk_data
.venv-gpu/bin/python -m nltk.downloader -d "$PWD/.runtime/nltk_data" punkt_tab averaged_perceptron_tagger_eng
```

如果仍超时，在可访问 NLTK 下载服务的机器安装相同版本 `nltk==3.10.3`，运行上面下载命令，再把生成的 `tokenizers/` 与 `taggers/` 目录复制到服务器 `.runtime/nltk_data/`。请保留解压后的子目录结构。NLTK 资源补齐后再启动；私有目录设置本身无法修复网络超时。

`npm audit` 的漏洞汇总不是此次 Python 崩溃的原因。需要查看详细报告确认受影响依赖后再更新，不要直接使用可能引入破坏性升级的 `npm audit fix --force`。

启动器的自动化测试覆盖 GPU/CPU 配置、Key 不进入配置、环境变量优先级、`.env` 不执行 shell、依赖选择及 API 错误处理。不包含实际模型推理或目标 Ubuntu/CUDA 环境的端到端验证，最终需在服务器安装后测试。

```bash
python3 -m unittest discover -s tests -p test_ubuntu_launcher.py -v
```

百炼接口参考：https://www.alibabacloud.com/help/en/model-studio/model-calling-in-sub-workspace

DeepSeek 官方接口和模型：https://api-docs.deepseek.com/zh-cn/ ，参数：https://api-docs.deepseek.com/api/create-chat-completion/ 。保留旧脚本/说明文件名以兼容现有安装，启动命令始终为 `bash run.sh`。
