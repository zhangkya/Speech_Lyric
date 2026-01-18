# 项目概述

本项目是一个命令行工具，用于从音频文件中生成歌词。它结合了多种AI模型来实现这一功能：

*   **Demucs:** 用于将人声从伴奏中分离出来。
*   **Faster-Whisper:** 用于将人声转录成带精确时间戳的文本。
*   **大语言模型 (LLM):** 用于清理转录后的歌词，移除无歌词内容，并为非中文歌曲翻译成双语格式（原文和中文）。

支持的 LLM 提供商：
*   **Google Gemini:** 云端 API，需要 API Key
*   **Ollama:** 本地运行，完全免费且保护隐私

该工具可以处理单个音频文件或包含多个音频文件的整个目录。它会生成LRC文件（一种标准的同步歌词格式），并且还可以将歌词直接写入音频文件的元数据（标签）中。

项目包含两个主要脚本：
*   `run.py`: 主脚本，可以处理中文和非中文歌曲，并包含翻译功能。
*   `run_zh.py`: 一个专门为中文歌曲优化的变体，只进行歌词清理，不执行翻译。

## 构建与运行

本项目使用Python编写，并依赖多个库。

**1. 安装依赖:**

强烈建议在虚拟环境中使用。

```bash
pip install -r requirements.txt
```
*（注意：项目中没有 `requirements.txt` 文件，但根据代码中的导入语句，需要以下依赖）*
```
torch
torchaudio
torchvision
faster-whisper
google-generativeai
python-dotenv
pydub
mutagen
demucs
```

**2. 设置环境变量:**

在项目根目录下创建一个 `.env` 文件，并添加以下内容：

```bash
# Whisper 模型配置
MODEL_SIZE=large-v2

# LLM 提供商选择
# 可选值: gemini, ollama
LLM_PROVIDER=gemini

# Gemini 配置
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
GEMINI_MODEL_NAME="gemini-2.5-flash"

# Ollama 配置（本地运行，无需 API Key）
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_MODEL="qwen2.5:7b"
```

**配置说明：**

| 变量 | 说明 | 示例值 |
|------|------|--------|
| `MODEL_SIZE` | Whisper 模型大小 | `large-v2`, `medium`, `small` |
| `LLM_PROVIDER` | LLM 提供商 | `gemini` 或 `ollama` |
| `GEMINI_API_KEY` | Google Gemini API Key | 从 Google AI Studio 获取 |
| `GEMINI_MODEL_NAME` | Gemini 模型名称 | `gemini-2.5-flash` |
| `OLLAMA_BASE_URL` | Ollama 服务地址 | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama 模型名称 | `qwen2.5:7b` |

**3. 运行工具:**

处理单个音频文件：

```bash
python run.py "path/to/your/audio.mp3"
```

处理一个目录下的所有音频文件：

```bash
python run.py "path/to/your/directory"
```

对于中文歌曲，你可以使用 `run_zh.py` 脚本以获得更专注的处理流程：

```bash
python run_zh.py "path/to/your/audio.mp3"
```

**4. 检查GPU支持**
要检查PyTorch是否已正确安装并支持CUDA，请运行：
```bash
python GPU_test.py
```

## 使用 Ollama 本地模型

Ollama 允许你在本地运行大语言模型，无需 API Key，完全免费且保护隐私。

### 安装 Ollama

1. **下载安装:** 访问 [https://ollama.com/](https://ollama.com/) 下载并安装
2. **启动服务:**
   ```bash
   ollama serve
   ```
3. **下载模型:**
   ```bash
   # 推荐中文模型 (需要约 5GB 显存)
   ollama pull qwen2.5:7b

   # 轻量级版本 (需要约 2GB 显存)
   ollama pull qwen2.5:3b

   # 通用模型
   ollama pull llama3:8b
   ```

### 配置使用 Ollama

编辑 `.env` 文件：

```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b
```

### 测试 Ollama 连接

```bash
python test_ollama.py
```

### Ollama vs Gemini 对比

| 特性 | Ollama | Gemini |
|------|--------|--------|
| 成本 | 免费（硬件成本） | 按用量计费 |
| 延迟 | 取决于本地硬件 | 网络延迟 |
| 隐私 | 完全本地 | 数据上传云端 |
| 配置复杂度 | 需安装和下载模型 | 只需 API Key |
| 质量 | 中高（7B 模型） | 高 |

## 开发约定

*   代码被组织成多个函数，每个函数负责流程中的一个特定部分（例如，分离人声、转录、清理歌词）。
*   项目使用 `argparse` 库来解析命令行参数。
*   使用 `pathlib` 库来处理文件路径。
*   使用 `mutagen` 库来读写音频文件的元数据。
*   项目依赖 `.env` 文件进行配置，并使用 `python-dotenv` 加载。
*   该工具被设计为模块化的，语言模型（LLM）提供商是可配置的。
