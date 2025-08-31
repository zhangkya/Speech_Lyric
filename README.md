# 项目概述

本项目是一个命令行工具，用于从音频文件中生成歌词。它结合了多种AI模型来实现这一功能：

*   **Demucs:** 用于将人声从伴奏中分离出来。
*   **Faster-Whisper:** 用于将人声转录成带精确时间戳的文本。
*   **Google Gemini:** 用于清理转录后的歌词，移除无歌词内容，并为非中文歌曲翻译成双语格式（原文和中文）。

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

```
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
MODEL_SIZE="large-v2" 
LLM_PROVIDER="gemini"
GEMINI_MODEL_NAME="gemini-2.5-flash"
```

*   `GEMINI_API_KEY`: 你的 Google Gemini API 密钥。
*   `MODEL_SIZE`: 要使用的 Whisper 模型大小 (例如, `tiny`, `base`, `small`, `medium`, `large-v2`)。
*   `LLM_PROVIDER`: 要使用的语言模型提供商。目前仅支持 "gemini"。
*   `GEMINI_MODEL_NAME`: 要使用的具体 Gemini 模型。

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

## 开发约定

*   代码被组织成多个函数，每个函数负责流程中的一个特定部分（例如，分离人声、转录、清理歌词）。
*   项目使用 `argparse` 库来解析命令行参数。
*   使用 `pathlib` 库来处理文件路径。
*   使用 `mutagen` 库来读写音频文件的元数据。
*   项目依赖 `.env` 文件进行配置，并使用 `python-dotenv` 加载。
*   该工具被设计为模块化的，语言模型（LLM）提供商是可配置的。
