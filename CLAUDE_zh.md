# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 提供在使用此代码仓库中的代码时的指导。

## 项目概览

这是一款 Python 命令行工具，用于从音频文件中生成歌词。它结合了多个 AI 模型：

1.  \*\*Demucs：\*\* 能够将人声与伴奏分离
2.  \*\*Faster-Whisper：\*\* 将人声精确地转录成文本，并附带精确的时间戳
3.  \*\*Google Gemini：\*\* 清理转录歌词，去除非歌词内容，并将非中文歌曲翻译成双语格式（原文 + 中文）

该工具可以处理单个音频文件，也可以处理包含多个音频文件的整个目录。它可以生成 LRC 文件（标准的同步歌词格式），并且可以将歌词直接写入音频文件的元数据（标签）中。

## 建筑学

代码库由模块化的函数组成，这些函数按照线性处理流程进行组织

*   \*\*音频处理：\*\* 使用 \`pydub\` 库进行音频处理
*   \*\*人声分离：\*\* 使用 \`demucs\` 包中的 Demucs 模型
*   \*\*语音转文本：\*\* 使用 Faster-Whisper 模型（\`faster\_whisper\` 软件包）
*   \*\*歌词清洗/翻译：\*\* 使用 Google Gemini API ( \`google\_generativeai\` 软件包)。
*   \*\*元数据处理：\*\* 使用 \`mutagen\` 库来读取和写入音频文件的元数据
*   \*\*文件输入/输出：\*\* 使用 \`pathlib\` 模块来处理文件路径

关键脚本：

*   \`run.py\`: 主要脚本，用于处理中文和非中文歌曲，并进行翻译。
*   \`run\_zh.py\`: 针对中文歌曲的优化版本（仅进行歌词清洗，不进行翻译）。
*   \`GPU\_test.py\`: 测试 PyTorch 的 CUDA/GPU 支持。
*   \`test\_song\_story.py\`: 用于测试歌曲故事生成的脚本。

## 开发命令

### 环境配置

```bash
# Create virtual environment (recommended)
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 运行应用程序

```bash
# Process single audio file
python run.py "path/to/audio.mp3"

# Process directory of audio files
python run.py "path/to/directory"

# Process Chinese songs only (no translation)
python run_zh.py "path/to/audio.mp3"
```

### 测试 GPU 支持

```bash
python GPU_test.py
```

## 配置

配置信息通过 \`.env\` 文件进行管理，该文件包含以下变量：

*   \`MODEL\_SIZE\`: Whisper 模型大小 (选择：\`tiny\`、\`base\`、\`small\`、\`medium\`、\`large-v2\`、\`large-v3\`)
*   \`LLM\_PROVIDER\`: 目前仅支持 \`gemini\`
*   \`GEMINI\_API\_KEY\`: 您的 Google Gemini API 密钥
*   \`GEMINI\_MODEL\_NAME\`: Gemini 模型名称（默认为：\`gemini-2.5-flash\`）。

## 代码结构模式

*   \*\*模块化设计：\*\* 每个函数负责处理管道中的一个特定部分
*   \*\*基于环境的配置：\*\* 使用 \`python-dotenv\` 库加载 \`.env\` 文件。
*   \*\*命令行界面：\*\* 使用 \`argparse\` 模块进行命令行参数解析
*   \*\*路径处理：\*\* 始终使用 \`pathlib.Path\` 对象。
*   \*\*错误处理：\*\* 当 API 密钥缺失时，提供优雅的备用方案
*   \*\*元数据集成：\*\* 支持将歌词写入音频文件的 ID3 标签

## 关键依赖项

*   \`torch\`、\`torchaudio\`、\`torchvision\`：PyTorch 生态系统
*   \`faster-whisper\`: 高效的 Whisper 实现
*   \`google-generativeai\`: Gemini API 客户端
*   \`python-dotenv\`: 环境变量管理
*   \`pydub\`: 音频处理
*   \`mutagen\`: 音频元数据处理
*   \`demucs\`: 声音源分离模型
*   \`soundfile\`: 音频文件输入/输出

## 输出文件

该工具会生成：

*   \`.lrc\` 文件：标准的带有时间戳的同步歌词格式。
*   已更新音频文件的元数据：歌词已存储在 ID3 标签（USLT 帧）中
*   \`separated/\` 目录下存在临时文件：人声分离的输出结果

## 开发笔记

*   该项目旨在批量处理音频文件
*   支持通过 PyTorch/CUDA 实现 GPU 加速
*   中文歌曲和非中文歌曲的处理流程是不同的
*   翻译是可选的，并且可以通过环境变量进行配置
*   \`separated/\` 目录用于存储中间的语音分离结果