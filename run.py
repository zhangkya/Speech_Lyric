import os
import sys
import io
import argparse
import subprocess
import shutil
import time
from pathlib import Path
from dotenv import load_dotenv
from pydub import AudioSegment
from faster_whisper import WhisperModel
import google.generativeai as genai
from mutagen import File
from mutagen.id3 import USLT
from mutagen.mp4 import MP4Tags
from mutagen.flac import Picture

# 加载 .env 文件
load_dotenv()

# Whisper模型配置
MODEL_SIZE = os.getenv('MODEL_SIZE', 'large-v2')

# LLM配置
LLM_PROVIDER = os.getenv('LLM_PROVIDER')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL_NAME = os.getenv('GEMINI_MODEL_NAME', 'gemini-2.5-flash') # 默认为2.5 flash

# Ollama配置
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5:7b')

# 定义支持的音频文件扩展名
SUPPORTED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wma'}

def configure_llm():
    """根据环境变量配置并返回LLM提供者实例"""
    from llm_providers import GeminiProvider, OllamaProvider

    llm_provider = None

    if LLM_PROVIDER and LLM_PROVIDER.lower() == 'gemini':
        provider = GeminiProvider(GEMINI_API_KEY, GEMINI_MODEL_NAME)
        llm_provider = provider.configure()
        if llm_provider:
            print(f"Gemini 模型 ({GEMINI_MODEL_NAME}) 已成功配置。")
        else:
            print("警告: .env 文件中未提供有效的 GEMINI_API_KEY。将跳过 LLM 歌词清理。")

    elif LLM_PROVIDER and LLM_PROVIDER.lower() == 'ollama':
        provider = OllamaProvider(OLLAMA_BASE_URL, OLLAMA_MODEL)
        llm_provider = provider.configure()
        if llm_provider:
            print(f"Ollama 模型 ({OLLAMA_MODEL}) 已成功配置。")

    if not llm_provider:
        print("警告: 未配置有效的 LLM。将跳过歌词清理和翻译。")

    return llm_provider

def get_audio_metadata(audio_path: Path):
    """使用mutagen读取音频文件的元数据"""
    try:
        audio = File(audio_path, easy=True)
        if audio:
            artist = audio.get('artist', ['未知艺术家'])[0]
            album = audio.get('album', ['未知专辑'])[0]
            title = audio.get('title', [audio_path.stem])[0]
            print(f"读取到元数据 -> 艺术家: {artist}, 专辑: {album}, 标题: {title}")
            return {'artist': artist, 'album': album, 'title': title}
    except Exception as e:
        print(f"读取元数据失败: {e}")
    return {'artist': '未知艺术家', 'album': '未知专辑', 'title': audio_path.stem}

def _clean_llm_lrc_output(raw_text: str) -> str:
    """
    Cleans the raw text output from an LLM to extract only the LRC content.
    It removes conversational prefixes and markdown code blocks.
    """
    lines = raw_text.strip().split('\n')
    
    # Find the first line that looks like an LRC timestamp
    first_lrc_line_index = -1
    for i, line in enumerate(lines):
        # A simple check for a line starting with '[' and containing ']'
        if line.strip().startswith('[') and ']' in line:
            first_lrc_line_index = i
            break
            
    if first_lrc_line_index == -1:
        # If no LRC-like line is found, return the original text after basic cleaning
        cleaned_text = raw_text.strip()
        if cleaned_text.startswith("```") and cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[3:-3].strip()
        return cleaned_text

    # Join the lines from the first LRC line onwards
    lrc_content = '\n'.join(lines[first_lrc_line_index:])
    
    # Remove potential trailing markdown code block markers
    if lrc_content.endswith("```"):
        lrc_content = lrc_content[:-3].strip()
    elif lrc_content.endswith("``"):
        lrc_content = lrc_content[:-2].strip()

    return lrc_content

def _call_llm(llm_model, prompt: str) -> str:
    """统一的 LLM 调用接口，适配不同的 Provider"""
    try:
        # Ollama 使用 Client 对象
        if llm_model.__class__.__name__ == 'Client' or hasattr(llm_model, 'chat'):
            response = llm_model.chat(
                model=OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                stream=False
            )
            return response['message']['content']
        # Gemini 使用 GenerativeModel 对象
        elif hasattr(llm_model, 'generate_content'):
            response = llm_model.generate_content(prompt)
            return response.text
        else:
            raise RuntimeError(f"不支持的 LLM 模型类型: {type(llm_model)}")
    except Exception as e:
        raise RuntimeError(f"调用 LLM 失败: {e}")


def clean_lyrics_bulk_with_llm(llm_model, raw_lrc_content: str, metadata: dict):
    """使用LLM一次性清理所有歌词"""
    if not llm_model:
        return raw_lrc_content

    artist = metadata.get('artist', '未知艺术家')
    title = metadata.get('title', '未知歌曲')

    prompt = f"""
    你是一个专业的歌词内容审查员。你的任务是精确地判断由"{artist}"演唱的歌曲"{title}"的歌词，删除文本中无用的部分，只保留歌词本身。
    规则：
    1.  输入内容是标准的LRC格式，包含 `[mm:ss.xx]` 时间戳。
    2.  你的输出也必须保持完整的LRC格式，不得损坏或删除任何有效的时间戳。
    3.  只删除明确不属于歌词的行，例如包含 "(音乐)", "...", "♪", "纯音乐", "间奏" 等的行。
    4.  绝对不要修改、添加或重写任何属于歌词的文字。
    5.  如果一行歌词是有效的，请原样保留（包括时间戳）。
    6.  不要在输出中包含任何额外的标记，如 ```lrc 或其他代码块标记。

    请处理以下LRC格式的歌词内容：
    ---
    {raw_lrc_content}
    ---
    """
    try:
        print(f"\n正在调用 LLM ({LLM_PROVIDER}) 进行批量清理，请稍候...")
        response_text = _call_llm(llm_model, prompt)
        cleaned_lrc = _clean_llm_lrc_output(response_text)

        # 找出被删除的行并打印
        original_lines = set(raw_lrc_content.strip().split('\n'))
        cleaned_lines = set(cleaned_lrc.strip().split('\n'))
        deleted_lines = original_lines - cleaned_lines

        if deleted_lines:
            print(f"--- LLM 清理报告 (使用 {LLM_PROVIDER}): 删除了以下几行 ---")
            for line in sorted(list(deleted_lines)):
                if line.strip():
                    print(line)
            print("---------------------------------")
        else:
            print("LLM 审查完成，未发现可删除的无用歌词。")

        return cleaned_lrc
    except Exception as e:
        print(f"调用 LLM 进行批量歌词清理时出错: {e}。将使用原始歌词。")
        return raw_lrc_content

def translate_lyrics_to_bilingual(llm_model, raw_lrc_content: str, metadata: dict):
    """将非中文歌词翻译成双语歌词（原文+中文翻译）"""
    if not llm_model:
        return raw_lrc_content

    artist = metadata.get('artist', '未知艺术家')
    title = metadata.get('title', '未知歌曲')

    # 获取歌曲背景故事
    # song_background = get_song_background_story(llm_model, artist, title)

    prompt = f"""
    请将由{artist}演唱的歌曲{title}翻译成中文，从而形成双语歌词。请严格遵守以下规则：
    1.输入内容是标准的LRC格式，包含 `[mm:ss.xx]` 时间戳。
    2.**翻译质量**: 翻译应保留原意，自然流畅，像中文歌词一样能唱出来。允许意译，但不要偏离核心意思。
    2.  **格式要求**: 每一行非中文歌词，都需要在新的一行提供中文翻译。原文和译文必须使用完全相同的时间戳。
        正确的格式示例：
        `[00:23.84]Faster than a hairpin trigger`
        `[00:23.84]比扣动扳机更迅疾`
    3.  **中文处理**: 如果某一行歌词已经是中文，请保持原样，不要为其添加重复的翻译行。
    4.  **纯净输出**: 你的回答必须直接以LRC内容开始，绝对不能包含任何解释性文字、前言或markdown标记（如 ```）。

    请处理以下LRC格式的歌词内容：
    ---
    {raw_lrc_content}
    ---
    """
    try:
        print(f"\n正在调用 LLM ({LLM_PROVIDER}) 进行双语歌词翻译，请稍候...")
        response_text = _call_llm(llm_model, prompt)
        bilingual_lrc = _clean_llm_lrc_output(response_text)

        return bilingual_lrc
    except Exception as e:
        print(f"调用 LLM 进行双语歌词翻译时出错: {e}。将使用原始歌词。")
        return raw_lrc_content

# def get_song_background_story(llm_model, artist: str, title: str):
#     """通过大模型搜索歌手演唱的歌曲的背景故事"""
#     if not llm_model:
#         return "无法获取歌曲背景故事：LLM模型未配置。"

#     prompt = f"""
#     请用最多500字以内的短文为我提供由{artist}演唱的歌曲"{title}"的背景故事和创作背景。包括但不限于以下信息：
#     1. 歌曲的创作背景和灵感来源
#     4. 歌手创作这首歌时的心境和经历以及背后的故事
#     5. 歌曲所表达的主题和情感
#     请以简洁明了的方式提供这些信息，不需要过多的修饰，重点是准确性和信息的完整性。
#     """
#     try:
#         print(f"\n正在搜索歌曲'{title}'的背景故事...")
#         response = llm_model.generate_content(prompt)
#         story = response.text.strip()

#         # Remove any potential markdown code block markers
#         if story.startswith("```") and story.endswith("```"):
#             story = story[3:-3].strip()
#         print(f"背景故事如下：\n{story}")
#         return story
#     except Exception as e:
#         print(f"获取歌曲背景故事时出错: {e}")
#         return f"无法获取歌曲背景故事：{e}"

def format_time(seconds):
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"[{minutes:02d}:{secs:05.2f}]"

def separate_vocals(audio_path: Path):
    print(f"\n步骤 1/4: 使用 Demucs 分离人声 -> {audio_path.name}")
    try:
        command = ["demucs", "--two-stems=vocals", str(audio_path.resolve())]
        # 移除 text=True 和 encoding，捕获原始字节流以避免解码错误
        subprocess.run(command, check=True, capture_output=True)
        vocal_path = Path("separated") / "htdemucs" / audio_path.stem / "vocals.wav"
        if vocal_path.exists():
            print(f"人声文件已成功分离 -> {vocal_path}")
            return vocal_path
    except subprocess.CalledProcessError as e:
        # 手动解码 stderr，替换无法识别的字符
        stderr_text = e.stderr.decode('utf-8', errors='replace')
        print(f"Demucs 分离人声失败: {e}")
        print(f"--- Demucs 详细错误信息 ---\n{stderr_text}\n--------------------------")
    except Exception as e:
        print(f"Demucs 分离人声时发生未知错误: {e}")
    return None

def detect_language(model, audio_path: Path):
    print(f"\n步骤 2/4: 检测音频语言 -> {audio_path.name}")
    try:
        audio = AudioSegment.from_file(audio_path)
        snippet = audio[len(audio)//2 - 7500 : len(audio)//2 + 7500]
        buffer = io.BytesIO()
        snippet.export(buffer, format="wav")
        buffer.seek(0)
        _, info = model.transcribe(buffer, beam_size=1, vad_filter=True)
        print(f"检测到的语言: {info.language} (置信度: {info.language_probability:.2f})")
        return info.language if info.language_probability > 0.5 else None
    except Exception as e:
        print(f"语言检测失败: {e}")
    return None

def transcribe_vocals(model, vocal_path: Path, language: str = None):
    print(f"\n步骤 3/4: 识别歌词 (禁用VAD) -> {vocal_path.name}")
    try:
        segments, _ = model.transcribe(str(vocal_path), beam_size=5, language=language)
        print("歌词识别完成。")
        return list(segments)
    except Exception as e:
        print(f"转录失败: {e}")
    return None

def is_chinese(text):
    """判断文本是否主要为中文"""
    if not text:
        return False
    # 统计中文字符数量
    chinese_chars = 0
    total_chars = 0
    for char in text:
        if '\u4e00' <= char <= '\u9fff':  # 中文字符范围
            chinese_chars += 1
        total_chars += 1
    # 如果中文字符占比超过50%，则认为是中文
    return chinese_chars / total_chars > 0.5 if total_chars > 0 else False

def generate_lrc_file(llm_model, original_audio_path: Path, segments, metadata: dict, detected_lang: str = None):
    lrc_path = original_audio_path.with_suffix('.lrc')
    print(f"\n步骤 4/4: 生成 LRC 文件 -> {lrc_path.name}")
    
    # 1. 生成原始LRC内容
    raw_lrc_lines = []
    all_text = ""  # 用于语言检测
    for segment in segments:
        text = segment.text.strip()
        if text:
            raw_lrc_lines.append(f"{format_time(segment.start)}{text}")
            all_text += text + " "
    raw_lrc_content = "\n".join(raw_lrc_lines)

    # 2. 判断是否为中文歌曲
    is_chinese_song = False
    if detected_lang:
        # 如果检测到的语言是中文相关语言
        if detected_lang in ['zh', 'zh-cn', 'zh-tw', 'zh-hk', 'zh-sg']:
            is_chinese_song = True
    else:
        # 如果没有检测到语言，则根据歌词内容判断
        is_chinese_song = is_chinese(all_text)
    
    print(f"歌曲语言检测结果: {'中文歌曲' if is_chinese_song else '非中文歌曲'}")

    # 3. 处理歌词内容
    if is_chinese_song:
        # 中文歌曲保持不变，仅进行清理
        print("中文歌曲，仅进行歌词清理...")
        final_lrc_content = clean_lyrics_bulk_with_llm(llm_model, raw_lrc_content, metadata)
    else:
        # 非中文歌曲，翻译成双语歌词
        print("非中文歌曲，生成双语歌词...")
        final_lrc_content = translate_lyrics_to_bilingual(llm_model, raw_lrc_content, metadata)

    # 4. 构建完整的LRC内容（包括头部信息）
    full_lrc_content = (
        f"[ar: {metadata.get('artist', '')}]\n"
        f"[al: {metadata.get('album', '')}]\n"
        f"[ti: {metadata.get('title', '')}]\n"
        f"[by: tiki]\n\n"
        f"[00:00.00]歌曲：{metadata.get('title', '')}\n"
        f"[00:00.00]演唱：{metadata.get('artist', '')}\n"
        f"{final_lrc_content}"
    )

    # 5. 写入最终文件
    with open(lrc_path, 'w', encoding='utf-8') as f:
        f.write(full_lrc_content)
        
    print("LRC 文件生成成功。")
    return full_lrc_content, is_chinese_song

def write_lyrics_to_tag(audio_path: Path, lrc_content: str):
    """将歌词内容写入音频文件的LYRICS标签，先清除旧标签。"""
    try:
        print(f"将歌词写入文件标签 -> {audio_path.name}")
        audio = File(audio_path)
        if audio is None:
            print("无法加载音频文件，跳过写入标签。")
            return

        # 清除旧歌词
        if isinstance(audio.tags, MP4Tags):
            if '\xa9lyr' in audio.tags:
                del audio.tags['\xa9lyr']
        elif hasattr(audio, 'tags') and hasattr(audio.tags, 'delall'): # For ID3
            audio.tags.delall('USLT')
        elif 'LYRICS' in audio: # For Vorbis Comments (FLAC, OGG)
            del audio['LYRICS']

        # 添加新歌词
        if isinstance(audio.tags, MP4Tags):
            audio.tags['\xa9lyr'] = lrc_content
        elif hasattr(audio, 'tags') and hasattr(audio.tags, 'add'): # For ID3
            audio.tags.add(USLT(encoding=3, lang='eng', desc='', text=lrc_content))
        else: # For Vorbis Comments (FLAC, OGG)
            audio['LYRICS'] = lrc_content
        
        audio.save()
        print("歌词已成功写入文件标签。")

    except Exception as e:
        print(f"将歌词写入文件标签时出错: {e}")

def process_file(whisper_model, llm_model, file_path: Path):
    print(f"\n{'='*50}\n开始处理文件: {file_path.name}\n{'='*50}")
    
    # 检查LRC文件是否已存在
    lrc_path = file_path.with_suffix('.lrc')
    if lrc_path.exists():
        print(f"歌词文件 '{lrc_path.name}' 已存在，跳过处理。")
        return False # 默认返回False，表示非中文歌曲，以便在跳过时也能正常等待
        
    metadata = get_audio_metadata(file_path)
    
    vocal_file_path = separate_vocals(file_path)
    if not vocal_file_path: return False

    detected_lang = detect_language(whisper_model, vocal_file_path)
    
    segments = transcribe_vocals(whisper_model, vocal_file_path, language=detected_lang)
    if not segments: return False

    lrc_content, is_chinese_song = generate_lrc_file(llm_model, file_path, segments, metadata, detected_lang)
    if lrc_content:
        write_lyrics_to_tag(file_path, lrc_content)
        
    print(f"\n文件 {file_path.name} 处理完成。")
    return is_chinese_song

def main():
    parser = argparse.ArgumentParser(description="AI 歌词生成工具 (Demucs + Whisper + LLM)")
    parser.add_argument("path", type=str, help="音频文件或文件夹路径")
    args = parser.parse_args()

    input_path = Path(args.path)
    if not input_path.exists():
        sys.exit(f"错误: 路径不存在 -> {input_path}")

    llm_model = configure_llm()
    
    print(f"正在加载 Whisper 模型 ({MODEL_SIZE})...")
    try:
        whisper_model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
        print("Whisper模型已加载到 GPU。")
    except Exception as e:
        print(f"加载 Whisper GPU 模型失败: {e}。尝试CPU...")
        try:
            whisper_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
            print("Whisper模型已加载到 CPU。")
        except Exception as e2:
            sys.exit(f"加载 Whisper CPU 模型失败: {e2}")

    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            process_file(whisper_model, llm_model, input_path)
    elif input_path.is_dir():
        audio_files = [p for p in input_path.glob("**/*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        total_files = len(audio_files)
        print(f"找到 {total_files} 个音频文件。")
        for i, file_path in enumerate(audio_files):
            is_chinese_song = process_file(whisper_model, llm_model, file_path)
            
            # 打印处理进度
            print(f"\n处理进度: {i + 1} / {total_files}")

            if i < len(audio_files) - 1:
                if is_chinese_song:
                    print("\n中文歌曲处理完成，立即处理下一个文件...")
                else:
                    print("\n为了遵守API速率限制，等待6秒后处理下一个文件...")
                    time.sleep(6)
    
    separated_dir = Path("separated")
    if separated_dir.exists():
        print("\n正在清理 'separated' 文件夹...")
        try:
            shutil.rmtree(separated_dir)
            print("清理完成。")
        except OSError as e:
            print(f"清理 'separated' 文件夹失败: {e}")

if __name__ == "__main__":
    main()
