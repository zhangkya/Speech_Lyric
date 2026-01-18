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
GEMINI_MODEL_NAME = os.getenv('GEMINI_MODEL_NAME', 'gemini-2.5-flash') # 默认为flash

# 定义支持的音频文件扩展名
SUPPORTED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wma'}

def configure_llm():
    """根据环境变量配置并返回LLM模型实例"""
    if LLM_PROVIDER and LLM_PROVIDER.lower() == 'gemini':
        if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
            print("警告: .env文件中未提供有效的GEMINI_API_KEY。将跳过LLM歌词清理。")
            return None
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            print(f"Gemini模型({GEMINI_MODEL_NAME})已成功配置。")
            return model
        except Exception as e:
            print(f"配置Gemini模型失败: {e}。将跳过LLM歌词清理。")
            return None
    return None

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

def clean_lyrics_bulk_with_llm(llm_model, raw_lrc_content: str, metadata: dict):
    """使用LLM一次性清理所有歌词"""
    if not llm_model:
        return raw_lrc_content

    artist = metadata.get('artist', '未知艺术家')
    title = metadata.get('title', '未知歌曲')

    prompt = f"""
    你是一个专业的歌词内容审查员。你的任务是精确地判断由“{artist}”演唱的歌曲“{title}”的歌词，删除文本中无用的部分，只保留歌词本身。
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
        print("\n正在调用LLM进行批量清理，请稍候...")
        response = llm_model.generate_content(prompt)
        cleaned_lrc = response.text.strip()

        # Remove any potential markdown code block markers
        if cleaned_lrc.startswith("```") and cleaned_lrc.endswith("```"):
            cleaned_lrc = cleaned_lrc[3:-3].strip()
        if cleaned_lrc.startswith("lrc") or cleaned_lrc.startswith("LRC"):
            cleaned_lrc = cleaned_lrc[3:].strip()

        # 找出被删除的行并打印
        original_lines = set(raw_lrc_content.strip().split('\n'))
        cleaned_lines = set(cleaned_lrc.strip().split('\n'))
        deleted_lines = original_lines - cleaned_lines
        
        if deleted_lines:
            print("--- LLM清理报告: 删除了以下几行 ---")
            for line in sorted(list(deleted_lines)):
                if line.strip():
                    print(line)
            print("---------------------------------")
        else:
            print("LLM审查完成，未发现可删除的无用歌词。")

        return cleaned_lrc
    except Exception as e:
        print(f"调用LLM进行批量歌词清理时出错: {e}。将使用原始歌词。")
        return raw_lrc_content

def format_time(seconds):
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"[{minutes:02d}:{secs:05.2f}]"

def separate_vocals(audio_path: Path):
    print(f"\n步骤 1/4: 使用 Demucs 分离人声 -> {audio_path.name}")
    try:
        command = ["demucs", "--two-stems=vocals", str(audio_path.resolve())]
        subprocess.run(command, check=True, capture_output=True, text=True)
        vocal_path = Path("separated") / "htdemucs" / audio_path.stem / "vocals.wav"
        if vocal_path.exists():
            print(f"人声文件已成功分离 -> {vocal_path}")
            return vocal_path
    except Exception as e:
        print(f"Demucs 分离人声失败: {e}")
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

def generate_lrc_file(llm_model, original_audio_path: Path, segments, metadata: dict):
    lrc_path = original_audio_path.with_suffix('.lrc')
    print(f"\n步骤 4/4: 生成 LRC 文件 (使用LLM批量清理) -> {lrc_path.name}")
    
    # 1. 生成原始LRC内容
    raw_lrc_lines = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            raw_lrc_lines.append(f"{format_time(segment.start)}{text}")
    raw_lrc_content = "\n".join(raw_lrc_lines)

    # 2. 使用LLM进行批量清理
    final_lrc_content = clean_lyrics_bulk_with_llm(llm_model, raw_lrc_content, metadata)

    # 3. 构建完整的LRC内容（包括头部信息）
    full_lrc_content = (
        f"[ar: {metadata.get('artist', '')}]\n"
        f"[al: {metadata.get('album', '')}]\n"
        f"[ti: {metadata.get('title', '')}]\n"
        f"[by: tiki]\n\n"
        f"[00:01.0]歌曲：{metadata.get('title', '')}\n"
        f"[00:02.0]演唱：{metadata.get('artist', '')}\n"
        f"{final_lrc_content}"
    )

    # 4. 写入最终文件
    with open(lrc_path, 'w', encoding='utf-8') as f:
        f.write(full_lrc_content)
        
    print("LRC 文件生成成功。")
    return full_lrc_content

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
    
    metadata = get_audio_metadata(file_path)
    
    vocal_file_path = separate_vocals(file_path)
    if not vocal_file_path: return

    # detected_lang = detect_language(whisper_model, vocal_file_path)
    detected_lang = 'mn'
    segments = transcribe_vocals(whisper_model, vocal_file_path, language=detected_lang)
    if not segments: return

    lrc_content = generate_lrc_file(llm_model, file_path, segments, metadata)
    if lrc_content:
        write_lyrics_to_tag(file_path, lrc_content)
        
    print(f"\n文件 {file_path.name} 处理完成。")

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
        print(f"找到 {len(audio_files)} 个音频文件。")
        for file_path in audio_files:
            process_file(whisper_model, llm_model, file_path)
    
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
