# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python command-line tool for generating lyrics from audio files. It combines multiple AI models:

1. **Demucs**: Separates vocals from accompaniment
2. **Faster-Whisper**: Transcribes vocals into text with precise timestamps
3. **Google Gemini**: Cleans transcribed lyrics, removes non-lyric content, and translates non-Chinese songs to bilingual format (original + Chinese)

The tool can process single audio files or entire directories containing multiple audio files. It generates LRC files (standard synchronized lyric format) and can write lyrics directly to audio file metadata (tags).

## Architecture

The codebase consists of modular functions organized in a linear processing pipeline:

- **Audio processing**: Uses `pydub` for audio manipulation
- **Vocal separation**: Uses Demucs model via `demucs` package
- **Speech transcription**: Uses Faster-Whisper models (`faster_whisper` package)
- **Lyric cleaning/translation**: Uses Google Gemini API (`google.generativeai` package)
- **Metadata handling**: Uses `mutagen` for reading/writing audio file metadata
- **File I/O**: Uses `pathlib` for file path handling

Key scripts:
- `run.py`: Main script for processing both Chinese and non-Chinese songs with translation
- `run_zh.py`: Optimized variant for Chinese songs (lyric cleaning only, no translation)
- `GPU_test.py`: Tests PyTorch CUDA/GPU support
- `test_song_story.py`: Test script for song story generation

## Development Commands

### Environment Setup
```bash
# Create virtual environment (recommended)
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Process single audio file
python run.py "path/to/audio.mp3"

# Process directory of audio files
python run.py "path/to/directory"

# Process Chinese songs only (no translation)
python run_zh.py "path/to/audio.mp3"
```

### Testing GPU Support
```bash
python GPU_test.py
```

## Configuration

Configuration is managed through `.env` file with these variables:
- `MODEL_SIZE`: Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`)
- `LLM_PROVIDER`: Currently only `gemini` supported
- `GEMINI_API_KEY`: Your Google Gemini API key
- `GEMINI_MODEL_NAME`: Gemini model name (default: `gemini-2.5-flash`)

## Code Structure Patterns

- **Modular design**: Each function handles a specific part of the pipeline
- **Environment-based configuration**: Uses `python-dotenv` to load `.env` file
- **Command-line interface**: Uses `argparse` for CLI argument parsing
- **Path handling**: Consistently uses `pathlib.Path` objects
- **Error handling**: Graceful fallbacks when API keys are missing
- **Metadata integration**: Supports writing lyrics to audio file ID3 tags

## Key Dependencies

- `torch`, `torchaudio`, `torchvision`: PyTorch ecosystem
- `faster-whisper`: Efficient Whisper implementation
- `google-generativeai`: Gemini API client
- `python-dotenv`: Environment variable management
- `pydub`: Audio manipulation
- `mutagen`: Audio metadata handling
- `demucs`: Source separation models
- `soundfile`: Audio file I/O

## Output Files

The tool creates:
- `.lrc` files: Standard synchronized lyric format with timestamps
- Updated audio file metadata: Lyrics stored in ID3 tags (USLT frame)
- Temporary files in `separated/` directory: Vocal separation outputs

## Development Notes

- The project is designed for batch processing of audio files
- GPU acceleration is supported via PyTorch/CUDA
- Chinese and non-Chinese songs follow different processing paths
- Translation is optional and configurable via environment variables
- The `separated/` directory stores intermediate vocal separation results