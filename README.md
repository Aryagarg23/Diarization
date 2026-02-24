# WhisperX Audio Transcription and Speaker Diarization

A Python tool for extracting, transcribing, and diarizing audio from video/audio files using WhisperX and Pyannote.audio. Optimized for GPUs with 16GB VRAM through sequential model loading.

## Features

- Audio extraction from video files (MP4, MOV, AVI, MKV, etc.)
- High-accuracy speech-to-text using WhisperX (large-v2 model, float16)
- Word-level timestamp alignment
- Speaker diarization with Pyannote.audio
- Sequential model loading to keep VRAM usage under 16GB
- Output format: "SPEAKER_ID: Transcribed text"

## Requirements

### Hardware
- NVIDIA GPU with 16GB+ VRAM (CUDA 12.1 compatible)
- 16GB+ system RAM
- 10GB+ free disk space

### Software
- Python 3.9+
- CUDA 12.1
- FFmpeg
- Git

**Note:** Native Linux provides optimal performance. WSL2 is recommended for Windows users.

## Installation

### 1. Install System Dependencies

**Linux/WSL2 (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install -y ffmpeg git python3-venv python3-pip
```

**Windows:**
- Install [FFmpeg](https://ffmpeg.org/download.html)
- Install [Git](https://git-scm.com/download/win)
- Install [Python 3.9+](https://www.python.org/downloads/)

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python3 -m venv whisper_env

# Activate (Linux/WSL2)
source whisper_env/bin/activate

# Activate (Windows PowerShell)
.\whisper_env\Scripts\Activate.ps1
```

### 3. Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1
pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

**Note:** You may see PyTorch version warnings during installation. These are safe to ignore - all components work correctly with PyTorch 2.5.1.

### 4. Configure Hugging Face Token

Get a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and accept terms for:
- [pyannote/speaker-diarization-3.0](https://huggingface.co/pyannote/speaker-diarization-3.0)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

**Option A: .env file (recommended)**
```bash
cp .env.example .env
# Edit .env and add: HF_TOKEN=hf_your_token_here
```

**Option B: Environment variable**
```bash
export HF_TOKEN="hf_your_token_here"
```

**Option C: Command line**
```bash
python transcribe_and_diarize.py video.mov --hf_token hf_your_token_here
```

## Usage

### Basic Usage

```bash
python transcribe_and_diarize.py input_video.mov
```

Output will be saved as `input_video.txt`.

### Advanced Options

```bash
# Custom output file
python transcribe_and_diarize.py video.mov --output transcript.txt

# Increase batch size for faster processing (requires more VRAM)
python transcribe_and_diarize.py video.mov --batch_size 32

# Specify token via command line
python transcribe_and_diarize.py video.mov --hf_token hf_xxxxx
```

### Command Line Arguments

```
positional arguments:
  media_file              Path to video or audio file

options:
  --hf_token HF_TOKEN     Hugging Face API token
  --batch_size BATCH_SIZE Batch size for transcription (default: 16)
  --output OUTPUT         Output text file path
```

## Performance

### VRAM Management

The script processes in 5 sequential stages to manage VRAM:

1. Audio Extraction (FFmpeg) - No VRAM
2. Transcription - ~10-12 GB
3. Alignment - ~3-4 GB
4. Diarization - ~6-8 GB
5. Export - Minimal VRAM

Models are loaded, used, deleted, and VRAM is cleared between stages using `gc.collect()` and `torch.cuda.empty_cache()`.

### Batch Size Recommendations

- **RTX 4060 Ti (16GB)**: Batch size 12-16
- **RTX 3090 Ti (24GB)**: Batch size 16-32
- **RTX 4090 (24GB)**: Batch size 32-48
- **A100 (80GB)**: Batch size 64+

Processing speed: Approximately 2-3x realtime with batch size 16 on RTX 3090 Ti.

## Troubleshooting

**CUDA out of memory**
```bash
python transcribe_and_diarize.py video.mov --batch_size 8
```

**FFmpeg not found**
```bash
# Linux/WSL2
sudo apt install ffmpeg

# Windows: Download from ffmpeg.org
```

**Hugging Face token error**
```bash
# Verify token is set
echo $HF_TOKEN

# Or set it
export HF_TOKEN="hf_your_token"
```

**Module not found errors**
```bash
# Verify virtual environment is activated
source whisper_env/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

## Project Structure

```
Diarization/
├── transcribe_and_diarize.py   # Main script
├── requirements.txt             # Python dependencies
├── .env                         # Your HF token (not in git)
├── .env.example                 # Template for .env
├── README.md                    # This file
└── whisper_env/                 # Virtual environment
```

## Output Format

```
SPEAKER_1: Welcome to the meeting. Today we'll discuss the quarterly results.
SPEAKER_2: Thank you. Let's start with the financial overview.
SPEAKER_1: As you can see, revenue increased by 15% this quarter.
SPEAKER_2: That's impressive. What were the main drivers?
```

## License & Attribution

- [WhisperX](https://github.com/m-bain/whisperX)
- [Whisper](https://github.com/openai/whisper) by OpenAI
- [Pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [PyTorch](https://pytorch.org/)

## Security

- Never commit `.env` file (already in `.gitignore`)
- Use `.env` file for token storage
- Rotate tokens if exposed at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
