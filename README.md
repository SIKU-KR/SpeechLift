# 🎬 Video-to-Text Transcriber

**Turn any video into text in minutes.** Fast, accurate transcription powered by OpenAI Whisper API with smart Voice Activity Detection.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenAI Whisper](https://img.shields.io/badge/OpenAI-Whisper-orange.svg)](https://openai.com/research/whisper)

---

### Why This Tool?

| Problem | Solution |
|---------|----------|
| 📺 Hours of lectures, meetings, interviews to transcribe manually | **Automated transcription in minutes** |
| 💸 Transcription services are expensive | **Pay only for speech** — silence and background music are excluded |
| 🐌 Single API requests are slow | **10x parallel processing** for large videos |
| ✂️ Naive splitting breaks sentences mid-word | **Silero VAD** splits at natural speech boundaries |

### 🆚 Why This Tool Over Alternatives?

| Tool | Approach | This Tool's Advantage |
|------|----------|----------------------|
| [WhisperX](https://github.com/m-bain/whisperX) | Local model, complex setup | **Zero setup** — no GPU, no model downloads |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Local model, requires CUDA | **Works anywhere** — just Python + API key |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) | C++ compilation required | **Pure Python** — `pip install` and go |
| [WhisperLive](https://github.com/collabora/WhisperLive) | Real-time focus | **Batch optimized** — 10x parallel for long videos |

### Key Differentiators

- **☁️ API-Based** — Uses OpenAI's Whisper API, not local models. No GPU? No problem.
- **⚡ Parallel Processing** — 10 concurrent API requests vs sequential processing
- **💰 Cost Transparency** — See exact cost before you start (and only pay for speech!)
- **📦 Single File** — One `main.py`, no complex architecture to understand
- **🔌 Always Latest Model** — API updates automatically, no manual model updates

> If you have an OpenAI API key and want transcription *now* without GPU setup, this is for you.

### 💡 Perfect For

- 📚 **Students & Researchers** — Convert lecture recordings to searchable notes
- 📝 **Content Creators** — Generate subtitle drafts for YouTube videos
- 💼 **Professionals** — Turn meeting recordings into minutes
- 📰 **Journalists** — Transcribe interviews instantly
- 🎓 **Language Learners** — Read along with foreign language content

---

## 🚀 Quick Start (30 seconds)

```bash
# 1. Clone
git clone https://github.com/yourusername/video-to-text-whisper.git
cd video-to-text-whisper

# 2. Install (auto-creates virtual environment)
source setup.sh

# 3. Run
python main.py
```

On first run, enter your OpenAI API key. → [Get your API key](https://platform.openai.com/api-keys)

---

## 📋 Requirements

- Python 3.10+
- FFmpeg
- OpenAI API key

### Install FFmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg
```

---

## 📖 Usage

1. Place video files in the project folder
2. Run `python main.py`
3. Select video → Confirm cost estimate → Done
4. Find your transcript at `<video_name>_transcription.txt`

---

## ⚙️ How It Works

```
Video File
    ↓
[FFmpeg] Extract audio (16kHz mono WAV)
    ↓
[Silero VAD] Detect speech segments
    ↓
[Chunking] Split into 8-15 sec chunks at natural boundaries
    ↓
[Whisper API] Transcribe 10 chunks in parallel (with auto-retry)
    ↓
[Assembly] Merge transcriptions in order
    ↓
Save transcription.txt
```

---

## 💵 Cost

**Whisper API: $0.006/minute**

- 1-hour video ≈ $0.36
- **Speech-only billing** — silence and music don't cost a cent!

---

## 🔧 API Key Setup

On first run, choose how to save your key:

1. **Config file** (recommended) — `~/.video_transcriber_config.json`
2. **Shell profile** — Added to `.zshrc` or `.bashrc`
3. **Session only** — Enter each time

Or set via environment variable:
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

---

## 🛠 Troubleshooting

| Issue | Solution |
|-------|----------|
| "FFmpeg not found" | Install FFmpeg (see above) |
| Missing dependencies | Run `source setup.sh` again |
| API rate limits | Auto-retries with backoff. If persistent, reduce `MAX_CONCURRENT_REQUESTS` in `main.py` |
| No speech detected | Ensure video contains audible speech |

---


## 🤝 Contributing

Contributions welcome! Feel free to submit bug reports, feature requests, or pull requests.

## 📄 License

MIT License — free to use, modify, and distribute.

<p align="center">
  <b>⭐ If you find this useful, please star the repo!</b>
</p>
