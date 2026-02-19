# Running Qwen3-TTS on Google Colab (Free Tier)

This guide shows you how to use the **enhanced Gradio UI** (`app.py`) — with **auto-transcription** and **SRT export** — on free Google Colab.

---

## 1. Open a Colab Notebook with GPU

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create a new notebook
3. Select **Runtime → Change runtime type → GPU** (choose **T4**)

---

## 2. Clone the Repository & Install Dependencies

Paste this into a Colab cell and run it:

```python
# Clone
!git clone https://github.com/YOUR_USERNAME/Qwen3-TTS.git
%cd Qwen3-TTS

# Install dependencies (Colab-optimised)
!pip install -r requirements_colab.txt -q
```

> **Note:** Replace `YOUR_USERNAME` with your GitHub username after pushing the repo.

---

## 3. Launch the Gradio App

```python
!python app.py --share
```

Gradio will print a **public URL** like:
```
Running on public URL: https://xxxxxxxx.gradio.live
```

Open that link in any browser (on your phone, another computer, etc.).

---

## 4. Using the UI

### 🔁 Voice Clone Tab (Recommended for Colab)

| Step | Action |
|------|--------|
| 1 | Upload a 3–10 second reference audio clip (clear speech, minimal background noise) |
| 2 | Click **🎤 Auto-Transcribe** — Whisper will transcribe the reference automatically |
| 3 | Review/edit the transcribed text if needed |
| 4 | Enter your target text in the **Text to Synthesize** box |
| 5 | Select language (or leave as **Auto**) |
| 6 | Click **🔊 Generate** |
| 7 | Click **📥 Export SRT** to download a subtitle file |

### 🎭 Custom Voice Tab

Choose from 9 built-in speakers (Vivian, Ryan, Aiden, etc.) and optionally add a style instruction.

### 🎨 Voice Design Tab

Describe the voice in natural language (e.g. *"warm male voice, 35 years old, slight British accent"*).

---

## 5. SRT Export

The exported `.srt` file looks like:

```
1
00:00:00,000 --> 00:00:02,150
First sentence of your text.

2
00:00:02,150 --> 00:00:04,800
Second sentence here.
```

Timing is proportional to the character count of each sentence relative to the total audio duration.

---

## 6. Tips for Best Results

- **Reference audio**: 5–10 seconds, clear speech, same language as target text
- **Auto-Transcribe model size**: `base` is fast; use `small` or `medium` for non-English or accented speech
- **Language field**: Set explicitly (e.g. `English`, `Chinese`) for best quality
- **x-vector only mode**: Tick this if you don't have the reference transcript — quality will be slightly lower but it still works

---

## 7. Memory Usage (Free T4 GPU)

| Model | VRAM |
|-------|------|
| Qwen3-TTS-12Hz-0.6B-Base | ~4 GB |
| Qwen3-TTS-12Hz-1.7B-Base | ~7–8 GB |

The `1.7B-Base` model fits comfortably in the T4's 15 GB VRAM.

---

## 8. Session Limits

Free Colab sessions disconnect after ~12 hours. The model will need to re-download/re-load each new session. Consider using **Colab Pro** or caching the model weights to Google Drive for faster restarts.
