# coding=utf-8
# Qwen3-TTS Gradio UI — Colab-friendly version
# Features:
#   1. Auto-transcribe reference audio via OpenAI Whisper (like F5-TTS)
#   2. SRT subtitle export from synthesized speech
#
# Usage (local):    python app.py [--share]
# Usage (Colab):    !python app.py --share
#
# Install first:
#   pip install qwen-tts openai-whisper gradio soundfile
#   (optional, for better quality transcription: whisper model auto-downloads)

import argparse
import io
import os
import re
import tempfile
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf
import torch

# ─────────────────────────────────────────────
# Lazy-loaded globals (loaded once on first use)
# ─────────────────────────────────────────────
_tts: Any = None
_whisper_model: Any = None
_model_id: str = ""

# ─────────────────────────────────────────────
# Helpers: device / dtype detection
# ─────────────────────────────────────────────

def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _detect_dtype() -> torch.dtype:
    """Use bfloat16 on Ampere+ (A100/H100), float16 on older GPUs (T4), float32 on CPU."""
    if not torch.cuda.is_available():
        return torch.float32
    # Check compute capability
    cap = torch.cuda.get_device_capability(0)
    major = cap[0]
    if major >= 8:
        return torch.bfloat16   # A100, H100, RTX 3xxx+
    return torch.float16        # T4, V100, etc.


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

def load_tts(model_id: str) -> Any:
    global _tts, _model_id
    if _tts is None or _model_id != model_id:
        from qwen_tts import Qwen3TTSModel
        device = _detect_device()
        dtype = _detect_dtype()
        print(f"[Qwen3-TTS] Loading {model_id} on {device} with dtype={dtype} ...")
        _tts = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device,
            dtype=dtype,
            # No flash_attention_2 — incompatible with free Colab T4 without extra build
        )
        _model_id = model_id
        print("[Qwen3-TTS] Model loaded.")
    return _tts


def load_whisper(whisper_size: str = "base") -> Any:
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print(f"[Whisper] Loading '{whisper_size}' model ...")
        _whisper_model = whisper.load_model(whisper_size)
        print("[Whisper] Whisper model loaded.")
    return _whisper_model


# ─────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────

def _normalize_audio(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)
    # Gradio returns int16 from microphone; convert to float32 [-1, 1]
    if np.issubdtype(wav.dtype, np.integer):
        info = np.iinfo(wav.dtype)
        wav = wav.astype(np.float32) / max(abs(info.min), info.max)
    peak = np.max(np.abs(wav))
    if peak > 1.0 + 1e-6:
        wav = wav / (peak + 1e-12)
    return np.clip(wav, -1.0, 1.0)


def _audio_tuple(audio_in: Any) -> Optional[Tuple[np.ndarray, int]]:
    """Convert Gradio audio output (sr, wav) → (float32 wav, sr)."""
    if audio_in is None:
        return None
    if isinstance(audio_in, tuple) and len(audio_in) == 2:
        sr, wav = audio_in
        return _normalize_audio(wav), int(sr)
    return None


def _wav_to_tempfile(wav: np.ndarray, sr: int, suffix: str = ".wav") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    sf.write(path, wav, sr)
    return path


# ─────────────────────────────────────────────
# Auto-transcription (Whisper)
# ─────────────────────────────────────────────

def auto_transcribe(audio_in: Any, whisper_size: str = "base") -> Tuple[str, str]:
    """
    Transcribe reference audio using Whisper.
    Returns (transcribed_text, status_message).
    """
    try:
        at = _audio_tuple(audio_in)
        if at is None:
            return "", "⚠️ Please upload or record a reference audio first."
        wav, sr = at
        tmp = _wav_to_tempfile(wav, sr)
        try:
            wm = load_whisper(whisper_size)
            result = wm.transcribe(tmp, fp16=torch.cuda.is_available())
            text = result.get("text", "").strip()
            if not text:
                return "", "⚠️ Whisper returned empty transcription. Try a cleaner audio clip."
            return text, f"✅ Transcribed ({len(text)} chars)"
        finally:
            os.remove(tmp)
    except Exception as e:
        return "", f"❌ Transcription error: {type(e).__name__}: {e}"


# ─────────────────────────────────────────────
# SRT generation
# ─────────────────────────────────────────────

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences on '.', '!', '?' boundaries."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    # Keep non-empty
    return [p.strip() for p in parts if p.strip()]


def seconds_to_srt_time(s: float) -> str:
    """Convert float seconds to SRT timestamp: HH:MM:SS,mmm"""
    s = max(0.0, s)
    hours = int(s // 3600)
    minutes = int((s % 3600) // 60)
    secs = int(s % 60)
    millis = int(round((s - int(s)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def make_srt(text: str, duration_seconds: float) -> str:
    """
    Build an SRT subtitle string.
    Timing is proportional to character count per sentence.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return ""
    total_chars = sum(len(s) for s in sentences)
    if total_chars == 0:
        return ""

    lines = []
    current = 0.0
    for i, sentence in enumerate(sentences, start=1):
        frac = len(sentence) / total_chars
        seg_dur = duration_seconds * frac
        start_ts = seconds_to_srt_time(current)
        end_ts = seconds_to_srt_time(current + seg_dur)
        lines.append(f"{i}")
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(sentence)
        lines.append("")
        current += seg_dur
    return "\n".join(lines)


def export_srt(text: str, audio_out: Any) -> Tuple[Optional[str], str]:
    """
    Generate and save an SRT file from the last synthesised audio and target text.
    Returns (srt_file_path, status_message).
    """
    try:
        if not text or not text.strip():
            return None, "⚠️ No target text found. Generate audio first."
        if audio_out is None:
            return None, "⚠️ No audio generated yet. Click Generate first."

        at = _audio_tuple(audio_out)
        if at is None:
            return None, "⚠️ Could not read audio output."
        wav, sr = at
        duration = len(wav) / sr

        srt_content = make_srt(text.strip(), duration)
        if not srt_content:
            return None, "⚠️ Could not split text into sentences."

        fd, path = tempfile.mkstemp(suffix=".srt", prefix="qwen3tts_")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        return path, f"✅ SRT exported ({len(split_into_sentences(text))} subtitle(s), {duration:.1f}s)"
    except Exception as e:
        return None, f"❌ SRT export error: {type(e).__name__}: {e}"


# ─────────────────────────────────────────────
# TTS generation functions
# ─────────────────────────────────────────────

def run_voice_clone(
    model_id: str,
    ref_audio: Any,
    ref_text: str,
    use_xvec: bool,
    target_text: str,
    language: str,
    whisper_size: str,
) -> Tuple[Optional[Any], str]:
    try:
        if not target_text or not target_text.strip():
            return None, "⚠️ Target text is required."
        at = _audio_tuple(ref_audio)
        if at is None:
            return None, "⚠️ Reference audio is required."
        if not use_xvec and (not ref_text or not ref_text.strip()):
            return None, (
                "⚠️ Reference text is required when 'Use x-vector only' is OFF.\n"
                "Either fill in the reference text (or click Auto-Transcribe), "
                "or enable 'Use x-vector only'."
            )
        tts = load_tts(model_id)
        wavs, sr = tts.generate_voice_clone(
            text=target_text.strip(),
            language=language,
            ref_audio=at,
            ref_text=ref_text.strip() if ref_text else None,
            x_vector_only_mode=bool(use_xvec),
        )
        return (sr, wavs[0].astype(np.float32)), "✅ Generation complete."
    except Exception as e:
        return None, f"❌ {type(e).__name__}: {e}"


def run_custom_voice(
    model_id: str,
    target_text: str,
    language: str,
    speaker: str,
    instruct: str,
) -> Tuple[Optional[Any], str]:
    try:
        if not target_text or not target_text.strip():
            return None, "⚠️ Target text is required."
        if not speaker:
            return None, "⚠️ Speaker is required."
        tts = load_tts(model_id)
        wavs, sr = tts.generate_custom_voice(
            text=target_text.strip(),
            language=language,
            speaker=speaker,
            instruct=(instruct or "").strip() or None,
        )
        return (sr, wavs[0].astype(np.float32)), "✅ Generation complete."
    except Exception as e:
        return None, f"❌ {type(e).__name__}: {e}"


def run_voice_design(
    model_id: str,
    target_text: str,
    language: str,
    design_instruct: str,
) -> Tuple[Optional[Any], str]:
    try:
        if not target_text or not target_text.strip():
            return None, "⚠️ Target text is required."
        if not design_instruct or not design_instruct.strip():
            return None, "⚠️ Voice design instruction is required."
        tts = load_tts(model_id)
        wavs, sr = tts.generate_voice_design(
            text=target_text.strip(),
            language=language,
            instruct=design_instruct.strip(),
        )
        return (sr, wavs[0].astype(np.float32)), "✅ Generation complete."
    except Exception as e:
        return None, f"❌ {type(e).__name__}: {e}"


# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────

SUPPORTED_LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]

CUSTOM_VOICE_SPEAKERS = [
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
]

WHISPER_SIZES = ["tiny", "base", "small", "medium", "large"]

BASE_MODELS = [
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
]
CUSTOM_MODELS = [
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
]
DESIGN_MODELS = [
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
]


CSS = """
.gradio-container { max-width: 1100px !important; margin: 0 auto; }
.tab-nav button { font-size: 15px; font-weight: 600; }
#status-box textarea { font-size: 13px; color: #555; }
"""

HEADER_MD = """
# 🎙️ Qwen3-TTS — Auto-Transcribe + SRT Export
**Voice Clone** · **Custom Voice** · **Voice Design**  
Auto-transcribe reference audio with Whisper · Export SRT subtitles

> **Colab tip:** Run with `python app.py --share` to get a public URL.
"""


def build_app(default_whisper: str = "base") -> gr.Blocks:
    with gr.Blocks(
        theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"]),
        css=CSS,
        title="Qwen3-TTS Demo",
    ) as demo:

        gr.Markdown(HEADER_MD)

        # ── Tab 1: Voice Clone (Base model) ──────────────────────────────────
        with gr.Tab("🔁 Voice Clone"):
            gr.Markdown(
                "Upload a 3-10 second reference clip. Click **Auto-Transcribe** to fill the transcript, "
                "then enter your target text and click **Generate**."
            )
            with gr.Row():
                # Left column: inputs
                with gr.Column(scale=4):
                    vc_model_id = gr.Dropdown(
                        label="Model",
                        choices=BASE_MODELS,
                        value=BASE_MODELS[0],
                        interactive=True,
                    )
                    vc_whisper_size = gr.Dropdown(
                        label="Whisper Size (for auto-transcription)",
                        choices=WHISPER_SIZES,
                        value=default_whisper,
                        interactive=True,
                    )
                    with gr.Group():
                        gr.Markdown("### 1️⃣ Reference Audio")
                        vc_ref_audio = gr.Audio(
                            label="Reference Audio (upload or record)",
                            sources=["upload", "microphone"],
                            type="numpy",
                        )
                        with gr.Row():
                            vc_transcribe_btn = gr.Button("🎤 Auto-Transcribe", variant="secondary")
                            vc_xvec = gr.Checkbox(
                                label="Use x-vector only (no transcript needed, lower quality)",
                                value=False,
                            )
                        vc_ref_text = gr.Textbox(
                            label="Reference Text (auto-filled or type manually)",
                            lines=3,
                            placeholder="The exact words spoken in the reference audio…",
                        )

                    with gr.Group():
                        gr.Markdown("### 2️⃣ Target Text")
                        vc_target_text = gr.Textbox(
                            label="Text to Synthesize",
                            lines=5,
                            placeholder="Enter text here. Multiple sentences supported for SRT export.",
                        )
                        vc_lang = gr.Dropdown(
                            label="Language",
                            choices=SUPPORTED_LANGUAGES,
                            value="Auto",
                            interactive=True,
                        )
                    vc_gen_btn = gr.Button("🔊 Generate", variant="primary", size="lg")

                # Right column: outputs
                with gr.Column(scale=3):
                    gr.Markdown("### Output")
                    vc_audio_out = gr.Audio(label="Synthesized Audio", type="numpy")
                    vc_status = gr.Textbox(
                        label="Status",
                        lines=3,
                        interactive=False,
                        elem_id="status-box",
                    )
                    with gr.Row():
                        vc_export_btn = gr.Button("📥 Export SRT", variant="secondary")
                    vc_srt_out = gr.File(label="SRT Subtitle File", file_types=[".srt"])

            # Event handlers
            vc_transcribe_btn.click(
                fn=auto_transcribe,
                inputs=[vc_ref_audio, vc_whisper_size],
                outputs=[vc_ref_text, vc_status],
            )

            vc_gen_btn.click(
                fn=run_voice_clone,
                inputs=[vc_model_id, vc_ref_audio, vc_ref_text, vc_xvec, vc_target_text, vc_lang, vc_whisper_size],
                outputs=[vc_audio_out, vc_status],
            )

            vc_export_btn.click(
                fn=export_srt,
                inputs=[vc_target_text, vc_audio_out],
                outputs=[vc_srt_out, vc_status],
            )

        # ── Tab 2: Custom Voice ───────────────────────────────────────────────
        with gr.Tab("🎭 Custom Voice"):
            gr.Markdown(
                "Use one of the 9 built-in premium speakers. "
                "Optionally provide a style instruction (e.g. *speak happily*)."
            )
            with gr.Row():
                with gr.Column(scale=4):
                    cv_model_id = gr.Dropdown(
                        label="Model",
                        choices=CUSTOM_MODELS,
                        value=CUSTOM_MODELS[0],
                        interactive=True,
                    )
                    with gr.Row():
                        cv_speaker = gr.Dropdown(
                            label="Speaker",
                            choices=CUSTOM_VOICE_SPEAKERS,
                            value="Ryan",
                            interactive=True,
                        )
                        cv_lang = gr.Dropdown(
                            label="Language",
                            choices=SUPPORTED_LANGUAGES,
                            value="Auto",
                            interactive=True,
                        )
                    cv_instruct = gr.Textbox(
                        label="Style Instruction (Optional)",
                        lines=2,
                        placeholder='e.g. "Speak with a very happy tone." / "用特别愤怒的语气说"',
                    )
                    cv_target_text = gr.Textbox(
                        label="Text to Synthesize",
                        lines=5,
                        placeholder="Enter the text you want synthesized…",
                    )
                    cv_gen_btn = gr.Button("🔊 Generate", variant="primary", size="lg")

                with gr.Column(scale=3):
                    gr.Markdown("### Output")
                    cv_audio_out = gr.Audio(label="Synthesized Audio", type="numpy")
                    cv_status = gr.Textbox(
                        label="Status",
                        lines=3,
                        interactive=False,
                        elem_id="status-box",
                    )
                    cv_export_btn = gr.Button("📥 Export SRT", variant="secondary")
                    cv_srt_out = gr.File(label="SRT Subtitle File", file_types=[".srt"])

            cv_gen_btn.click(
                fn=run_custom_voice,
                inputs=[cv_model_id, cv_target_text, cv_lang, cv_speaker, cv_instruct],
                outputs=[cv_audio_out, cv_status],
            )
            cv_export_btn.click(
                fn=export_srt,
                inputs=[cv_target_text, cv_audio_out],
                outputs=[cv_srt_out, cv_status],
            )

        # ── Tab 3: Voice Design ───────────────────────────────────────────────
        with gr.Tab("🎨 Voice Design"):
            gr.Markdown(
                "Describe the voice you want in natural language, then generate. "
                "Great for creating unique character voices."
            )
            with gr.Row():
                with gr.Column(scale=4):
                    vd_model_id = gr.Dropdown(
                        label="Model",
                        choices=DESIGN_MODELS,
                        value=DESIGN_MODELS[0],
                        interactive=True,
                    )
                    vd_lang = gr.Dropdown(
                        label="Language",
                        choices=SUPPORTED_LANGUAGES,
                        value="Auto",
                        interactive=True,
                    )
                    vd_instruct = gr.Textbox(
                        label="Voice Description",
                        lines=4,
                        placeholder=(
                            'e.g. "Male voice, 30 years old, warm and calm, slight British accent." '
                            "/ 体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显"
                        ),
                        value="Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice.",
                    )
                    vd_target_text = gr.Textbox(
                        label="Text to Synthesize",
                        lines=5,
                        value="It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!",
                    )
                    vd_gen_btn = gr.Button("🔊 Generate", variant="primary", size="lg")

                with gr.Column(scale=3):
                    gr.Markdown("### Output")
                    vd_audio_out = gr.Audio(label="Synthesized Audio", type="numpy")
                    vd_status = gr.Textbox(
                        label="Status",
                        lines=3,
                        interactive=False,
                        elem_id="status-box",
                    )
                    vd_export_btn = gr.Button("📥 Export SRT", variant="secondary")
                    vd_srt_out = gr.File(label="SRT Subtitle File", file_types=[".srt"])

            vd_gen_btn.click(
                fn=run_voice_design,
                inputs=[vd_model_id, vd_target_text, vd_lang, vd_instruct],
                outputs=[vd_audio_out, vd_status],
            )
            vd_export_btn.click(
                fn=export_srt,
                inputs=[vd_target_text, vd_audio_out],
                outputs=[vd_srt_out, vd_status],
            )

        # Footer
        gr.Markdown(
            """
---
**Disclaimer:** Audio generated by AI for demonstration purposes only.
Do not use to replicate voices without consent or for unlawful purposes.
"""
        )

    return demo


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Gradio App (Colab-friendly)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run Gradio on (default: 7860)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind Gradio (default: 0.0.0.0)")
    parser.add_argument(
        "--whisper-size",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Default Whisper model size for auto-transcription (default: base)",
    )
    args = parser.parse_args()

    demo = build_app(default_whisper=args.whisper_size)
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
