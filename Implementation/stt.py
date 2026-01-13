# ================================
# Speech-to-Text (Whisper EN)
# ================================

import warnings
import torch
from transformers import pipeline

# ----------------
# Suppress warnings
# ----------------
warnings.filterwarnings("ignore")

# ----------------
# Device & dtype
# ----------------
_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# ----------------
# Load model ONCE
# ----------------
print("🎙️ Loading Whisper STT model...")
_stt_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype=_DTYPE,
    device=_DEVICE,
)
print("✅ Whisper model loaded")


def speech_to_text_en(
    audio_path: str,
    language: str = "english"
) -> str:
    """
    Convert speech to English text using Whisper

    Args:
        audio_path (str): path to audio file (.wav, .mp3, etc.)
        language (str): language hint for Whisper

    Returns:
        str: recognized text (English)
    """

    result = _stt_pipe(
        audio_path,
        generate_kwargs={"language": "english"}
    )

    return result["text"]

# ตัวอย่างการใช้งาน

# audio_path = "/content/videofinal.mp4"
# en_text = speech_to_text_en(audio_path)
# print(en_text)