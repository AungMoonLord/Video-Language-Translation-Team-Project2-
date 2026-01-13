import torch
import soundfile as sf
import scipy.io.wavfile as wavfile

# -----------------------------
# F5-TTS (Voice Cloning)
# -----------------------------
from f5_tts_th.tts import TTS

print("🔊 Loading F5-TTS-THAI model (v1)...")
_f5_tts = TTS(model="v1")
print("✅ F5-TTS model loaded")

F5_REFERENCE_AUDIO = "voice4.webm"
F5_REFERENCE_TEXT = "เชื่อว่าหลายๆคนต้องเคยเจอ"


def tts_f5_thai(
    text: str,
    output_path: str = "thai_f5.wav",
    reference_audio: str = F5_REFERENCE_AUDIO,
    reference_text: str = F5_REFERENCE_TEXT,
    sample_rate: int = 24000,
    step: int = 32,
    cfg: float = 2.0,
    speed: float = 1.0,
):

    try:
        wav = _f5_tts.infer(
            ref_audio=reference_audio,
            ref_text=reference_text,
            gen_text=text,
            step=step,
            cfg=cfg,
            speed=speed,
        )

        sf.write(output_path, wav, sample_rate)

        return {
            "success": True,
            "audio_path": output_path,
            "sample_rate": sample_rate,
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "audio_path": None,
            "sample_rate": None,
            "error": str(e),
        }


# -----------------------------
# MMS-TTS (Standard TTS)
# -----------------------------
from transformers import VitsModel, AutoTokenizer

print("🔊 Loading MMS-TTS-THAI model...")
_mms_model_name = "facebook/mms-tts-tha"
_mms_model = VitsModel.from_pretrained(_mms_model_name)
_mms_tokenizer = AutoTokenizer.from_pretrained(_mms_model_name)
print("✅ MMS-TTS model loaded")


def tts_mms_thai(
    text: str,
    output_path: str = "thai_mms.wav",
):

    try:
        inputs = _mms_tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            waveform = _mms_model(**inputs).waveform

        sample_rate = _mms_model.config.sampling_rate

        wavfile.write(
            output_path,
            rate=sample_rate,
            data=waveform[0].cpu().numpy(),
        )

        return {
            "success": True,
            "audio_path": output_path,
            "sample_rate": sample_rate,
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "audio_path": None,
            "sample_rate": None,
            "error": str(e),
        }


# ตัวอย่างการใช้งาน

# text = "อากาศวันนี้ดีมาก เหมาะกับการเขียนโค้ด"

# # F5 (Voice Clone)
# r1 = tts_f5_thai(text, output_path="f5.wav")

# # MMS (เร็ว เบา)
# r2 = tts_mms_thai(text, output_path="mms.wav")