# -----------------------------
# MMS-TTS (Standard TTS)
# -----------------------------
import torch
import scipy.io.wavfile as wavfile
from transformers import VitsModel, AutoTokenizer

print("🔊 Loading MMS-TTS-THAI model...")
_mms_model_name = "facebook/mms-tts-tha"
_mms_model = VitsModel.from_pretrained(_mms_model_name)
_mms_tokenizer = AutoTokenizer.from_pretrained(_mms_model_name)
print("✅ MMS-TTS model loaded")


def text_to_speech_TH(
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


# # ตัวอย่างการใช้งาน

text = "อากาศวันนี้ดีมาก เหมาะกับการเขียนโค้ด"

# # MMS (เร็ว เบา)
# r2 = text_to_speech_TH(text, output_path="mms.wav")
r2 = text_to_speech_TH(text, output_path="D:/Team Project 2/Video-Language-Translation-Team-Project2-/Implementation/Model/mms.mp3")
print(r2["audio_path"])