# ================================
# Thai Text-to-Speech (F5-TTS-THAI)
# ================================

from f5_tts_th.tts import TTS
import soundfile as sf
from IPython.display import Audio, display

# ----------------
# 1. โหลดโมเดล (ครั้งเดียว)
# ----------------
print("Loading TTS model (v1)...")
tts = TTS(model="v1")   # v1 = เสียงไทยเป็นธรรมชาติ
print("Model loaded successfully!")

# ----------------------------------------
# 2. ตั้งค่าเสียงอ้างอิง (Voice Clone)
# ----------------------------------------
REFERENCE_AUDIO = "voice4.webm"
REFERENCE_TEXT = "เชื่อว่าหลายๆคนต้องเคยเจอ"

# ------------------------------------------------
# 3. ฟังก์ชันสร้างเสียง (ปรับจูนได้)
# ------------------------------------------------
def generate_thai_voice(
    text_to_speak,
    output_filename,
    sample_rate=24000,
    step=32,
    cfg=2.0,
    speed=1.0
):
    print(f"\n🎙️ กำลังสร้างเสียง:\n{text_to_speak}\n")

    try:
        wav = tts.infer(
            ref_audio=REFERENCE_AUDIO,
            ref_text=REFERENCE_TEXT,
            gen_text=text_to_speak,
            step=step,
            cfg=cfg,
            speed=speed
        )

        sf.write(output_filename, wav, sample_rate)
        print(f"✅ สร้างไฟล์เสียงสำเร็จ: {output_filename}")

        display(Audio(output_filename))
        return True

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        return False

text_1 = (
    "อากาศสดชื่น ธรรมชาติอุดมสมบูรณ์ "
    "รุ่งรวยวัฒนธรรม ท่ามกลางโลกสมัยใหม่ที่พัฒนาอย่างรวดเร็ว"
)

generate_thai_voice(
    text_to_speak=text_1,
    output_filename="thai_voice1_v1.wav",
    step=32,
    cfg=2.0,
    speed=1.0
)

