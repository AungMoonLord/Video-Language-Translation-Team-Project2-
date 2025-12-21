import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from huggingface_hub import login
import textwrap

# 1. Login และตั้งค่า Model
HF_TOKEN = "ใส่ Access Token ของ gemma" #ต้องกด Accept Licencse จาก Hugging Face ก่อน
login(token=HF_TOKEN)

# ตรวจสอบว่ามองเห็น GPU หรือไม่
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"กำลังทำงานบน: {device.upper()}")

# 2. ตั้งค่าการโหลดโมเดล
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

model_id = "google/gemma-3-4b-it"
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_id)

def translate_with_context(long_text, chunk_size=2000):
    """
    แปลข้อความยาวโดยแบ่งเป็นส่วนๆ และรักษา context
    """
    chunks = textwrap.wrap(long_text, width=chunk_size, break_long_words=False, replace_whitespace=False)
    
    full_translation = []
    
    # System prompt ที่ชัดเจนมากขึ้น - ห้ามใส่คำอธิบายหรือข้อความภาษาอังกฤษ
    chat_history = [
        {
            "role": "user",
            "content": """You are a professional Thai translator. Your task is to translate English text to Thai ONLY.

STRICT RULES:
1. Output ONLY the Thai translation - no English text at all
2. Do NOT include any notes, explanations, or meta-commentary
3. Do NOT use markdown formatting (**, ##, etc.)
4. Do NOT add phrases like "Here's the translation", "Okay", "Notes:", etc.
5. Do NOT include the original English text
6. Just provide clean, natural Thai translation

Translate naturally while maintaining consistency with specialized terms from previous parts."""
        },
        {
            "role": "assistant",
            "content": "เข้าใจแล้ว ฉันจะแปลเป็นภาษาไทยเท่านั้น โดยไม่มีคำอธิบายหรือข้อความภาษาอังกฤษใดๆ"
        }
    ]

    for i, chunk in enumerate(chunks):
        print(f"--- กำลังแปลส่วนที่ {i+1}/{len(chunks)} ---")
        
        # เพิ่มข้อความใหม่เข้าไปใน history พร้อมเน้นย้ำกฎอีกครั้ง
        chat_history.append({
            "role": "user", 
            "content": f"""Translate this to Thai ONLY. No English. No notes. No explanations. Just Thai translation:

{chunk}"""
        })

        prompt = processor.apply_chat_template(chat_history, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.2,  # ลดอุณหภูมิเพื่อให้แปลแม่นยำกว่า
                top_p=0.9
            )

        response = processor.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # ลบข้อความที่ไม่ต้องการออก (ถ้ามี)
        response = clean_translation_output(response)
        
        print(f"เสร็จส่วนที่ {i+1}")
        full_translation.append(response)

        chat_history.append({"role": "assistant", "content": response})

        # จำกัด history เพื่อประหยัด memory
        if len(chat_history) > 10: 
            chat_history = [chat_history[0], chat_history[1]] + chat_history[-6:]

    return "\n\n".join(full_translation)


def clean_translation_output(text):
    """
    ลบข้อความที่ไม่ต้องการออกจากผลลัพธ์การแปล
    """
    # รายการคำหรือวลีที่ไม่ต้องการ
    unwanted_phrases = [
        "Okay, here's the translation",
        "Here's the translation",
        "**Notes and Considerations:**",
        "**Chapter",
        "Do you want me to",
        "Let me know if",
        "To help me refine",
        "I've kept this as",
        "I've translated",
        "I've used",
        "Notes:",
        "**",
        "---"
    ]
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # ข้ามบรรทัดที่มีคำที่ไม่ต้องการ
        should_skip = False
        for phrase in unwanted_phrases:
            if phrase.lower() in line.lower():
                should_skip = True
                break
        
        # ข้ามบรรทัดที่เป็นภาษาอังกฤษล้วนๆ (มากกว่า 70%)
        if not should_skip and line.strip():
            # นับจำนวนตัวอักษรไทยในบรรทัด
            thai_chars = sum(1 for c in line if '\u0E00' <= c <= '\u0E7F')
            total_chars = sum(1 for c in line if c.isalpha())
            
            if total_chars > 0:
                thai_ratio = thai_chars / total_chars
                # ถ้ามีภาษาไทยมากกว่า 30% ถึงจะเก็บไว้
                if thai_ratio > 0.3:
                    cleaned_lines.append(line)
            elif not line.strip().replace('*', '').replace('-', '').replace('#', ''):
                # ข้ามบรรทัดที่มีแต่ markdown formatting
                continue
            else:
                cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()


# --- วิธีใช้งาน ---
LN_part2 = """Chapter 1: Ais SOS
Loki Familia's expedition has failed, and despair filled the air as reports confirmed their devastating loss on the 60th Floor. Adventurers and observers alike were struck with shock and terror as the news spread throughout the city, overwhelming them with grief and disbelief. The sight of fallen comrades, bloodied and injured, painted a grim picture in the minds of those who witnessed it.

Amidst the chaos, Raul, clutching his injured arm, insisted that they must help their comrades still trapped in the Deep Floors. The realization that brave adventurers had not returned, including healers and support units, ignited anger and sorrow. The crowd reacted with cries of rage and sorrow, and many were unable to process the reality of the situation, crumbling under the weight of despair.

As chaos erupted in Orario, Bell realized that he couldn't find Ais among the returned, causing his heart to race, and despite knowing the circumstances, he shouted her name in desperation. Just as he prepared to run forward, Eina held him back, urging him to escape the suffocating chaos."""

result = translate_with_context(LN_part2)

# บันทึกลงไฟล์
with open("translated_thai_clean.txt", "w", encoding="utf-8") as f:
    f.write(result)

print("\n=== เสร็จสิ้น ===")
print(f"บันทึกไฟล์: translated_thai_clean.txt")
print(f"\nตัวอย่างผลลัพธ์:\n{result[:500]}...")