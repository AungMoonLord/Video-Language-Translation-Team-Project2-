"""แปล Text ภาษาอังกฤษเป็น Text ภาษาไทย"""

"""แปลง Text ไทย เป็นเสียงไทย"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

def text_translation(ENG_text_path):
    #โหลด Model
    model_id = "scb10x/typhoon-translate-4b"

    # โหลด Tokenizer และ Model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map={"": 0}, #บังคับใช้ GPU
)
    #ข้อความที่ต้องการแปล
    text = ENG_text_path


    # สำคัญ: ต้องใช้ System Prompt ตามที่โมเดลกำหนดเพื่อให้ผลลัพธ์แม่นยำ
    messages3 = [
        {"role": "system", "content": "Translate the following text into Thai."},
        {"role": "user", "content": text},
    ]

    # เตรียม Input
    input_ids = tokenizer.apply_chat_template(
    messages3, 
    add_generation_prompt=True, 
    return_tensors="pt"
    ).to(model.device)

    
        # สั่งให้โมเดล Generate ผลลัพธ์
    outputs = model.generate(
        input_ids, 
        max_new_tokens=512, 
        do_sample=False, # แนะนำให้ปิด sampling เพื่อความแม่นยำในการแปล
        temperature=None, # ล้างค่าที่ไม่จำเป็นออก
        top_p=None
    )
    TH_text = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
    return TH_text

