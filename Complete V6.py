import os
import gc
import json
import time
import warnings
import asyncio
import nest_asyncio
import subprocess
import torch
import re
import threading
import datetime
import socket
import winsound # ✅ เพิ่ม Library สำหรับเสียงแจ้งเตือน (Windows)
from tkinter import filedialog, END, messagebox
import sys

import customtkinter as ctk

# Library สำหรับ AI
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import edge_tts
from moviepy.editor import AudioFileClip, concatenate_audioclips

# ---------------------------------------------------------
# CONFIG & LOCALIZATION
# ---------------------------------------------------------
warnings.filterwarnings("ignore")
try:
    nest_asyncio.apply()
except:
    pass

# Language Dictionary
LANG_TEXT = {
    "app_title": {"TH": "โปรแกรมแปลภาษาวิดีโอด้วย AI", "EN": "AI Video Translator"},
    "subtitle": {"TH": "ขับเคลื่อนด้วย Typhoon 4B, Whisper v3 และ Edge TTS", "EN": "Powered by Typhoon 4B & Whisper v3 & Edge TTS"},
    "hw_prefix": {"TH": "ฮาร์ดแวร์: ", "EN": "Hardware: "},
    "btn_browse": {"TH": "📂 เลือกไฟล์วิดีโอ", "EN": "📂 Browse Video"},
    "placeholder_source": {"TH": "ยังไม่ได้เลือกไฟล์...", "EN": "No file selected..."},
    "btn_save": {"TH": "💾 บันทึกเป็น...", "EN": "💾 Save As..."},
    "placeholder_target": {"TH": "กรุณาเลือกตำแหน่งบันทึกไฟล์ (Required)...", "EN": "Please select output destination (Required)..."},
    "lbl_voice": {"TH": "เลือกเสียงพากย์:", "EN": "Select Voice:"},
    "voice_male": {"TH": "ชาย (นิวัฒน์)", "EN": "Male (Niwat)"},
    "voice_female": {"TH": "หญิง (เปรมวดี)", "EN": "Female (Premwadee)"},
    "btn_start": {"TH": "🚀 เริ่มประมวลผล", "EN": "🚀 Start Processing"},
    "status_ready": {"TH": "พร้อมทำงาน", "EN": "Ready to start"},
    "status_processing": {"TH": "กำลังประมวลผล... กรุณารอสักครู่", "EN": "Processing... Please wait."},
    "status_done": {"TH": "เสร็จสมบูรณ์! 🎉", "EN": "Done! 🎉"},
    "log_header": {"TH": "บันทึกระบบ (System Log)", "EN": "System Log"},
    
    # Error Messages
    "err_no_file": {"TH": "❌ กรุณาเลือกไฟล์วิดีโอต้นฉบับก่อน", "EN": "❌ Please select a source video first."},
    "err_no_save": {"TH": "❌ กรุณาเลือกตำแหน่งบันทึกไฟล์ก่อนเริ่มงาน", "EN": "❌ Please select a save destination before starting."},
    "err_checking_net": {"TH": "🔄 กำลังตรวจสอบอินเทอร์เน็ต...", "EN": "🔄 Checking Internet Connection..."},
    "err_no_internet": {"TH": "❌ ไม่มีการเชื่อมต่ออินเทอร์เน็ต (จำเป็นสำหรับ TTS)", "EN": "❌ No Internet Connection (Required for TTS)."},
    
    # Progress Steps
    "step1": {"TH": "กำลังถอดเสียง (Whisper)...", "EN": "Transcribing audio (Whisper)..."},
    "step1_done": {"TH": "ถอดเสียงเสร็จสิ้น", "EN": "Transcription complete."},
    "step2": {"TH": "กำลังแปลภาษา (Typhoon)...", "EN": "Translating text (Typhoon)..."},
    "step2_done": {"TH": "แปลภาษาเสร็จสิ้น", "EN": "Translation complete."},
    "step3": {"TH": "กำลังสร้างเสียงพากย์ (ใช้เน็ต)...", "EN": "Generating TTS Audio (Online)..."},
    "step3_merge": {"TH": "กำลังรวมไฟล์เสียง...", "EN": "Merging audio files..."},
    "step4": {"TH": "ซิงค์ภาพและเสียง...", "EN": "Syncing video & audio..."},
    "err_general": {"TH": "เกิดข้อผิดพลาด", "EN": "Error occurred"},
    "eta": {"TH": "เวลาที่เหลือ: ", "EN": "Time Remaining: "}
}

# Global Config
DEVICE = "cpu"
TORCH_DTYPE = torch.float32
QUANTIZATION_CONFIG = None 

# Check Hardware
if torch.cuda.is_available():
    DEVICE = "cuda:0"
    gpu_props = torch.cuda.get_device_properties(0)
    total_vram_gb = gpu_props.total_memory / (1024**3)
    major_cc, minor_cc = torch.cuda.get_device_capability(0)
    cc_ver = float(f"{major_cc}.{minor_cc}")
    
    SAFE_THRESHOLD = 12.0
    if total_vram_gb < SAFE_THRESHOLD and cc_ver >= 7.0:
        TORCH_DTYPE = torch.bfloat16
        QUANTIZATION_CONFIG = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        TORCH_DTYPE = torch.bfloat16 if cc_ver >= 8.0 else torch.float16
        QUANTIZATION_CONFIG = None

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def check_internet():
    """เช็คการเชื่อมต่ออินเทอร์เน็ตโดยการ ping ไปที่ Google DNS"""
    try:
        # connect to 8.8.8.8 on port 53 (DNS) with 3s timeout
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

def play_success_sound():
    """เล่นเสียงแจ้งเตือนเมื่อทำงานเสร็จ"""
    try:
        # MB_ICONASTERISK เล่นเสียง SystemAsterisk (เสียง Ting!)
        winsound.MessageBeep(winsound.MB_ICONASTERISK)
    except:
        pass

def get_ffmpeg_path():
    # ตรวจสอบว่ารันจากไฟล์ .exe (PyInstaller) หรือรันปกติ
    if getattr(sys, 'frozen', False):
        # ถ้าเป็น .exe ให้หาในโฟลเดอร์ที่โปรแกรมอยู่
        base_path = os.path.dirname(sys.executable)
    else:
        # ถ้ารันปกติ ให้หาในโฟลเดอร์โค้ด
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(base_path, "ffmpeg.exe")

def get_ffprobe_path():
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(base_path, "ffprobe.exe")

def get_duration_ffmpeg(file_path):
    try:
        if not os.path.exists(file_path): return None
        creation_flags = 0x08000000 if os.name == 'nt' else 0
        cmd = [get_ffprobe_path(), "-v", "error", "-show_entries", "format=duration", "-of", "json", file_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=creation_flags)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except:
        return None

def smart_text_splitter(text, max_length=1500):
    paragraphs = text.split('\n')
    final_chunks = []
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para: continue
        if len(para) > max_length:
            sentences = re.split(r'(?<=[.?!])\s+', para)
        else:
            sentences = [para]
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_length:
                if current_chunk: final_chunks.append(current_chunk)
                current_chunk = sentence
            else:
                separator = " " if current_chunk and not current_chunk.endswith("\n") else ""
                current_chunk += separator + sentence
    if current_chunk: final_chunks.append(current_chunk)
    return final_chunks

# ---------------------------------------------------------
# AI MODULES
# ---------------------------------------------------------

def speech_to_text_en(audio_path, logger=None):
    if logger: logger("🎙️ [Step 1/4] Whisper Start...", "info")
    
    temp_wav = "temp_stt.wav"
    pipe = None
    try:
        subprocess.run([
            get_ffmpeg_path(), "-y", "-i", audio_path, 
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_wav
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            torch_dtype=dtype,
            device=DEVICE,
        )

        result = pipe(
            temp_wav,
            chunk_length_s=30, 
            batch_size=8, 
            return_timestamps=True,
            generate_kwargs={"language": "english"}
        )
        text = result["text"].strip()
        if logger: logger(f"   ✅ STT Done ({len(text)} chars)", "success")
        return text
    except Exception as e:
        if logger: logger(f"❌ Error STT: {e}", "error")
        return ""
    finally:
        del pipe
        cleanup_gpu()
        if os.path.exists(temp_wav): os.remove(temp_wav)

def text_translation(long_text, logger=None, progress_callback=None):
    if logger: logger("🌍 [Step 2/4] Typhoon Translation...", "info")
    
    if not long_text: return ""
    model_id = "scb10x/typhoon-translate-4b"
    model = None; tokenizer = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=TORCH_DTYPE, device_map="auto", quantization_config=QUANTIZATION_CONFIG
        )

        text_chunks = smart_text_splitter(long_text, max_length=1500)
        total_chunks = len(text_chunks)
        full_translation = []
        
        base_progress = 25
        step_range = 35 

        for i, chunk in enumerate(text_chunks):
            if not chunk.strip(): continue
            
            # Progress calculation
            current_pct = base_progress + ((i / total_chunks) * step_range)
            if progress_callback: progress_callback(current_pct, "step2")


            system_prompt = """
            You are a professional video subtitle translator. 
            Your task is to translate English text into natural, easy-to-read Thai.
            Rules:
            1. Use informal but polite spoken Thai (ภาษาพูดที่สุภาพและเป็นธรรมชาติ).
            2. Avoid literal translation. Focus on the meaning and context.
            3. Keep sentences concise to fit video timing.
            4. If technical terms (AI, Python, Code) appear, keep them in English or use standard Thai technical terms.
            """

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk}
            ]
            
            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids, max_new_tokens=1024, do_sample=True, temperature=0.3, 
                    top_p=0.9, repetition_penalty=1.15, pad_token_id=tokenizer.eos_token_id
                )
            
            translated_text = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
            full_translation.append(translated_text)
            
            if i % 10 == 0: torch.cuda.empty_cache()

        if logger: logger("   ✅ Translation Done", "success")
        return "\n".join(full_translation)

    except Exception as e:
        if logger: logger(f"❌ Error Translation: {e}", "error")
        return ""
    finally:
        del model; del tokenizer; cleanup_gpu()

def text_to_speech_TH(text, output_path, gender="Female", logger=None, progress_callback=None):
    if logger: logger(f"🗣️ [Step 3/4] TTS Generation ({gender})...", "info")
    if not text: return

    # --- VOICE SELECTION CONFIG ---
    VOICE_MAP = {
        "Female": "th-TH-PremwadeeNeural",
        "Male": "th-TH-NiwatNeural" 
    }
    VOICE = VOICE_MAP.get(gender, "th-TH-PremwadeeNeural")
    
    chunks = smart_text_splitter(text, max_length=1000)
    base_progress = 60
    step_range = 30
    total_chunks = len(chunks)

    async def _gen_sequence():
        files = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip(): continue
            fname = f"temp_tts_{i}.mp3"
            
            current_pct = base_progress + ((i / total_chunks) * step_range)
            if progress_callback: progress_callback(current_pct, "step3")

            try:
                comm = edge_tts.Communicate(chunk, VOICE, rate="-5%")
                await comm.save(fname)
                files.append(fname)
            except Exception as e:
                if logger: logger(f"     ⚠️ TTS Fail chunk {i}: {e}", "warning")
        return files

    temp_files = []
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        temp_files = loop.run_until_complete(_gen_sequence())

        if temp_files:
            if logger: logger("   🔗 Merging Audio...", "normal")
            if progress_callback: progress_callback(90, "step3_merge")
            
            clips = []
            for f in temp_files:
                try: clips.append(AudioFileClip(f))
                except: pass
            
            if clips:
                final = concatenate_audioclips(clips)
                final.write_audiofile(output_path, fps=24000, verbose=False, logger=None)
                final.close()
                for c in clips: c.close()
                if logger: logger(f"   ✅ TTS Saved", "success")
            else:
                if logger: logger("   ❌ Error: No valid audio", "error")
        else:
            if logger: logger("   ⚠️ No audio generated", "warning")

    except Exception as e:
        if logger: logger(f"❌ Error TTS: {e}", "error")
    finally:
        for f in temp_files:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass

def video_sound_editor(video_path, audio_path, output_path, logger=None):
    if logger: logger("🎬 [Step 4/4] Syncing Video...", "info")
    
    if not os.path.exists(audio_path): 
        if logger: logger("❌ Audio missing", "error")
        return

    vid_dur = get_duration_ffmpeg(video_path)
    aud_dur = get_duration_ffmpeg(audio_path)

    if not vid_dur or not aud_dur: return

    speed_factor = aud_dur / vid_dur
    speed_factor = max(0.6, min(speed_factor, 1.5)) 
    
    if logger: logger(f"   🔧 Speed Adj: {speed_factor:.2f}x", "normal")

    cmd = [
        get_ffmpeg_path(), "-y", "-i", video_path, "-i", audio_path,
        "-filter_complex", f"[1:a]atempo={speed_factor}[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-shortest", output_path
    ]
    try:
        creation_flags = 0x08000000 if os.name == 'nt' else 0
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creation_flags)
        if logger: logger(f"   ✅ Job Done! Saved: {output_path}", "success")
    except Exception as e:
        if logger: logger(f"❌ FFmpeg Error: {e}", "error")

# ---------------------------------------------------------
# GUI CLASS (MODERN & LOCALIZED)
# ---------------------------------------------------------
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class ModernApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # --- Config & Variables ---
        self.current_lang = "EN" # Default Language
        self.gender_var = ctk.StringVar(value="Female") # Default Voice
        self.video_path = None
        self.save_path = None
        self.start_time = None
        self.is_running = False

        # Window Setup
        self.title(LANG_TEXT["app_title"][self.current_lang])
        self.geometry("900x700")
        self.minsize(800, 650)
        
        # Grid Configuration
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(3, weight=1) # Console expands
        
        # --- UI COMPONENTS ---
        self._create_header()
        self._create_input_section()
        self._create_progress_section()
        self._create_console_section()

        # Init Text
        self.update_ui_text()

    def get_text(self, key):
        return LANG_TEXT.get(key, {}).get(self.current_lang, key)

    def update_ui_text(self):
        # Update Header
        self.title(self.get_text("app_title"))
        self.lbl_title.configure(text=self.get_text("app_title"))
        self.lbl_subtitle.configure(text=self.get_text("subtitle"))
        self.lbl_hw.configure(text=self.get_text("hw_prefix") + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))
        
        # Update Buttons & Inputs
        self.btn_browse.configure(text=self.get_text("btn_browse"))
        if not self.video_path:
            self.lbl_source.configure(state="normal")
            self.lbl_source.configure(placeholder_text=self.get_text("placeholder_source"))
            self.lbl_source.configure(state="readonly")
            
        self.btn_save.configure(text=self.get_text("btn_save"))
        if not self.save_path:
            self.lbl_target.configure(state="normal")
            self.lbl_target.configure(placeholder_text=self.get_text("placeholder_target"))
            self.lbl_target.configure(state="readonly")

        self.btn_start.configure(text=self.get_text("btn_start"))
        
        # Update Voice Section
        self.lbl_voice_title.configure(text=self.get_text("lbl_voice"))
        self.voice_seg_btn.configure(values=[self.get_text("voice_female"), self.get_text("voice_male")])
        current_val = self.get_text("voice_female") if self.gender_var.get() == "Female" else self.get_text("voice_male")
        self.voice_seg_btn.set(current_val)

        # Update Status & Console Header
        if not self.is_running:
            self.lbl_status.configure(text=self.get_text("status_ready"))
        self.lbl_console_header.configure(text=self.get_text("log_header"))

    def toggle_language(self, value):
        self.current_lang = value 
        self.update_ui_text()

    def select_voice(self, value):
        if value == self.get_text("voice_female"):
            self.gender_var.set("Female")
        else:
            self.gender_var.set("Male")

    def _create_header(self):
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="ew")
        
        self.lbl_title = ctk.CTkLabel(header_frame, text="", font=("Roboto Medium", 24))
        self.lbl_title.pack(side="left")
        
        self.lbl_subtitle = ctk.CTkLabel(header_frame, text="", font=("Roboto", 12), text_color="gray")
        self.lbl_subtitle.pack(side="left", padx=10, pady=(10, 0))

        right_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        right_frame.pack(side="right")

        self.lang_switch = ctk.CTkSegmentedButton(right_frame, values=["TH", "EN"], command=self.toggle_language, width=80)
        self.lang_switch.set(self.current_lang) 
        self.lang_switch.pack(side="top", pady=(0, 5))

        self.lbl_hw = ctk.CTkLabel(right_frame, text="", font=("Consolas", 11), text_color="#4B8BBE")
        self.lbl_hw.pack(side="top")

    def _create_input_section(self):
        input_frame = ctk.CTkFrame(self, corner_radius=10)
        input_frame.grid(row=1, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        
        self.btn_browse = ctk.CTkButton(input_frame, text="", command=self.browse_source, width=150, height=35)
        self.btn_browse.grid(row=0, column=0, padx=15, pady=15)
        self.lbl_source = ctk.CTkEntry(input_frame, width=400, state="readonly")
        self.lbl_source.grid(row=0, column=1, columnspan=2, padx=10, sticky="ew")
        
        self.btn_save = ctk.CTkButton(input_frame, text="", command=self.browse_target, width=150, height=35, fg_color="#E07A5F", hover_color="#D06A4F")
        self.btn_save.grid(row=1, column=0, padx=15, pady=(0, 15))
        self.lbl_target = ctk.CTkEntry(input_frame, width=400, state="readonly")
        self.lbl_target.grid(row=1, column=1, columnspan=2, padx=10, pady=(0, 15), sticky="ew")

        voice_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        voice_frame.grid(row=2, column=0, columnspan=2, padx=15, pady=(0, 15), sticky="w")

        self.lbl_voice_title = ctk.CTkLabel(voice_frame, text="", font=("Roboto", 14))
        self.lbl_voice_title.pack(side="left", padx=(0, 10))

        self.voice_seg_btn = ctk.CTkSegmentedButton(voice_frame, command=self.select_voice)
        self.voice_seg_btn.pack(side="left")

        self.btn_start = ctk.CTkButton(input_frame, text="", command=self.start_thread, width=150, height=50, font=("Roboto Bold", 16), fg_color="#2A9D8F", hover_color="#21867A")
        self.btn_start.grid(row=2, column=2, padx=15, pady=(0, 15), sticky="e")
        
        input_frame.grid_columnconfigure(1, weight=1)

    def _create_progress_section(self):
        prog_frame = ctk.CTkFrame(self, fg_color="transparent")
        prog_frame.grid(row=2, column=0, columnspan=2, padx=20, pady=5, sticky="ew")
        
        self.lbl_status = ctk.CTkLabel(prog_frame, text="", font=("Roboto", 14), text_color="#A8DADC")
        self.lbl_status.pack(side="top", anchor="w")
        
        self.progressbar = ctk.CTkProgressBar(prog_frame, height=15)
        self.progressbar.set(0)
        self.progressbar.pack(fill="x", pady=5)
        
        self.lbl_percent = ctk.CTkLabel(prog_frame, text="0%", font=("Roboto Bold", 12))
        self.lbl_percent.pack(side="left")
        
        self.lbl_eta = ctk.CTkLabel(prog_frame, text="ETA: --:--", font=("Roboto", 12), text_color="gray")
        self.lbl_eta.pack(side="right")

    def _create_console_section(self):
        console_frame = ctk.CTkFrame(self, corner_radius=10)
        console_frame.grid(row=3, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")
        
        self.lbl_console_header = ctk.CTkLabel(console_frame, text="", font=("Roboto", 12, "bold"))
        self.lbl_console_header.pack(anchor="w", padx=10, pady=5)
        
        self.console_box = ctk.CTkTextbox(console_frame, font=("Consolas", 12), state="disabled")
        self.console_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        self.console_box.tag_config("info", foreground="#A8DADC")
        self.console_box.tag_config("normal", foreground="#F1FAEE")
        self.console_box.tag_config("success", foreground="#95D5B2")
        self.console_box.tag_config("warning", foreground="#FFE8D6")
        self.console_box.tag_config("error", foreground="#FF6B6B")

    # --- FUNCTIONS ---

    def log(self, message, level="normal"):
        timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
        full_msg = f"{timestamp} {message}\n"
        self.console_box.configure(state="normal")
        self.console_box.insert(END, full_msg, level)
        self.console_box.see(END)
        self.console_box.configure(state="disabled")

    def update_progress(self, percent, status_key=None):
        self.after(0, lambda: self._update_gui_progress(percent, status_key))

    def _update_gui_progress(self, percent, status_key):
        self.progressbar.set(percent / 100)
        self.lbl_percent.configure(text=f"{int(percent)}%")
        
        if status_key:
            text = self.get_text(status_key)
            if text == status_key and " " in status_key: 
                 self.lbl_status.configure(text=status_key)
            else:
                 self.lbl_status.configure(text=text)
            
        if self.start_time and percent > 0:
            elapsed = time.time() - self.start_time
            estimated_total = elapsed / (percent / 100)
            remaining = estimated_total - elapsed
            eta_str = str(datetime.timedelta(seconds=int(remaining)))
            prefix = self.get_text("eta")
            self.lbl_eta.configure(text=f"{prefix}{eta_str}")

    def browse_source(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")])
        if path:
            self.video_path = path
            self.lbl_source.configure(state="normal")
            self.lbl_source.delete(0, END)
            self.lbl_source.insert(0, path)
            self.lbl_source.configure(state="readonly")
            
            # --- ✅ FIX: REMOVED AUTO SAVE PATH ---
            # User must select manually

    def browse_target(self):
        path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 File", "*.mp4")])
        if path:
            self.save_path = path
            self.lbl_target.configure(state="normal")
            self.lbl_target.delete(0, END)
            self.lbl_target.insert(0, path)
            self.lbl_target.configure(state="readonly")

    def start_thread(self):
        if self.is_running: return
        
        # --- ✅ CHECK 1: FILE ---
        if not self.video_path or not os.path.exists(self.video_path):
            self.log(self.get_text("err_no_file"), "error")
            return
        
        # --- ✅ CHECK 2: SAVE PATH (STRICT MANUAL CHECK) ---
        if not self.save_path:
            self.log(self.get_text("err_no_save"), "error")
            messagebox.showerror("Missing Path", self.get_text("err_no_save")) # แจ้งเตือน Popup ด้วย
            return

        # --- ✅ CHECK 3: INTERNET (WITH GUI NOTIFICATION) ---
        self.log(self.get_text("err_checking_net"), "info")
        self.update() # Force GUI Update before freezing in check
        
        if not check_internet():
            error_text = self.get_text("err_no_internet")
            self.log(error_text, "error")
            messagebox.showerror("No Internet", error_text)
            return

        self.is_running = True
        self.btn_start.configure(state="disabled", fg_color="gray")
        self.btn_browse.configure(state="disabled")
        self.btn_save.configure(state="disabled")
        self.lang_switch.configure(state="disabled")
        self.voice_seg_btn.configure(state="disabled")
        
        self.start_time = time.time()
        self.log("🚀 Pipeline Started...", "info")
        self.lbl_status.configure(text=self.get_text("status_processing"))
        
        selected_gender = self.gender_var.get()
        
        threading.Thread(target=self.run_pipeline, args=(selected_gender,), daemon=True).start()

    def run_pipeline(self, gender):
        try:
            base_dir = os.path.dirname(self.save_path)
            audio_temp_path = os.path.join(base_dir, "temp_dubbing_audio.wav")
            
            def gui_logger(msg, level="normal"):
                self.after(0, lambda: self.log(msg, level))
            
            def gui_progress(pct, key):
                self.update_progress(pct, key)

            # STEP 1: STT
            gui_progress(5, "step1")
            eng_text = speech_to_text_en(self.video_path, gui_logger)
            if not eng_text: raise Exception("STT Failed")
            gui_progress(25, "step1_done")
            
            # STEP 2: Translation
            gui_progress(25, "step2") 
            th_text = text_translation(eng_text, gui_logger, gui_progress)
            if not th_text: raise Exception("Translation Failed")
            gui_progress(60, "step2_done")
            
            # STEP 3: TTS
            gui_progress(60, "step3")
            text_to_speech_TH(th_text, audio_temp_path, gender, gui_logger, gui_progress)
            if not os.path.exists(audio_temp_path): raise Exception("TTS Failed")
            gui_progress(90, "step4")
            
            # STEP 4: Dubbing
            video_sound_editor(self.video_path, audio_temp_path, self.save_path, gui_logger)
            
            if os.path.exists(audio_temp_path):
                try: os.remove(audio_temp_path)
                except: pass
            
            gui_progress(100, "status_done")
            gui_logger(f"✨ Success! Saved: {self.save_path}", "success")
            
            # --- ✅ PLAY SOUND ON SUCCESS ---
            play_success_sound()
            
        except Exception as e:
            gui_logger(f"💀 CRITICAL ERROR: {e}", "error")
            gui_progress(0, "err_general")
        finally:
            self.is_running = False
            self.after(0, lambda: self.btn_start.configure(state="normal", fg_color="#2A9D8F"))
            self.after(0, lambda: self.btn_browse.configure(state="normal"))
            self.after(0, lambda: self.btn_save.configure(state="normal"))
            self.after(0, lambda: self.lang_switch.configure(state="normal"))
            self.after(0, lambda: self.voice_seg_btn.configure(state="normal"))
            self.after(0, lambda: self.lbl_status.configure(text=self.get_text("status_ready")))

if __name__ == "__main__":
    app = ModernApp()
    app.mainloop()
