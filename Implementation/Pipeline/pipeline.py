from Model.stt import speech_to_text_en
from Model.ttt import text_translation
from Model.tts_facebook import text_to_speech_TH
from Media_Editor.media import video_sound_editor


def processing_pipline():
    original_video_path = "D:/Implementation/Data/Code Minecraft With Python in 57 Seconds.mp4"
    
    eng_text = speech_to_text_en(original_video_path)
    th_text = text_translation(eng_text)

    # Path สำหรับเก็บไฟล์เสียงใหม่
    sound_path = "D:/Implementation/Data/translated_audio.wav"

    dummy = text_to_speech_TH(th_text,sound_path)

    # Path สำหรับ Video ต้นฉบับที่ต้องการแทรกเสียงใหม่เข้าไป
    translated_video_path = "D:/Implementation/Data/translated_Video.mp4"
    
    video_sound_editor(sound_path,translated_video_path)

