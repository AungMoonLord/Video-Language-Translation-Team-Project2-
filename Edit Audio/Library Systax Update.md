 # โค้ดเก่า (v1.0.3)             โค้ดใหม่ (v2.0+)                  หน้าที่
 moviepy.editor              moviepy                         import library (from moviepy import Videoclip)
.subclip(start, end)        .subclipped(start, end)         ตัดบางส่วนของคลิป
.set_audio(audio)           .with_audio(audio)              ใส่เสียงเข้าไปในวิดีโอ
.set_duration(t)            .with_duration(t)               กำหนดความยาวคลิป
.set_fps(fps)               .with_fps(fps)                  กำหนด Frame Rate 