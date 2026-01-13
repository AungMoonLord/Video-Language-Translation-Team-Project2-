from moviepy.editor import VideoFileClip, AudioFileClip


def video_sound_editor(new_sound_path,video_path,output_path):
    """ใส่ไฟล์เสียงใหม่กลับเข้าไปใน Video"""
    # โหลดวิดีโอ
    video = VideoFileClip(video_path)

    # โหลดเสียงใหม่ และตัดให้ยาวเท่าวิดีโอ
    new_audio = AudioFileClip(new_sound_path).subclip(0, video.duration)

    # ใส่เสียงใหม่แทน
    final = video.set_audio(new_audio)

    # export
    final.write_videofile(
        output_path + "/translated video.mp4",
        fps=video.fps,
        codec="libx264",
        audio_codec="aac",
        audio=True
    )

    # ปิด resource
    video.close()
    new_audio.close()
    final.close()

