1. เอาไฟล์วิดิโอภาษาอังกฤษมาใส่ในโค้ด เพื่อดึงเสียง
    - Edit Audio (ใช้โมเดล openai/whisper-large-v3)
    > Input: ไฟล์วิดิโอ (ภาษาอังกฤษ)
    > Output: ไฟล์เสียง (ภาษาอังกฤษ)

2. ส่งไฟล์เสียงจาก (1.) เข้าไปที่
    - Speech to Text (ENG) > ใช้โมเดล openai/whisper-large-v3
    > Input: ไฟล์เสียง (ภาษาอังกฤษ)
    > Output: ข้อความ (ภาษาอังกฤษ)
    
### ขั้นตอนที่ 1-2 เป็นขั้นตอนเดียวกัน

3. ส่ง Text จาก (2.) เข้าไปที่
    - Translate Text to Text (ENG-TH) > ใช้โมเดล typhoon-translate-4b
    > Input: ข้อความ (ภาษาอังกฤษ) 
    > Output: ข้อความ (ภาษาไทย)

4. ส่ง Text จาก (3.) เข้าไปที่
    - Text to Speech (Thai) > ใช้โมเดล mms-tts-tha
    > Input: ข้อความ (ภาษาไทย)
    > Output: ไฟล์เสียง (ภาษาไทย)

5. ส่งไฟล์เสียงจาก (4.) เข้าไปที่
    - Edit Audio
    > Input: ไฟล์เสียง (ภาษาไทย)
    > Output: ไฟล์วิดิโอ (ภาษาไทย)




