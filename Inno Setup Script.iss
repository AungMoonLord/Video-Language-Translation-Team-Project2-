; Script generated for Video AI Translator
; User: aungl

#define MyAppName "Video AI Translator"
#define MyAppVersion "1.0"
#define MyAppPublisher "Capybara"
#define MyAppExeName "Video AI Translator.exe"

[Setup]
; AppId สร้างใหม่ให้ไม่ซ้ำใคร
AppId={{AUNGL-VIDEO-AI-TRANSLATOR-2026}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
; ลงที่ Program Files อัตโนมัติ (ตอบโจทย์เรื่อง Destination folder)
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
; ชื่อไฟล์ติดตั้งที่จะได้ออกมา
OutputBaseFilename=Video AI Translator Setup
; ไอคอนของตัวติดตั้ง
SetupIconFile=C:\Users\aungl\make app\icon_app.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
; บังคับโหมด 64-bit
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; ดึงไฟล์ทั้งหมดจากโฟลเดอร์ที่ PyInstaller สร้างไว้
Source: "C:\Users\aungl\make app\dist\Video AI Translator\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent