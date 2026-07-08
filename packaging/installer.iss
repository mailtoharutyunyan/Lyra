; Inno Setup script for the Windows installer.
; Build:  iscc /DVersion=%VERSION% packaging\installer.iss
; Expects the PyInstaller onedir output at dist\Lyra\ (from packaging\app.spec).

#ifndef AppName
  #define AppName "Lyra"
#endif
#ifndef Version
  #define Version "0.1.0"
#endif

[Setup]
AppName={#AppName}
AppVersion={#Version}
AppPublisher=Lyra
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
UninstallDisplayIcon={app}\{#AppName}.exe
OutputDir=dist\installer
OutputBaseFilename={#AppName}-{#Version}-setup
Compression=lzma2
SolidCompression=yes
; Per-user install: no admin/UAC prompt.
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "dist\{#AppName}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppName}.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppName}.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Run]
Filename: "{app}\{#AppName}.exe"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
