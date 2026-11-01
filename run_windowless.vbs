' Double-click: starts PolyglotSTT with no window at all (not even a flash).
' Needs venv already set up (run run.bat once first). Output goes to logs\app.log.
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
root = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root
shell.Run """" & root & "\venv\Scripts\pythonw.exe"" """ & root & "\moonshine_stt.py""", 0, False
