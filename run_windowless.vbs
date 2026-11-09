' Double-click: starts PolyglotSTT with no window at all (not even a flash).
' Needs venv already set up (run run.bat once first). Output goes to
' %LOCALAPPDATA%\PolyglotSTT\logs\app.log - pythonw has no console.
'
' WindowStyle 0 = hidden, so cmd.exe (needed for the env var) stays invisible.
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
root = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root
pyw = root & "\venv\Scripts\pythonw.exe"
If Not fso.FileExists(pyw) Then
    MsgBox "No venv found. Run run.bat once first to install everything.", _
           vbExclamation, "PolyglotSTT"
    WScript.Quit 1
End If
cmd = "set ""POLYGLOTSTT_WINDOWLESS=1"" && " & _
      "set ""PATH=" & root & "\venv\Scripts;" & root & "\python;%PATH%"" && " & _
      "set ""PYTHONPATH="" && set ""PYTHONHOME="" && " & _
      """" & pyw & """ """ & root & "\moonshine_stt.py"""
shell.Run "cmd /c " & cmd, 0, False
