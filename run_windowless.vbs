' Double-click: starts PolyglotSTT with no window at all (not even a flash).
' Needs venv already set up (run run.bat once first). Output goes to logs\app.log.
CreateObject("Wscript.Shell").Run "venv\Scripts\pythonw.exe moonshine_stt.py", 0, False
