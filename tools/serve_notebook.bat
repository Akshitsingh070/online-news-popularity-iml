@echo off
REM Start the Jupyter server for this project (no token, localhost only) in a detached window.
REM Usage:  tools\serve_notebook.bat   then open http://localhost:8888/tree
cd /d "%~dp0.."
if not exist logs mkdir logs
".venv\Scripts\python.exe" -m notebook ^
  --no-browser ^
  --ServerApp.token= ^
  --ServerApp.password= ^
  --ServerApp.port=8888 ^
  --ServerApp.root_dir="%CD%" >> "logs\jupyter.log" 2>&1
