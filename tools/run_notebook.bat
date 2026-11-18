@echo off
cd /d "%~dp0.."
".venv\Scripts\python.exe" -u tools\build_notebook.py > "logs\notebook_build.log" 2>&1
