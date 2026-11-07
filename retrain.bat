@echo off
rem Pulls new games and refits all four models on them, about 40 minutes.
cd /d "%~dp0"
if not exist .venv\Scripts\lol-predict.exe (python -m venv .venv && .venv\Scripts\python -m pip install -e .) || goto :fail
.venv\Scripts\lol-predict retrain
:fail
pause
