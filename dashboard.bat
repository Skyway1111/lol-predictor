@echo off
rem Opens the dashboard in your browser. The first run sets up Python and builds the database.
cd /d "%~dp0"
if not exist .venv\Scripts\lol-predict.exe (python -m venv .venv && .venv\Scripts\python -m pip install -e .) || goto :fail
if not exist data\league_predictor.db (.venv\Scripts\lol-predict update || goto :fail)
.venv\Scripts\lol-predict dashboard --open || goto :fail
exit /b
:fail
pause
exit /b 1
