@echo off
rem Pulls new games from Oracle's Elixir and scores them with the current models, about 8 minutes.
cd /d "%~dp0"
if not exist .venv\Scripts\lol-predict.exe (python -m venv .venv && .venv\Scripts\python -m pip install -e .) || goto :fail
.venv\Scripts\lol-predict update
:fail
pause
