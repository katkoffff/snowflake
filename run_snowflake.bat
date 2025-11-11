@echo off
title Snowflake App

REM --- Backend ---
echo Starting backend...
cd backend
call conda activate snowflake
start "" /b uvicorn main:app --reload --port 8000

REM --- Frontend ---
timeout /t 6 > nul
echo Starting frontend...
cd ../frontend
start "" /b npm run dev

REM --- Открыть браузер ---
timeout /t 3 > nul
start http://localhost:5173

echo Snowflake is running. Close this window to stop everything.
pause > nul
