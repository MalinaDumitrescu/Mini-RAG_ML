@echo off
echo ===================================================
echo      Starting ChudGTP (Backend + Frontend)
echo ===================================================

:: 1. Start Backend in a new window
echo Starting Backend Server...
start "ChudGTP Backend" cmd /k "call .venv\Scripts\activate && python scripts\run_backend.py"

:: Wait a few seconds for backend to initialize
timeout /t 5

:: 2. Start Frontend in a new window
echo Starting Frontend...
cd frontend
start "ChudGTP Frontend" cmd /k "npm run dev"

echo.
echo ===================================================
echo    App is running!
echo    Backend: http://localhost:8000
echo    Frontend: http://localhost:5173 (Check the Frontend window for exact URL)
echo ===================================================
pause