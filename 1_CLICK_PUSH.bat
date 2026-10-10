@echo off
setlocal
title AI Attendance System - 1-Click Push to GitHub

echo ======================================================================
echo    AI Face Recognition Attendance System
echo    Pushing project to: https://github.com/Lucifer8757/face-recognition-attendance-system
echo ======================================================================
echo.

cd /d "C:\Users\ankit\.gemini\antigravity\scratch\face-attendance-system"

echo [*] Starting git push...
echo [*] If a GitHub browser popup appears, simply click "Sign in with your browser".
echo.

git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ======================================================================
    echo  [+] SUCCESS! All project files are now LIVE on GitHub!
    echo  [+] View repository: https://github.com/Lucifer8757/face-recognition-attendance-system
    echo ======================================================================
    echo.
    echo [*] Opening Vercel deployment page in your browser...
    start https://vercel.com/new
) else (
    echo.
    echo [-] If push failed due to authentication:
    echo     Please paste your GitHub Personal Access Token or ensure you are logged into GitHub in your browser.
)

echo.
pause
