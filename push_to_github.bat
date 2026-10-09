@echo off
setlocal
title Push Face Recognition Attendance System to GitHub

set "PATH=%PATH%;C:\Users\ankit\.gemini\antigravity\scratch\mingit\cmd;C:\Users\ankit\.gemini\antigravity\scratch\gh\bin"
cd /d "C:\Users\ankit\.gemini\antigravity\scratch\face-attendance-system"

echo ======================================================================
echo    AI Face Recognition Attendance System - GitHub Uploader
echo    Target: https://github.com/Lucifer8757/face-recoginition-attendace-system
echo ======================================================================
echo.

echo Choose authentication method:
echo   [1] Use GitHub Personal Access Token (Fastest, 100%% reliable)
echo   [2] Login via GitHub Web / Device Code (gh auth login)
echo.
set /p CHOICE="Select option (1 or 2): "

if "%CHOICE%"=="1" (
    echo.
    echo If you do not have a token yet:
    echo 1. Open https://github.com/settings/tokens in your browser
    echo 2. Click "Generate new token (classic)"
    echo 3. Note: "Attendance Project", check "repo" scope, click Generate.
    echo.
    set /p GHTOKEN="Paste your GitHub Personal Access Token here: "
    if not "%GHTOKEN%"=="" (
        git remote set-url origin https://%GHTOKEN%@github.com/Lucifer8757/face-recoginition-attendace-system.git
        echo.
        echo [*] Pushing code to main branch...
        git push -u origin main
        echo.
        if %ERRORLEVEL% EQU 0 (
            echo [+] SUCCESS! Your code has been uploaded to GitHub!
            echo [+] Repository: https://github.com/Lucifer8757/face-recoginition-attendace-system
        ) else (
            echo [-] Push failed. Please check your token permissions (must have 'repo' checked).
        )
    )
) else (
    echo.
    echo [*] Starting GitHub web device login...
    gh auth login --hostname github.com --git-protocol https --web
    echo.
    echo [*] Pushing to main branch...
    git push -u origin main
    echo.
    if %ERRORLEVEL% EQU 0 (
        echo [+] SUCCESS! Your code has been uploaded to GitHub!
        echo [+] Repository: https://github.com/Lucifer8757/face-recoginition-attendace-system
    )
)

echo.
echo ======================================================================
echo NEXT STEP: Deploy on Vercel
echo 1. Open https://vercel.com/new
echo 2. Import "Lucifer8757/face-recoginition-attendace-system"
echo 3. Click "Deploy" (Vercel builds and hosts automatically!)
echo ======================================================================
pause
