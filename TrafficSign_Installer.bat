@echo off
color 0a
echo ========================================================
echo       Hybrid Traffic Sign Detection Framework
echo ========================================================
echo.

:: NOTE: Replace the URL below with your actual GitHub Repository URL once you upload it!
set REPO_URL=https://github.com/NRJ900/Traffic-Sign-Detection.git
set PROJECT_DIR=Traffic-Sign-Detection-App

:: Step 1: Check if the repository is already downloaded
if not exist "%PROJECT_DIR%" (
    echo [1/3] Downloading project from GitHub...
    git clone %REPO_URL% %PROJECT_DIR%
    if errorlevel 1 (
        echo ERROR: Git clone failed. Make sure Git is installed on your computer.
        pause
        exit /b
    )
) else (
    echo [1/3] Project folder '%PROJECT_DIR%' already exists. Skipping download.
)

cd %PROJECT_DIR%

:: Step 2: Check for Virtual Environment and install requirements
if not exist "venv\Scripts\activate.bat" (
    echo [2/3] Creating Python Virtual Environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment. Ensure Python is installed and in your PATH.
        pause
        exit /b
    )
    
    echo [2/3] Activating environment...
    call venv\Scripts\activate.bat
    
    echo [2/3] Installing Dependencies (This might take a few minutes)...
    pip install -r requirements.txt
) else (
    echo [2/3] Virtual Environment already exists. Checking for updates...
    call venv\Scripts\activate.bat
)

:: Step 3: Launch the Streamlit App
echo [3/3] Starting Hybrid AI Framework Server...
echo The UI will automatically pop up in your default web browser!
echo (Keep this black window open while using the app)
echo.

streamlit run Codes/app_hybrid.py --server.port 8502

pause
