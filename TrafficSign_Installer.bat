@echo off
color 0a

:MENU
cls
echo ========================================================
echo       Hybrid Traffic Sign Detection Framework
echo ========================================================
echo.
echo Please select an option:
echo.
echo [1] Install Project (Download Repo ^& Install Requirements)
echo [2] Start Application (Launch Streamlit Server)
echo [3] Exit
echo.
set /p choice="Enter your choice (1, 2, or 3): "

if "%choice%"=="1" goto INSTALL
if "%choice%"=="2" goto RUN
if "%choice%"=="3" goto EOF
goto MENU

:INSTALL
cls
echo ========================================================
echo [STEP 1] Downloading repository and setting up environment
echo ========================================================
echo.

:: NOTE: Replace the URL below with your actual GitHub Repository URL once you upload it!
set REPO_URL=https://github.com/NRJ900/Traffic-Sign-Detection.git
set PROJECT_DIR=Traffic-Sign-Detection-App

if not exist "%PROJECT_DIR%" (
    echo Downloading project from GitHub...
    git clone %REPO_URL% %PROJECT_DIR%
    if errorlevel 1 (
        echo ERROR: Git clone failed. Make sure Git is installed on your computer.
        pause
        goto MENU
    )
) else (
    echo Project folder '%PROJECT_DIR%' already exists. Skipping download.
)

cd %PROJECT_DIR%

if not exist "venv\Scripts\activate.bat" (
    echo Creating Python Virtual Environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment. Ensure Python is installed and in your PATH.
        pause
        goto MENU
    )
    
    echo Activating environment...
    call venv\Scripts\activate.bat
    
    echo Installing Dependencies (This might take a few minutes)...
    pip install -r requirements.txt
) else (
    echo Virtual Environment already exists. Checking for updates...
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
)

echo.
echo Installation Complete!
pause
cd ..
goto MENU

:RUN
cls
echo ========================================================
echo [STEP 2] Starting Hybrid AI Framework Server...
echo ========================================================
echo.

:: Check if the user is running this from outside the project directory
if exist "Traffic-Sign-Detection-App" (
    cd Traffic-Sign-Detection-App
)

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual Environment not found. Please run Option 1 (Install) first!
    pause
    goto MENU
)

call venv\Scripts\activate.bat

echo The UI will automatically pop up in your default web browser!
echo (Keep this black window open while using the app)
echo.

streamlit run Codes/app_hybrid.py --server.port 8502

pause
goto MENU

:EOF
exit
