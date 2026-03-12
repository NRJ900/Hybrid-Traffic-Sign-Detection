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
echo [1] Install Project (Clone Repo ^& Setup Environment)
echo [2] Setup Environment (Use Current Folder)
echo [3] Start Application (Launch Streamlit Server)
echo [4] Exit
echo.
set /p choice="Enter your choice (1, 2, 3, or 4): "

if "%choice%"=="1" goto INSTALL
if "%choice%"=="2" goto SETUP_LOCAL
if "%choice%"=="3" goto RUN
if "%choice%"=="4" goto EOF
goto MENU

:INSTALL
cls
echo ========================================================
echo [STEP 1] Downloading repository and setting up environment
echo ========================================================
echo.

:: NOTE: Replace the URL below with your actual GitHub Repository URL once you upload it!
set REPO_URL=https://github.com/NRJ900/Hybrid-Traffic-Sign-Detection.git
set PROJECT_DIR=Hybrid-Traffic-Sign-Detection

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

:SETUP_LOCAL
cls
echo ========================================================
echo [STEP 1.5] Setting up Virtual Environment Locally...
echo ========================================================
echo.

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
    
    if exist "requirements.txt" (
        echo Installing Dependencies ^(This might take a few minutes^)...
        pip install -r requirements.txt
    ) else (
        echo ERROR: requirements.txt not found in current directory.
    )
) else (
    echo Virtual Environment already exists. Checking for updates...
    call venv\Scripts\activate.bat
    if exist "requirements.txt" (
        pip install -r requirements.txt
    )
)

echo.
echo Setup Complete!
pause
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
    echo ERROR: Virtual Environment not found. Please run Option 1 Install first!
    pause
    goto MENU
)

call venv\Scripts\activate.bat

echo Which version of YOLO would you like to run?
echo.
echo [1] YOLOv5 (Old Model)
echo [2] YOLOv8 (New Model)
echo.
set /p yolo_choice="Enter your choice (1 or 2): "

echo.
echo The UI will automatically pop up in your default web browser!
echo (Keep this window open while using the app)
echo.

if "%yolo_choice%"=="2" (
    streamlit run Codes/app_hybrid_v8.py --server.port 8502
) else (
    streamlit run Codes/app_hybrid.py --server.port 8502
)

pause
goto MENU

:EOF
exit
