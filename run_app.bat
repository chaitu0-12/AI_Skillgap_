@echo off
echo AI-Powered Skill Gap Analyzer
echo ==============================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python and make sure it's in your PATH.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Error creating virtual environment.
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo Error activating virtual environment.
    pause
    exit /b 1
)

REM Check if requirements are installed
echo Checking if requirements are installed...
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing required packages...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo Error installing required packages.
        pause
        exit /b 1
    )
)

REM Check if spaCy model is installed
echo Checking if spaCy model is installed...
python -c "import spacy; spacy.load('en_core_web_sm')" >nul 2>&1
if %errorlevel% neq 0 (
    echo Downloading spaCy English model...
    python -m spacy download en_core_web_sm
    if %errorlevel% neq 0 (
        echo Error downloading spaCy model.
        pause
        exit /b 1
    )
)

REM Run the application
echo Starting the application...
streamlit run app/main.py

pause