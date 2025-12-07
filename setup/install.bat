@echo off
title Installer Demo_EDBT
setlocal enabledelayedexpansion

echo ============================================
echo        INSTALLER DEMO_EDBT (Unificato)
echo ============================================
echo.
echo Scegli il tipo di ambiente da creare:
echo   1) Conda environment
echo   2) Python venv
echo.
set /p CHOICE="Inserisci 1 o 2: "

REM ---------------------------------------------------------
REM  PATH BASE & FILE LOCALI
REM ---------------------------------------------------------

set "SCRIPT_DIR=%~dp0"
set "PARENT_DIR=%SCRIPT_DIR%.."
set "VENV_DIR=%PARENT_DIR%\Demo_EDBT"
set "CONDA_ENV_DIR=%PARENT_DIR%\Demo_EDBT"

set "OLLAMA_EXE=%SCRIPT_DIR%OllamaSetup.exe"
set "LMSTUDIO_EXE=%SCRIPT_DIR%LM-Studio-0.3.31-7-x64.exe"

echo.
echo Cartella script : %SCRIPT_DIR%
echo Cartella target : %PARENT_DIR%
echo.

if not exist "%OLLAMA_EXE%" (
    echo ERRORE: non trovo OllamaSetup.exe nella cartella dello script!
    pause
    exit /b 1
)

if not exist "%LMSTUDIO_EXE%" (
    echo ERRORE: non trovo LM-Studio*.exe nella cartella dello script!
    pause
    exit /b 1
)

REM ---------------------------------------------------------
REM  SCELTA UTENTE
REM ---------------------------------------------------------

if "%CHOICE%"=="1" goto USE_CONDA
if "%CHOICE%"=="2" goto USE_VENV

echo Scelta non valida.
pause
exit /b 1


REM =========================================================
REM ===============   INSTALLAZIONE CONDA   =================
REM =========================================================
:USE_CONDA
echo ============================================
echo      CREO AMBIENTE CONDA Demo_EDBT
echo ============================================

conda env create -f "%SCRIPT_DIR%conda_env.yml" -p "%CONDA_ENV_DIR%"
if %errorlevel% neq 0 (
    echo ERRORE creazione ambiente Conda
    pause
    exit /b 1
)

call conda activate "%CONDA_ENV_DIR%"
goto AFTER_ENV_SETUP


REM =========================================================
REM ===============   INSTALLAZIONE VENV   ==================
REM =========================================================
:USE_VENV
echo ============================================
echo    CREO PYTHON VENV Demo_EDBT
echo ============================================

python -m venv "%VENV_DIR%"
call "%VENV_DIR%\Scripts\activate.bat"

python -m pip install --upgrade pip
python -m pip install huggingface_hub spacy
python -m pip install -r "%SCRIPT_DIR%requirements.txt"

goto AFTER_ENV_SETUP


REM =========================================================
REM =========   PARTI COMUNI DOPO CONDA O VENV   ===========
REM =========================================================
:AFTER_ENV_SETUP
echo.

echo ============================================
echo        LOGIN AUTOMATICO A HUGGINGFACE
echo ============================================
:: ⚠ Metti qui il tuo token HF (meglio NON committare questo file su Git)
set HF_TOKEN=
huggingface-cli login --token %HF_TOKEN%
echo Login completato.
echo.

echo ============================================
echo   DOWNLOAD META-LLAMA-3-8B
echo ============================================
echo.

:: ID del modello su Hugging Face
set MODEL_ID=meta-llama/Meta-Llama-3-8B

:: Cartella locale dove salvare il modello (accanto al .bat)
set MODEL_DIR=%~dp0Meta-Llama-3-8B

echo Verifica presenza di huggingface-cli...
where huggingface-cli >nul 2>&1
if %errorlevel% neq 0 (
    echo ERRORE: huggingface-cli non trovato. Installa con:
    echo     pip install -U "huggingface_hub[cli]"
    echo.
    pause
    exit /b 1
)

echo.
echo Eseguo login su Hugging Face...
huggingface-cli login --token %HF_TOKEN% --add-to-git-credential >nul 2>&1
if %errorlevel% neq 0 (
    echo ERRORE: login fallito. Controlla il token HF.
    echo.
    pause
    exit /b 1
)

echo Login completato con successo.
echo.

:: Controllo se il modello sembra gia' scaricato (presenza di tokenizer.model)
if exist "%MODEL_DIR%\tokenizer.model" (
    echo Il modello sembra gia' scaricato in:
    echo   %MODEL_DIR%
    echo (trovato tokenizer.model)
    echo.
) else (
    echo Il modello non e' ancora presente in:
    echo   %MODEL_DIR%
    echo Avvio il download di %MODEL_ID% ...
    echo.

    huggingface-cli download %MODEL_ID% ^
        --local-dir "%MODEL_DIR%" ^
        --local-dir-use-symlinks False

    if %errorlevel% neq 0 (
        echo.
        echo ERRORE: download del modello fallito.
        echo - Hai i permessi sul repo %MODEL_ID%?
        echo - Il token HF e' corretto?
        echo.
        pause
        exit /b 1
    )

    echo.
    echo Download completato con successo.
)

echo.


echo ============================================
echo          INSTALLAZIONE OLLAMA
echo ============================================
start /wait "%OLLAMA_EXE%"
echo Ollama installato.

echo Avvio Ollama...
start "" "C:\Program Files\Ollama\ollama.exe"
echo.


echo ============================================
echo          INSTALLAZIONE LM STUDIO
echo ============================================
start /wait "%LMSTUDIO_EXE%"
echo LM Studio installato.
echo.


echo ============================================
echo       DOWNLOAD MODELLO LLaMA 3.1 8B
echo ============================================
mkdir "%PARENT_DIR%\models"
pushd "%PARENT_DIR%\models"

huggingface-cli download meta-llama/Llama-3.1-8B ^
    --local-dir Llama-3.1-8B ^
    --include "*.safetensors" "*.json" "*.model"

popd
echo Download completato.
echo.


echo ============================================
echo    INSTALLAZIONE MODELLO SU OLLAMA
echo ============================================
start cmd /k "ollama serve"
ollama pull llama3.1:8b
echo Fatto.
echo.


echo ============================================
echo   INSTALLAZIONE MODELLO SU LM STUDIO
echo ============================================
start cmd /k "lms server start"
lms get https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF
echo Fatto.
echo.


echo ============================================
echo         INSTALLAZIONE SPACY
echo ============================================
python -m spacy download en_core_web_sm
echo spaCy installato.
echo.


echo ============================================
echo         INSTALLAZIONE COMPLETA
echo ============================================
pause
exit /b 0
