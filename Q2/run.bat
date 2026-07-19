@echo off
setlocal

REM One-command runner for Question 2 preprocessing.
REM Put all input CSV files inside a folder named "datasets" next to this file.
REM Then open CMD in this folder and type: run

where py >nul 2>nul
if %errorlevel%==0 (
    py preprocess_question2.py
) else (
    python preprocess_question2.py
)

if errorlevel 1 (
    echo.
    echo The preprocessing script failed. Check the error message above.
    pause
    exit /b 1
)

echo.
echo Created training_dataset.csv successfully.
pause
