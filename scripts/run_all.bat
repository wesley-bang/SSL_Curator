@echo off
setlocal

set "PY_CMD="

if defined PYTHON (
    set "PY_CMD=%PYTHON%"
) else if exist ".\.venv\Scripts\python.exe" (
    set "PY_CMD=.\.venv\Scripts\python.exe"
) else if exist ".\.venv\Scripts\python.bat" (
    set "PY_CMD=.\.venv\Scripts\python.bat"
) else (
    set "PY_CMD=python"
)

set "PY_RUN=%PY_CMD%"
set "NEEDS_QUOTE=0"
if not "%PY_CMD:\=%"=="%PY_CMD%" set "NEEDS_QUOTE=1"
if not "%PY_CMD::=%"=="%PY_CMD%" set "NEEDS_QUOTE=1"
if %NEEDS_QUOTE%==1 (
    if not exist "%PY_CMD%" (
        echo Specified Python interpreter "%PY_CMD%" not found.>&2
        exit /b 1
    )
    set "PY_RUN=\"%PY_CMD%\""
)

echo Using interpreter: %PY_CMD%
echo.

echo [1/3] Scoring manifest...
call %PY_RUN% -m scoring.score ^
  --manifest data\toy_manifest.csv ^
  --out scores.parquet ^
  --ssl hubert_base ^
  --layers-content 9 10 11 12 ^
  --layers-speaker 3 4 5 6 ^
  --mos dnsmos ^
  --seed 1337
if errorlevel 1 goto fail

echo [2/3] Selecting curated subset...
call %PY_RUN% -m selector.select ^
  --scores scores.parquet ^
  --quotas configs\quotas.yaml ^
  --alpha 0.10 ^
  --K-hours 10 ^
  --diversity-min-cos 0.02 ^
  --uncert-beta 0.0 ^
  --out curated\curated_train_K=10.csv ^
  --alpha-sweep logs\alpha_sweep.csv ^
  --slice-stats logs\slice_stats.json
if errorlevel 1 goto fail

echo [3/3] Running evaluation...
call %PY_RUN% -m eval.tail_metrics ^
  --scores scores.parquet ^
  --selected curated\curated_train_K=10.csv ^
  --out eval\eval_report.md
if errorlevel 1 goto fail

echo.
echo Pipeline completed successfully.
exit /b 0

:fail
echo.
echo Pipeline failed with exit code %errorlevel%.
exit /b %errorlevel%