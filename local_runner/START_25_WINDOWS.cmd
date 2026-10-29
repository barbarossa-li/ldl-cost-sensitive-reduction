@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
echo Downloading the verified LDL-25 online installer...
py -3.12 -c "import urllib.request,hashlib,pathlib,runpy; raw=urllib.request.urlopen('https://raw.githubusercontent.com/barbarossa-li/ldl-cost-sensitive-reduction/ec3f050523ea2142280726343666acaf3c63efb7/local_runner/BOOTSTRAP_25.py',timeout=180).read(); assert hashlib.sha256(raw).hexdigest()=='fa86b7141f4198730118d5fc52376e1a2cf04b18c876f332fd8e3eb5c75c670f', 'Installer hash mismatch'; p=pathlib.Path('BOOTSTRAP_25.py'); p.write_bytes(raw); runpy.run_path(str(p),run_name='__main__')"
if errorlevel 1 goto failed
echo Complete. Upload LDL_25_Expanded_Run\RESULTS_RETURN_25.zip to the chat.
pause
exit /b 0
:failed
echo Installation or execution stopped. Keep this folder and send a screenshot.
echo Double-click this file again to retry; completed checkpoints are retained.
pause
exit /b 1
