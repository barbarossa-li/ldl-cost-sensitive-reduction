@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
if not exist "run_all.py" goto wrongfolder
if not exist ".venv\Scripts\python.exe" goto wrongfolder
echo Installing the verified exact-score GBFRS optimizer...
.venv\Scripts\python.exe -u -c "import urllib.request,hashlib,pathlib,runpy; raw=urllib.request.urlopen('https://raw.githubusercontent.com/barbarossa-li/ldl-cost-sensitive-reduction/b1c93d22d9db4f901a86408f51892fb94499e67f/local_runner/INSTALL_FAST_25.py',timeout=180).read(); assert hashlib.sha256(raw).hexdigest()=='7be4213a9785b27b13bae95ccba1f426f873c749d5dbfcab6b6eb02c484c48bf', 'Installer hash mismatch'; p=pathlib.Path('INSTALL_FAST_25.py'); p.write_bytes(raw); runpy.run_path(str(p),run_name='__main__')"
if errorlevel 1 goto failed
echo All complete. Upload RESULTS_RETURN_25.zip to the chat.
pause
exit /b 0
:wrongfolder
echo Place this CMD file INSIDE the existing LDL_25_Expanded_Run folder, beside run_all.py.
pause
exit /b 1
:failed
echo The optimizer stopped. Keep all files and send the error message.
echo If the old experiment is still running, stop it with Ctrl+C and wait for its return archive.
echo Use this same START_FAST_WINDOWS.cmd to resume.
pause
exit /b 1
