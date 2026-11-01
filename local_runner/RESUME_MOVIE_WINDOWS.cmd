@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
if not exist "run_all.py" goto wrongfolder
if not exist ".venv\Scripts\python.exe" goto wrongfolder
echo Resuming existing LDL-25 results with explicit GBFRS numerical amendment...
.venv\Scripts\python.exe -u -c "import urllib.request,hashlib,pathlib,runpy; raw=urllib.request.urlopen('https://raw.githubusercontent.com/barbarossa-li/ldl-cost-sensitive-reduction/2a4ac6beae366b0f402abfc8f9a2d78a3101ed58/local_runner/RESUME_25_MOVIE.py',timeout=180).read(); assert hashlib.sha256(raw).hexdigest()=='5a69d164367c350daf55e3330e2bd3591162916943345a736f2747210bd0d716', 'Patch hash mismatch'; p=pathlib.Path('RESUME_25_MOVIE.py'); p.write_bytes(raw); runpy.run_path(str(p),run_name='__main__')"
if errorlevel 1 goto failed
echo Complete. Upload RESULTS_RETURN_25.zip to the chat.
pause
exit /b 0
:wrongfolder
echo Place this CMD file INSIDE the existing LDL_25_Expanded_Run folder.
echo It must be beside run_all.py. Do not create a new experiment folder.
pause
exit /b 1
:failed
echo The run stopped. Keep all files and send the error and RESULTS_RETURN_25.zip.
echo Use this same resume CMD to continue. Do not edit the frozen source files.
pause
exit /b 1
