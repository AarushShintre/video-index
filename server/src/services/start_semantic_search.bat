@echo off
REM Start the semantic search Python service on Windows

cd /d "%~dp0"
python semanticSearch.py --port 5001 --results-dir ..\..\output

