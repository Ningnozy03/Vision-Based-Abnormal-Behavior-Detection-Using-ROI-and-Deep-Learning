@echo off
cd /d C:\Users\NIGNOSZY\Desktop\gate
call .\.venv\Scripts\activate

python src\api_server.py

pause