@echo off
cd /d C:\Users\NIGNOSZY\Desktop\gate

start "AI - run_v2_with_clip" cmd /k scripts\start_ai.bat

start "API - api_server" cmd /k scripts\start_api.bat