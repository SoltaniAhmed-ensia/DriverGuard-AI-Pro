@echo off
REM Opens the live dashboard in a second window, then starts the driver monitor.
start "Dashboard" cmd /k streamlit run dashboard.py
python main.py --lang ar --yolo --voice
pause
