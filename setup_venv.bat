@echo off
echo Installing missing dependencies...
venv\Scripts\pip install pyvis flask
echo Dependencies installed successfully!
echo.
echo To activate the virtual environment, run:
echo venv\Scripts\activate
echo.
echo To run the Flask app:
echo venv\Scripts\python app.py
pause