@echo off
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated!
echo Installing dependencies...
pip install -r requirements.txt
echo Setup complete! You can now run your Python scripts.
cmd /k