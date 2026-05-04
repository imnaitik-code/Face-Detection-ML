# Setup

This project uses `face_recognition`, which depends on `dlib`. On Windows, install and run it with Python 3.10 or 3.11.

1. Install Python 3.11 from python.org and enable "Add python.exe to PATH".
2. Open PowerShell in this folder.
3. Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

4. Install packages:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

5. Add training images under `images`, with one folder per person:

```text
images/
  Elon Musk/
    image1.jpg
  Jeff Bezos/
    image1.jpg
```

6. Run:

```powershell
python main.py
```
or

cd D:\Face-Detection-ML
.\.venv\Scripts\python.exe main.py


Use `Esc` to close the webcam window.
