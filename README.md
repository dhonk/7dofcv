FOR USE ON WINDOWS MACHINE WITH WSL

setup
python3 -m venv ./venv
pip install -r requirements.txt

model installation
wget "https://storage.googleapis.com/mediapipe-models/pose\_landmarker/pose\_landmarker\_full/float16/latest/pose\_landmarker\_full.task"
wget "https://storage.googleapis.com/mediapipe-models/hand\_landmarker/hand\_landmarker/float16/latest/hand\_landmarker.task"

open powershell in administrator
> usbipd list
> usbipd bind --busid <camera id>
> usbipd attach --wsl --busid <camera id>
