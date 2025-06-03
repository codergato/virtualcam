Intercepts your camera and creates a virtual camera that uses AI to remove the background. Allows for custom backgrounds in any video call app.

Instructions:

# Install dependencies (no pyvirtualcam)
pip install torch torchvision opencv-python pillow numpy

# For Apple Silicon Macs, use MPS acceleration
pip install torch torchvision torchaudio

brew install --cask obs

python3 virtualcam.py

q to quit
s to save current frame
1-4 to switch background modes

Start OBS Virtual Camera
Add OBS source - screen capture - choose program window 

./run.sh 
