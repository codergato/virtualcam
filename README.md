Intercepts your camera and creates a virtual camera that uses AI to remove the background. Allows for custom backgrounds in any video call app.

Instructions:

sudo apt install v4l2loopback-dkms v4l2loopback-utils  # virtual‑camera kernel module
python3 -m pip install torch torchvision opencv-python pyvirtualcam pillow numpy

./run.sh optional_background_file.png
