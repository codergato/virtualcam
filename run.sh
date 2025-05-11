sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback devices=1 video_nr=10              card_label="NoBlueCam" exclusive_caps=1 fmt=bgr24
python3 virtualcam.py $1
