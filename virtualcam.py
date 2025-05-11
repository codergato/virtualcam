#!/usr/bin/env python3
from __future__ import annotations
import os
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import pyvirtualcam
from pyvirtualcam import PixelFormat

# -- Config via env vars --------------------------------------------------------
SRC_IDX      = int(os.getenv("SOURCE_INDEX", 0))
VDEV         = os.getenv("VIRTUAL_DEVICE", "/dev/video10")
DOWNSAMPLE   = float(os.getenv("DOWNSAMPLE", "0.25"))
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------------------

def make_background(bg_file: str, size: tuple[int, int]) -> torch.Tensor:
    """Return a 1×3×H×W RGB tensor in 0‑to‑1 range containing the chosen
    background image, resized to *size*.  Falls back to a muted teal when
    the file is missing.
    """
    if bg_file:
        img = Image.open(bg_file).convert("RGB").resize(size, Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    else:
        arr = np.full((*reversed(size), 3), (0.1, 0.65, 0.75), dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def cleanup(cap) -> None:
    if cap and cap.isOpened():
        cap.release()
    sys.exit(0)


@torch.no_grad()
def main(bg_file=None) -> None:
    # 1  Open physical camera ---------------------------------------------------
    cap = cv2.VideoCapture(SRC_IDX, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"❌  Cannot open camera index {SRC_IDX}")

    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"🚀  {w}×{h}@{fps:.0f}fps  device={DEVICE}")

    # 2  Load RVM model ---------------------------------------------------------
    model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3")
    rec = [None] * 4
    model.to(DEVICE).eval()

    # 3  Prepare static background ---------------------------------------------
    bg = make_background(bg_file, (w, h)).to(DEVICE)

    # 4  Handle signals ---------------------------------------------------------
    signal.signal(signal.SIGINT,  lambda *_: cleanup(cap))
    signal.signal(signal.SIGTERM, lambda *_: cleanup(cap))

    # 5  Main loop --------------------------------------------------------------
    with pyvirtualcam.Camera(w, h, fps, device=VDEV, fmt=PixelFormat.BGR) as cam:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            # BGR to RGB, to torch 1×3×H×W in 0‑1
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            src = (torch.from_numpy(rgb)
                   .to(torch.float32)
                   .to(device=DEVICE)
                   .permute(2, 0, 1).unsqueeze(0) / 255.0)

            # RVM forward pass
            fgr, pha, *rec = model(src, *rec, downsample_ratio=DOWNSAMPLE)

            # Alpha composite
            comp = pha * fgr + (1 - pha) * bg

            # Back to uint8 BGR for pyvirtualcam
            out = (comp[0].permute(1, 2, 0).clamp_(0, 1).cpu().numpy() * 255).astype(np.uint8)
            cam.send(out[:, :, ::-1])

            cam.sleep_until_next_frame()

if __name__ == "__main__":
    main(bg_file=sys.argv[1] if len(sys.argv) >= 2 else None)
