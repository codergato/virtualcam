#!/usr/bin/env python3
from __future__ import annotations
import os
import signal
import sys
import time
import math
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
BG_MODE      = os.getenv("BG_MODE", "gradient")  # gradient, slideshow, video, particles
BG_SPEED     = float(os.getenv("BG_SPEED", "1.0"))  # Animation speed multiplier

# ------------------------------------------------------------------------------

class AnimatedBackground:
    def __init__(self, size: tuple[int, int], mode: str = "gradient", speed: float = 1.0):
        self.size = size
        self.mode = mode
        self.speed = speed
        self.frame_count = 0
        self.bg_images = []
        self.current_bg_idx = 0
        
        # Load background images if in slideshow mode
        if mode == "slideshow":
            self._load_background_images()
    
    def _load_background_images(self):
        """Load all background images from a directory"""
        bg_dir = Path("backgrounds")
        if bg_dir.exists():
            for ext in ["*.jpg", "*.png", "*.jpeg"]:
                self.bg_images.extend(list(bg_dir.glob(ext)))
        
        if not self.bg_images:
            print("⚠️  No background images found in 'backgrounds/' directory")
    
    def get_frame(self) -> torch.Tensor:
        """Generate animated background frame"""
        self.frame_count += 1
        
        if self.mode == "gradient":
            return self._generate_gradient_bg()
        elif self.mode == "slideshow":
            return self._generate_slideshow_bg()
        elif self.mode == "particles":
            return self._generate_particles_bg()
        elif self.mode == "waves":
            return self._generate_waves_bg()
        else:
            return self._generate_gradient_bg()
    
    def _generate_gradient_bg(self) -> torch.Tensor:
        """Generate animated gradient background"""
        w, h = self.size
        t = self.frame_count * self.speed * 0.02
        
        # Create coordinate grids
        y, x = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')
        
        # Animated gradient parameters
        r = 0.5 + 0.3 * np.sin(t) + 0.2 * np.sin(x * 4 + t * 2)
        g = 0.3 + 0.4 * np.cos(t * 1.3) + 0.3 * np.cos(y * 3 + t * 1.5)
        b = 0.6 + 0.3 * np.sin(t * 0.8) + 0.2 * np.sin((x + y) * 2 + t)
        
        # Combine channels and clamp
        arr = np.stack([r, g, b], axis=-1).astype(np.float32)
        arr = np.clip(arr, 0, 1)
        
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    
    def _generate_slideshow_bg(self) -> torch.Tensor:
        """Cycle through background images"""
        if not self.bg_images:
            return self._generate_gradient_bg()
        
        # Switch image every 3 seconds (assuming ~30fps)
        switch_interval = int(90 / self.speed)
        if self.frame_count % switch_interval == 0:
            self.current_bg_idx = (self.current_bg_idx + 1) % len(self.bg_images)
        
        # Load and resize current background
        img_path = self.bg_images[self.current_bg_idx]
        img = Image.open(img_path).convert("RGB").resize(self.size, Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        
        # Apply fade effect during transitions
        fade_duration = 30  # frames
        fade_progress = (self.frame_count % switch_interval) / fade_duration
        if fade_progress < 1.0:
            # Blend with previous image or gradient
            blend_factor = 0.5 + 0.5 * np.sin(fade_progress * np.pi - np.pi/2)
            gradient_bg = self._generate_gradient_bg()[0].permute(1, 2, 0).numpy()
            arr = arr * blend_factor + gradient_bg * (1 - blend_factor)
        
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    
    def _generate_particles_bg(self) -> torch.Tensor:
        """Generate animated particle background"""
        w, h = self.size
        t = self.frame_count * self.speed * 0.05
        
        # Create base background
        bg = np.full((h, w, 3), (0.05, 0.1, 0.2), dtype=np.float32)
        
        # Generate floating particles
        num_particles = 50
        for i in range(num_particles):
            # Particle position with movement
            x = (0.3 * np.sin(t + i * 0.1) + 0.5 + i * 0.02) % 1.0
            y = (0.2 * np.cos(t * 1.3 + i * 0.15) + 0.5 + i * 0.03) % 1.0
            
            # Convert to pixel coordinates
            px = int(x * w)
            py = int(y * h)
            
            # Particle properties
            size = int(5 + 3 * np.sin(t * 2 + i))
            brightness = 0.3 + 0.2 * np.cos(t + i * 0.2)
            
            # Draw particle (simple circle)
            for dy in range(-size, size + 1):
                for dx in range(-size, size + 1):
                    if 0 <= py + dy < h and 0 <= px + dx < w:
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist <= size:
                            alpha = brightness * (1 - dist / size)
                            color = [0.6, 0.8, 1.0]  # Light blue particles
                            for c in range(3):
                                bg[py + dy, px + dx, c] = min(1.0, bg[py + dy, px + dx, c] + alpha * color[c])
        
        return torch.from_numpy(bg).permute(2, 0, 1).unsqueeze(0)
    
    def _generate_waves_bg(self) -> torch.Tensor:
        """Generate animated wave background"""
        w, h = self.size
        t = self.frame_count * self.speed * 0.03
        
        # Create coordinate grids
        y, x = np.meshgrid(np.linspace(0, 4*np.pi, h), np.linspace(0, 4*np.pi, w), indexing='ij')
        
        # Wave patterns
        wave1 = np.sin(x + t) * np.cos(y + t * 0.7)
        wave2 = np.cos(x * 0.5 + t * 1.3) * np.sin(y * 0.8 + t * 0.9)
        wave3 = np.sin((x + y) * 0.3 + t * 0.5)
        
        # Combine waves
        combined = (wave1 + wave2 + wave3) / 3
        
        # Map to colors
        r = 0.3 + 0.3 * combined
        g = 0.5 + 0.2 * np.sin(combined * np.pi + t)
        b = 0.7 + 0.3 * np.cos(combined * np.pi * 2 + t * 1.5)
        
        arr = np.stack([r, g, b], axis=-1).astype(np.float32)
        arr = np.clip(arr, 0, 1)
        
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

def cleanup(cap) -> None:
    if cap and cap.isOpened():
        cap.release()
    sys.exit(0)

@torch.no_grad()
def main(bg_mode=None) -> None:
    # 1  Open physical camera ---------------------------------------------------
    cap = cv2.VideoCapture(SRC_IDX, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"❌  Cannot open camera index {SRC_IDX}")
    
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"🚀  {w}×{h}@{fps:.0f}fps  device={DEVICE}")
    
    # 2  Load RVM model ---------------------------------------------------------
    model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3")
    rec = [None] * 4
    model.to(DEVICE).eval()
    
    # 3  Setup animated background ----------------------------------------------
    bg_mode = bg_mode or BG_MODE
    animated_bg = AnimatedBackground((w, h), bg_mode, BG_SPEED)
    print(f"🎨  Background mode: {bg_mode} (speed: {BG_SPEED}x)")
    
    # 4  Handle signals ---------------------------------------------------------
    signal.signal(signal.SIGINT,  lambda *_: cleanup(cap))
    signal.signal(signal.SIGTERM, lambda *_: cleanup(cap))
    
    # 5  Main loop --------------------------------------------------------------
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
            
            # Get animated background frame
            bg = animated_bg.get_frame().to(DEVICE)
            
            # Alpha composite
            comp = pha * fgr + (1 - pha) * bg
            
            # Back to uint8 BGR for pyvirtualcam
            out = (comp[0].permute(1, 2, 0).clamp_(0, 1).cpu().numpy() * 255).astype(np.uint8)
            cam.send(out[:, :, ::-1])
            cam.sleep_until_next_frame()

if __name__ == "__main__":
    bg_mode = sys.argv[1] if len(sys.argv) >= 2 else None
    main(bg_mode)
