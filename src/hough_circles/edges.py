from typing import Tuple
import numpy as np
from PIL import Image

def load_grayscale(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32)

def normalize_uint8(img: np.ndarray) -> np.ndarray:
    m, M = float(np.min(img)), float(np.max(img))
    if M - m < 1e-9:
        return np.zeros_like(img, dtype=np.uint8)
    out = (img - m) / (M - m) * 255.0
    return out.astype(np.uint8)

def simple_sobel_edges(gray: np.ndarray, thresh: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    Ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)

    def conv2(img, K):
        H, W = img.shape
        kh, kw = K.shape
        pad_h, pad_w = kh//2, kw//2
        padded = np.pad(img, ((pad_h,pad_h),(pad_w,pad_w)), mode="edge")
        out = np.zeros_like(img, dtype=np.float32)
        for i in range(H):
            for j in range(W):
                patch = padded[i:i+kh, j:j+kw]
                out[i,j] = float(np.sum(patch * K))
        return out

    gx = conv2(gray, Kx)
    gy = conv2(gray, Ky)
    mag = np.sqrt(gx*gx + gy*gy)
    mag_bin = (mag >= thresh).astype(np.uint8) * 255
    return mag_bin, mag
