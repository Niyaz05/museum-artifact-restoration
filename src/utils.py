# src/utils.py
import os
import numpy as np
import cv2
from typing import Tuple

# ----------------- filesystem helper -----------------
def ensure_dir(path: str):
    """Create directory if missing (no error if exists)."""
    if path is None or path == '':
        return
    os.makedirs(path, exist_ok=True)

# ----------------- tensor / image helpers -----------------
def tensor_to_uint8(x) -> np.ndarray:
    """
    Convert a torch tensor or numpy array to uint8 HxWx3 (RGB).
    Accepts:
      - torch.Tensor [C,H,W] or [B,C,H,W] floats in 0..1 or 0..255
      - numpy array HxWxC or CxHxW
    Returns: H x W x 3 uint8 (RGB)
    """
    # lazy import torch to avoid hard dependency if not used
    try:
        import torch
        is_tensor = isinstance(x, torch.Tensor)
    except Exception:
        is_tensor = False

    if is_tensor:
        a = x.detach().cpu()
        # handle batched
        if a.ndim == 4:
            a = a[0]
        # expect [C,H,W]
        if a.ndim != 3:
            raise ValueError(f"tensor_to_uint8 expected 3-d tensor but got shape {a.shape}")
        a = a.numpy()
        a = np.transpose(a, (1,2,0))
    else:
        a = np.array(x)

    # Now a is HxWxC or HxW
    if a.ndim == 2:
        a = np.stack([a,a,a], axis=-1)
    if a.ndim == 3 and a.shape[2] == 1:
        a = np.concatenate([a,a,a], axis=2)
    if a.ndim == 3 and a.shape[2] == 4:
        # assume RGBA, drop alpha
        a = a[...,:3]

    # If floats in 0..1 scale to 0..255
    if np.issubdtype(a.dtype, np.floating):
        if a.max() <= 1.01:
            a = (a * 255.0).clip(0,255)
        a = np.round(a).astype('uint8')
    elif np.issubdtype(a.dtype, np.integer):
        a = a.astype('uint8')
    else:
        a = (a * 255.0).clip(0,255).astype('uint8')

    # ensure three channels
    if a.ndim == 3 and a.shape[2] == 3:
        return a
    # fallback: convert grayscale to RGB
    if a.ndim == 2:
        return np.stack([a,a,a], axis=-1)
    # otherwise try to reshape if channel first
    if a.ndim == 3 and a.shape[0] == 3:
        # channel-first -> HWC
        return np.transpose(a, (1,2,0)).astype('uint8')
    raise ValueError(f"Unsupported image shape in tensor_to_uint8: {a.shape}")

# ----------------- PSNR -----------------
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio for uint8 images (H,W,3).
    Returns PSNR in dB. If identical returns high value 100.0.
    """
    if a.dtype != np.uint8:
        a = tensor_to_uint8(a)
    if b.dtype != np.uint8:
        b = tensor_to_uint8(b)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    mse = np.mean((a_f - b_f) ** 2)
    if mse == 0:
        return 100.0
    PIXEL_MAX = 255.0
    return 20.0 * float(np.log10(PIXEL_MAX / np.sqrt(mse)))

# ----------------- robust SSIM -----------------
def _to_uint8(img):
    """Ensure numpy array is uint8 HxW or HxWx3."""
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a
    if a.max() <= 1.01:
        a = (a * 255.0).clip(0,255)
    a = np.round(a).astype('uint8')
    # drop alpha if present
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[...,:3]
    return a

def _single_channel_ssim(img1, img2, win_size=11, sigma=1.5, K1=0.01, K2=0.03):
    """SSIM for single channel images (float64)."""
    import numpy as np
    import cv2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    # gaussian blur
    if win_size % 2 == 0:
        win_size += 1
    mu1 = cv2.GaussianBlur(img1, (win_size, win_size), sigma)
    mu2 = cv2.GaussianBlur(img2, (win_size, win_size), sigma)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(img1 * img1, (win_size, win_size), sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (win_size, win_size), sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (win_size, win_size), sigma) - mu1_mu2
    L = 255.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    num = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    # avoid divide by zero
    ssim_map = np.ones_like(num)
    valid = den > 0
    ssim_map[valid] = num[valid] / den[valid]
    return float(np.mean(ssim_map))

def safe_ssim(a, b):
    """
    Robust SSIM wrapper. Inputs are numpy arrays HxW or HxWx3 in 0..255 or 0..1 or torch tensors.
    Returns float in [0,1] or 0.0 on error.
    """
    try:
        A = _to_uint8(a)
        B = _to_uint8(b)
        if A.shape != B.shape:
            B = cv2.resize(B, (A.shape[1], A.shape[0]), interpolation=cv2.INTER_AREA)
        # too small for window
        if min(A.shape[0], A.shape[1]) < 11:
            return 0.0
        if A.ndim == 3 and A.shape[2] == 3:
            s1 = _single_channel_ssim(A[...,0].astype(np.float64), B[...,0].astype(np.float64))
            s2 = _single_channel_ssim(A[...,1].astype(np.float64), B[...,1].astype(np.float64))
            s3 = _single_channel_ssim(A[...,2].astype(np.float64), B[...,2].astype(np.float64))
            return float((s1 + s2 + s3) / 3.0)
        else:
            if A.ndim == 3:
                A = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
                B = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
            return float(_single_channel_ssim(A.astype(np.float64), B.astype(np.float64)))
    except Exception:
        # return 0.0 on any error to keep training/validation robust
        return 0.0
