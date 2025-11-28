# src/damage_simulation.py
"""
Museum-grade strong damage simulator for textile / artifact images.

Usage:
    from damage_simulation import simulate_damage_strong
    damaged, mask = simulate_damage_strong(image_np, seed=42, strength='strong')

Outputs:
    damaged: uint8 HxWx3 (0..255)
    mask: uint8 HxW (0/1) where 1 means "damaged region" (useful for inpainting/refinement)
"""
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import cv2
import random
from typing import Tuple

def _rand_state(seed):
    rnd = random.Random()
    rnd.seed(seed)
    return rnd

def _make_elliptical_mask(h, w, center, axes, angle=0):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    return mask // 255

def _add_stain(img, rnd, severity=0.8):
    # add irregular translucent stain (brownish / water mark)
    h,w,_ = img.shape
    # random center near object area (we don't detect object; put stain anywhere)
    cx = rnd.randint(int(w*0.1), int(w*0.9))
    cy = rnd.randint(int(h*0.1), int(h*0.9))
    axes = (rnd.randint(int(w*0.05), int(w*0.25)), rnd.randint(int(h*0.05), int(h*0.25)))
    angle = rnd.randint(0,359)
    stain_mask = _make_elliptical_mask(h,w,(cx,cy),axes,angle)
    # apply gaussian-warped irregularities
    k = int(max(3, min(h,w) * 0.02))
    stain_blur = cv2.GaussianBlur(stain_mask.astype(np.float32), (k|1,k|1), sigmaX=axes[0]*0.08+1)
    stain_blur = cv2.normalize(stain_blur, None, 0.0, 1.0, cv2.NORM_MINMAX)
    # stain color (brown/orange/tea)
    stain_color = np.array([140, 100, 60], dtype=np.float32)  # BGR approx
    stain_color = stain_color[::-1]  # convert to RGB
    # blend
    alpha = (0.35 + severity*0.35) * stain_blur[...,None]
    out = img.astype(np.float32)
    out = out*(1-alpha) + alpha*stain_color[None,None,:]
    return out.astype(np.uint8), (stain_blur>0.05).astype(np.uint8)

def _add_mold_spots(img, rnd, severity=0.8):
    h,w,_ = img.shape
    mask = np.zeros((h,w), dtype=np.uint8)
    out = img.copy().astype(np.float32)
    # choose number of spots depending on severity
    n = rnd.randint(6, 30) if severity>0.6 else rnd.randint(3,15)
    for _ in range(n):
        # ensure radius > 0
        min_r = max(1, int(min(h,w)*0.008))
        max_r = max(2, int(min(h,w)*0.06))
        if max_r < min_r:
            max_r = min_r + 1
        r = rnd.randint(min_r, max_r)
        cx = rnd.randint(r, w-r-1) if w - 2*r -1 > 0 else rnd.randint(0, w-1)
        cy = rnd.randint(r, h-r-1) if h - 2*r -1 > 0 else rnd.randint(0, h-1)
        # safe distance map (avoid divide by zero)
        y_grid, x_grid = np.ogrid[:h,:w]
        # compute normalized squared distance; add epsilon to denominator
        denom = float(r*r) + 1e-6
        d = ((x_grid - cx)**2 + (y_grid - cy)**2) / denom
        # create spot (use nan_to_num to remove any accidental NaNs)
        spot = np.exp(-d)
        spot = cv2.GaussianBlur(spot.astype(np.float32), (0,0), sigmaX=max(1.0, r*0.2))
        spot = np.nan_to_num(spot, nan=0.0, posinf=0.0, neginf=0.0)
        # color and blend
        col = np.array([80 + rnd.randint(-15,15), 60 + rnd.randint(-10,10), 40 + rnd.randint(-10,10)], dtype=np.float32)
        alpha = np.clip(0.25 + severity*0.4, 0.1, 0.7) * spot[...,None]
        out = out*(1-alpha) + alpha*col[None,None,:]
        mask = np.maximum(mask, (spot>0.05).astype(np.uint8))
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8), mask


def _add_dust_and_specks(img, rnd, severity=0.7):
    h,w,_ = img.shape
    out = img.copy().astype(np.float32)
    mask = np.zeros((h,w), dtype=np.uint8)
    # random small specks
    num = int((h*w) / (2000) * (0.5 + severity))
    for _ in range(num):
        x = rnd.randint(0,w-1)
        y = rnd.randint(0,h-1)
        r = rnd.randint(1,4)
        cv2.circle(out, (x,y), r, (rnd.randint(50,120),)*3, -1)  # small dark speck
        cv2.circle(mask, (x,y), r+1, 1, -1)
    # light film: multiplicative haze
    film = np.ones((h,w,1), dtype=np.float32)
    if severity>0.4:
        haze = cv2.GaussianBlur((np.random.RandomState(rnd.randint(0,9999)).rand(h,w)*255).astype(np.uint8), (101,101), sigmaX=30)
        hz = (haze.astype(np.float32)/255.0 - 0.5) * (0.05*severity)
        film = 1.0 - hz[...,None]
        out = out * film
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8), mask

def _add_scratch_lines(img, rnd, severity=0.6):
    # add thin bright/dark scratches
    h,w,_ = img.shape
    out = img.copy()
    mask = np.zeros((h,w), dtype=np.uint8)
    num = rnd.randint(1,6) if severity>0.6 else rnd.randint(0,3)
    for _ in range(num):
        x1 = rnd.randint(0,w-1); y1 = rnd.randint(0,h-1)
        x2 = rnd.randint(0,w-1); y2 = rnd.randint(0,h-1)
        thickness = rnd.randint(1,3)
        color = (rnd.randint(30,220), rnd.randint(30,220), rnd.randint(30,220))
        cv2.line(out, (x1,y1), (x2,y2), color, thickness)
        cv2.line(mask, (x1,y1), (x2,y2), 1, thickness+1)
    # blur mask slightly
    mask = cv2.GaussianBlur(mask.astype(np.float32), (7,7), 0)
    mask = (mask>0.01).astype(np.uint8)
    return out, mask

def _add_occlusion_patch(img, rnd, severity=0.9):
    # add a solid/texture patch (like tape/patch)
    h,w,_ = img.shape
    out = img.copy().astype(np.float32)
    mask = np.zeros((h,w), dtype=np.uint8)
    # choose size relative to image
    rw = rnd.randint(int(w*0.08), int(w*0.35))
    rh = rnd.randint(int(h*0.06), int(h*0.25))
    x = rnd.randint(0, w - rw)
    y = rnd.randint(0, h - rh)
    patch = np.ones((rh, rw, 3), dtype=np.uint8) * np.array([rnd.randint(60,200), rnd.randint(60,200), rnd.randint(60,200)], dtype=np.uint8)
    # optionally textured patch (noise)
    if rnd.random() < 0.7:
        noise = (np.random.RandomState(rnd.randint(0,9999)).rand(rh, rw) * 80).astype(np.uint8)
        for c in range(3):
            patch[...,c] = np.clip(patch[...,c] + noise, 0, 255)
    # blend with edges softened
    k = max(3, int(min(rh, rw) * 0.06))
    alpha = np.ones((rh, rw), dtype=np.float32)
    border = cv2.GaussianBlur(alpha, (k|1,k|1), sigmaX= k*0.6)
    blended = out[y:y+rh, x:x+rw] * (1 - border[...,None]) + patch.astype(np.float32) * border[...,None]
    out[y:y+rh, x:x+rw] = blended
    mask[y:y+rh, x:x+rw] = 1
    return out.astype(np.uint8), mask

def _add_local_blur(img, rnd, severity=0.7):
    h,w,_ = img.shape
    out = img.copy()
    mask = np.zeros((h,w), dtype=np.uint8)
    # blur a region (simulate camera-smudge)
    rw = rnd.randint(int(w*0.06), int(w*0.25))
    rh = rnd.randint(int(h*0.06), int(h*0.25))
    x = rnd.randint(0, w-rw)
    y = rnd.randint(0, h-rh)
    region = out[y:y+rh, x:x+rw]
    k = int(max(3, min(rw, rh) * 0.06))
    b = cv2.GaussianBlur(region, (k|1,k|1), sigmaX=k*0.8+1)
    out[y:y+rh, x:x+rw] = b
    mask[y:y+rh, x:x+rw] = 1
    return out, mask

def simulate_damage_strong(img: np.ndarray, seed: int = 0, strength: str = 'strong', max_types: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    img: HxWx3 uint8 RGB
    seed: random seed
    strength: 'strong' or 'moderate' (controls severity scale)
    max_types: maximum number of damage types to apply (1..5)
    returns damaged_img (uint8), damage_mask (0/1)
    """
    rnd = _rand_state(seed)
    h,w,_ = img.shape
    severity = 1.0 if strength == 'strong' else 0.6

    # pool of damage generators with approximate weight
    generators = [
        ('stain', 1.0, _add_stain),
        ('occlusion', 0.9, _add_occlusion_patch),
        ('mold', 0.8, _add_mold_spots),
        ('dust', 0.6, _add_dust_and_specks),
        ('scratch', 0.5, _add_scratch_lines),
        ('blur', 0.5, _add_local_blur),
    ]

    # Always apply at least 1, choose up to max_types
    n_types = rnd.randint(max(1, max_types-1), max_types) if max_types>1 else 1
    chosen = rnd.sample(generators, k=n_types)

    out = img.copy().astype(np.uint8)
    overall_mask = np.zeros((h,w), dtype=np.uint8)
    # Apply in random order but stronger types earlier
    for name, weight, func in chosen:
        # small random chance to amplify severity
        local_sev = severity * (0.8 + rnd.random()*0.6)
        out, mask = func(out, rnd, local_sev)
        overall_mask = np.maximum(overall_mask, mask)

    # final subtle color fade / vignetting to simulate aging
    if rnd.random() < 0.6:
        # apply radial warm tint
        y,x = np.ogrid[:h,:w]
        cy = h//2 + rnd.randint(-int(h*0.1), int(h*0.1))
        cx = w//2 + rnd.randint(-int(w*0.1), int(w*0.1))
        maxr = np.hypot(max(cx,w-cx), max(cy,h-cy))
        rmap = np.hypot(x-cx, y-cy) / (maxr + 1e-6)
        tint = (0.02 + rnd.random()*0.06) * (1 - rmap**1.5)
        out = out.astype(np.float32)
        out[...,0] = np.clip(out[...,0] * (1 - tint), 0, 255)
        out[...,1] = np.clip(out[...,1] * (1 - tint*0.9), 0, 255)
        out[...,2] = np.clip(out[...,2] * (1 - tint*0.8), 0, 255)
        out = out.astype(np.uint8)

    return out, overall_mask

# small test helper
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python damage_simulation.py input.jpg output_damaged.png")
        sys.exit(1)
    inp = sys.argv[1]
    outp = sys.argv[2]
    im = np.array(Image.open(inp).convert('RGB'))
    dam, m = simulate_damage_strong(im, seed=42, strength='strong', max_types=4)
    Image.fromarray(dam).save(outp)
    Image.fromarray((m*255).astype(np.uint8)).save(outp.replace('.png','_mask.png'))
