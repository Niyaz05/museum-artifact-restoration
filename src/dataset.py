# src/dataset.py
import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

# Optional: your simulator (kept for backward compatibility)
try:
    from damage_sim_museum import simulate_damage_strong
except Exception:
    simulate_damage_strong = None

def pad_to_multiple_of_32(img: np.ndarray):
    h, w = img.shape[:2]
    nh = ((h + 31) // 32) * 32
    nw = ((w + 31) // 32) * 32
    pad_h = nh - h
    pad_w = nw - w
    pt = pad_h // 2
    pb = pad_h - pt
    pl = pad_w // 2
    pr = pad_w - pl
    img_padded = np.pad(img, ((pt, pb), (pl, pr), (0, 0)), mode='reflect')
    return img_padded, (pt, pb, pl, pr)

def unpad(img: np.ndarray, pad_info):
    pt, pb, pl, pr = pad_info
    h_end = None if pb == 0 else -pb
    w_end = None if pr == 0 else -pr
    return img[pt:h_end, pl:w_end]

class ArtifactDataset(Dataset):
    """
    Dataset supporting:
      - precomputed damaged images + optional masks (recommended)
      - OR simulate=True to generate damage on-the-fly (if simulate function available)

    Returns per item:
      inp_tensor [3,H,W], tgt_tensor [3,H,W], obj_id (str), pad_info (pt,pb,pl,pr), orig_size (h,w), mask_tensor_or_None (1,H,W float 0..1)
    """
    def __init__(self,
                 clean_dir: str,
                 metadata_csv: str,
                 damaged_dir: str = None,
                 ext: str = 'jpg',
                 simulate: bool = False,
                 sim_seed: int = None,
                 max_damage_types: int = 2):
        self.clean_dir = Path(clean_dir)
        self.damaged_dir = Path(damaged_dir) if damaged_dir else None
        self.ext = ext
        self.simulate = simulate
        self.sim_seed = sim_seed
        self.max_damage_types = max_damage_types

        df = pd.read_csv(metadata_csv)
        if 'obj_id' not in df.columns:
            # allow single-column CSV of ids (no header)
            if df.shape[1] == 1:
                ids = df.iloc[:,0].astype(str).tolist()
            else:
                raise ValueError("metadata CSV must contain 'obj_id' column or be a single-column list of ids")
        else:
            ids = df['obj_id'].astype(str).tolist()

        # strip whitespace from ids
        self.ids = [x.strip() for x in ids if str(x).strip() != ""]
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.ids)

    def _load_img_uint8(self, path: Path):
        img = Image.open(str(path)).convert('RGB')
        arr = np.array(img).astype(np.uint8)
        return arr

    def _find_damaged_path(self, damaged_dir: Path, obj: str):
        """
        Try various strategies to find a damaged image file for `obj`:
          - exact extension self.ext
          - common extensions
          - glob any file starting with obj (covers obj_final.jpg, obj_v1.png, etc.)
        Returns Path or None.
        """
        tried = []
        # 1) exact expected extension
        p = damaged_dir / f"{obj}.{self.ext}"
        tried.append(str(p))
        if p.exists():
            return p, tried

        # 2) common extensions
        for ext in ('jpg','jpeg','png','JPG','PNG'):
            p = damaged_dir / f"{obj}.{ext}"
            tried.append(str(p))
            if p.exists():
                return p, tried

        # 3) glob for anything starting with obj (covers obj_*.jpg etc.)
        matches = list(damaged_dir.glob(f"{obj}*"))
        tried.append(f"glob:{damaged_dir}/{obj}* -> found {len(matches)}")
        if matches:
            # prefer files with image-like extensions
            for m in matches:
                if m.suffix.lower() in ('.jpg','.jpeg','.png'):
                    return m, tried
            # otherwise return first
            return matches[0], tried

        return None, tried

    def __getitem__(self, idx):
        obj = self.ids[idx]
        fname = f"{obj}.{self.ext}"

        clean_path = self.clean_dir / fname
        if not clean_path.exists():
            # fallback to any extension
            found = list(self.clean_dir.glob(f"{obj}.*"))
            if not found:
                raise FileNotFoundError(f"Clean image not found for id {obj} at {clean_path}")
            clean_path = found[0]
        clean = self._load_img_uint8(clean_path)

        mask_arr = None
        if self.simulate:
            if simulate_damage_strong is None:
                raise RuntimeError("simulate=True but simulate_damage_strong not available")
            seed = None
            if self.sim_seed is not None:
                seed = int(self.sim_seed) + int(idx)
            damaged, mask = simulate_damage_strong(clean, seed=seed, max_damage_types=self.max_damage_types)
            # ensure mask is 0/1 single channel if returned
            if mask is not None:
                mask_arr = (mask.astype(np.uint8) > 0).astype(np.uint8) * 255
        else:
            if self.damaged_dir is None:
                raise ValueError("damaged_dir must be provided when simulate=False")
            damaged_path = self.damaged_dir / fname
            if not damaged_path.exists():
                # try to locate using helper (extensions/glob)
                found_path, tried = self._find_damaged_path(self.damaged_dir, obj)
                if found_path is None:
                    # helpful error with what was attempted
                    msg = (f"Damaged image for id {obj} not found in {self.damaged_dir}.\n"
                           f"Attempted: {', '.join(tried[:10])} ...\n"
                           f"Make sure filenames match obj_id values in CSV (extensions, suffixes, leading zeros).\n"
                           f"If filenames use a different pattern, either rename files or update CSV.")
                    raise FileNotFoundError(msg)
                else:
                    damaged_path = found_path

            damaged = self._load_img_uint8(damaged_path)

            # optional mask file named "<obj_id>_mask.*" (0/255) - try several patterns
            mask_arr = None
            possible_mask_names = [f"{obj}_mask", f"{obj}_mask.png", f"{obj}_mask.jpg"]
            # also try any file starting with obj and mask
            mask_found = None
            # direct check for common mask file
            for ext in ('png','jpg','jpeg','PNG','JPG'):
                p = self.damaged_dir / f"{obj}_mask.{ext}"
                if p.exists():
                    mask_found = p
                    break
            if mask_found is None:
                # glob search for pattern obj_mask*
                mask_matches = list(self.damaged_dir.glob(f"{obj}_mask*"))
                if mask_matches:
                    mask_found = mask_matches[0]

            if mask_found is not None:
                try:
                    mimg = Image.open(str(mask_found)).convert('L')
                    m_arr = np.array(mimg).astype(np.uint8)
                    # normalize: if mask is 0/255, convert to 0/255; if 0/1 convert to 0/255
                    if m_arr.max() <= 1:
                        m_arr = (m_arr * 255).astype(np.uint8)
                    mask_arr = (m_arr > 127).astype(np.uint8) * 255
                except Exception:
                    mask_arr = None
            else:
                mask_arr = None

        damaged_p, pad_info = pad_to_multiple_of_32(damaged)
        clean_p, _ = pad_to_multiple_of_32(clean)

        inp = self.to_tensor(damaged_p).float()  # 3,H,W  (0..1)
        tgt = self.to_tensor(clean_p).float()

        mask_tensor = None
        if mask_arr is not None:
            mask_p, _ = pad_to_multiple_of_32(mask_arr[..., None])  # H,W,1
            # convert to [1,H,W] float 0..1
            mask_tensor = torch.from_numpy(mask_p.transpose(2,0,1)).float() / 255.0

        return inp, tgt, obj, pad_info, clean.shape[:2], mask_tensor
