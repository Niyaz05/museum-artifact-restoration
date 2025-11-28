# scripts/place_damaged_for_training.py
from pathlib import Path
import shutil

# CSVs and target folders (adjust if your paths differ)
train_csv = Path("data/splits/small_train_ids.csv")
val_csv   = Path("data/splits/small_val_ids.csv")
dst_train = Path("data/damaged/train")
dst_val   = Path("data/damaged/val")

# search root: where to look for existing damaged files
search_root = Path(".")  # repo root; will rglob recursively

# image and mask extensions to consider
img_exts = ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']
mask_suffixes = ['_mask', '_mask.png', '_mask.jpg', '_mask.jpeg']

dst_train.mkdir(parents=True, exist_ok=True)
dst_val.mkdir(parents=True, exist_ok=True)

def read_ids(csv_path):
    ids = []
    with open(csv_path, 'r', newline='') as f:
        for line in f:
            v = line.strip()
            if not v or v.lower() == 'obj_id':
                continue
            ids.append(v)
    return ids

def find_first_image_for_id(root: Path, obj: str):
    # prefer files under any damaged folder first
    # search order: data/damaged/**, data/damaged_*/**, data/**, everything
    search_priority = [
        Path("data/damaged"),
        Path("data/damaged_small"),
        Path("data/damaged_small/train"),
        Path("data/damaged_small/val"),
        Path("data/damaged/val"),
        Path("data/damaged/train")
    ]
    seen = set()
    for base in search_priority + [root]:
        if not base.exists():
            continue
        # try exact-name with common exts
        for ext in img_exts:
            p = base / f"{obj}{ext}"
            if p.exists() and p.is_file():
                return p
        # try glob under base
        for p in base.rglob(f"{obj}*"):
            if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png'):
                return p
        # avoid repeating same nodes
    # global fallback
    for p in root.rglob(f"{obj}*"):
        if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png'):
            return p
    return None

def copy_mask_if_exists(src_img_path: Path, dst_dir: Path, obj: str):
    # check neighboring mask files near src image or anywhere under same parent
    parent = src_img_path.parent
    # check same folder for obj_mask.*
    for ext in img_exts:
        cand = parent / f"{obj}_mask{ext}"
        if cand.exists():
            shutil.copy(cand, dst_dir / cand.name)
            return True
    # global search for obj_mask*
    matches = list(parent.glob(f"{obj}_mask*"))
    for m in matches:
        if m.is_file():
            shutil.copy(m, dst_dir / m.name)
            return True
    # also try searching repo-wide (slower)
    for ext in img_exts:
        g = list(Path(".").rglob(f"{obj}_mask{ext}"))
        if g:
            shutil.copy(g[0], dst_dir / g[0].name)
            return True
    return False

def ensure_copied(ids, dst_dir):
    copied = 0
    not_found = []
    for obj in ids:
        dst_target = dst_dir / f"{obj}.jpg"  # name preserved from src when copying; we use original name
        if dst_target.exists():
            # already present
            continue
        src = find_first_image_for_id(search_root, obj)
        if src is None:
            not_found.append(obj)
            continue
        # copy the source image into dst_dir (keep original filename)
        shutil.copy(src, dst_dir / src.name)
        copy_mask_if_exists(src, dst_dir, obj)
        copied += 1
    return copied, not_found

train_ids = read_ids(train_csv) if train_csv.exists() else []
val_ids   = read_ids(val_csv)   if val_csv.exists()   else []

print(f"Train ids: {len(train_ids)}, Val ids: {len(val_ids)}")
copied_train, not_found_train = ensure_copied(train_ids, dst_train)
copied_val, not_found_val = ensure_copied(val_ids, dst_val)

print("Copied summary:")
print(f"  train: copied {copied_train}, not found {len(not_found_train)}")
print(f"  val:   copied {copied_val}, not found {len(not_found_val)}")

if not_found_train:
    print("Train ids not found (sample 50):", not_found_train[:50])
if not_found_val:
    print("Val ids not found (sample 50):", not_found_val[:50])

# final quick counts
print("Final counts in target folders:")
print("  data/damaged/train:", len(list(dst_train.glob("*.*"))))
print("  data/damaged/val:  ", len(list(dst_val.glob("*.*"))))
