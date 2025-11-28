# scripts/precompute_damage_subset.py
import os
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from damage_sim_museum import simulate_damage_strong   # or damage_sim_museum
import argparse
from tqdm import tqdm

def precompute_list(clean_dir, ids_csv, out_dir, strength='strong', max_types=4, base_seed=0, write_mask=True):
    clean_dir = Path(clean_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ids_csv)
    ids = df['obj_id'].astype(str).tolist()
    for i, oid in enumerate(tqdm(ids, desc=f"Precompute {out_dir.name}")):
        # prefer .jpg then fallback
        found = list(clean_dir.glob(f"{oid}.*"))
        if not found:
            tqdm.write(f"Missing clean image for {oid}, skipping")
            continue
        clean_path = found[0]
        out_path = out_dir / f"{oid}.jpg"
        mask_path = out_dir / f"{oid}_mask.png"
        if out_path.exists() and (not write_mask or mask_path.exists()):
            continue
        img = np.array(Image.open(str(clean_path)).convert('RGB'))
        seed = base_seed + i
        damaged, mask = simulate_damage_strong(img, seed=seed, strength=strength, max_types=max_types)
        Image.fromarray(damaged).save(str(out_path))
        if write_mask:
            Image.fromarray((mask*255).astype('uint8')).save(str(mask_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean-dir', default='data/clean')
    parser.add_argument('--train-csv', default='data/splits/small_train_ids.csv')
    parser.add_argument('--val-csv', default='data/splits/small_val_ids.csv')
    parser.add_argument('--out-root', default='data/damaged_small')
    parser.add_argument('--strength', default='strong', choices=['strong','moderate'])
    parser.add_argument('--max-types', type=int, default=4)
    parser.add_argument('--base-seed', type=int, default=0)
    parser.add_argument('--write-mask', action='store_true')
    args = parser.parse_args()

    precompute_list(args.clean_dir, args.train_csv, os.path.join(args.out_root, 'train'), strength=args.strength, max_types=args.max_types, base_seed=args.base_seed, write_mask=args.write_mask)
    precompute_list(args.clean_dir, args.val_csv, os.path.join(args.out_root, 'val'), strength=args.strength, max_types=args.max_types, base_seed=args.base_seed+100000, write_mask=args.write_mask)
