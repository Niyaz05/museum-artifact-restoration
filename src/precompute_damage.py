# scripts/precompute_damage.py
import os
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from damage_sim_museum import simulate_damage_strong   # or damage_sim_museum if that's your file
import argparse
from tqdm import tqdm

def precompute_for_split(clean_dir, ids_csv, out_dir, strength='strong', max_types=4, base_seed=0, write_mask=True):
    clean_dir = Path(clean_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ids_csv)
    # assume column 'obj_id'
    ids = df['obj_id'].astype(str).tolist()
    for i, oid in enumerate(tqdm(ids, desc=f"Precompute {out_dir.name}")):
        clean_path = clean_dir / f"{oid}.jpg"
        if not clean_path.exists():
            # try other extensions
            found = list(clean_dir.glob(f"{oid}.*"))
            if found:
                clean_path = found[0]
            else:
                tqdm.write(f"Missing clean file for {oid}, skipping.")
                continue
        out_path = out_dir / f"{oid}.jpg"
        mask_path = out_dir / f"{oid}_mask.png"
        if out_path.exists() and (not write_mask or mask_path.exists()):
            # already generated
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
    parser.add_argument('--train-csv', default='data/splits/train_ids.csv')
    parser.add_argument('--val-csv', default='data/splits/val_ids.csv')
    parser.add_argument('--test-csv', default='data/splits/test_ids.csv')
    parser.add_argument('--out-root', default='data/damaged')
    parser.add_argument('--strength', default='strong', choices=['strong','moderate'])
    parser.add_argument('--max-types', type=int, default=4)
    parser.add_argument('--base-seed', type=int, default=0)
    parser.add_argument('--write-mask', action='store_true')
    args = parser.parse_args()

    # train
    precompute_for_split(args.clean_dir, args.train_csv, os.path.join(args.out_root, 'train'), strength=args.strength, max_types=args.max_types, base_seed=args.base_seed)
    # val
    precompute_for_split(args.clean_dir, args.val_csv, os.path.join(args.out_root, 'val'), strength=args.strength, max_types=args.max_types, base_seed=args.base_seed+100000)
    # test
    if Path(args.test_csv).exists():
        precompute_for_split(args.clean_dir, args.test_csv, os.path.join(args.out_root, 'test'), strength=args.strength, max_types=args.max_types, base_seed=args.base_seed+200000)
