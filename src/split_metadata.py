# src/split_metadata.py
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def split_metadata(csv_path, out_dir='data/splits', seed=42, train_frac=0.8, obj_col='obj_id'):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    if obj_col not in df.columns:
        raise ValueError(f"metadata CSV must contain column '{obj_col}'")
    ids = pd.Series(df[obj_col].astype(str).values).unique().tolist()
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_frac)
    n_val = n - n_train
    train_ids = ids[:n_train]
    val_ids = ids[n_train:]
    pd.DataFrame({'obj_id': train_ids}).to_csv(out_dir / 'train_ids.csv', index=False)
    pd.DataFrame({'obj_id': val_ids}).to_csv(out_dir / 'val_ids.csv', index=False)
    print(f"Total unique ids: {n}")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
    # show small sample for quick sanity check
    print("Train sample (first 5):", train_ids[:5])
    print("Val sample (first 5):", val_ids[:5])
    return out_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='data/metadata.csv')
    parser.add_argument('--out-dir', default='data/splits')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-frac', type=float, default=0.8)
    parser.add_argument('--obj-col', type=str, default='obj_id')
    args = parser.parse_args()
    split_metadata(args.csv, out_dir=args.out_dir, seed=args.seed, train_frac=args.train_frac, obj_col=args.obj_col)
