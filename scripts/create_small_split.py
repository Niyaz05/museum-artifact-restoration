# scripts/create_small_split.py
import pandas as pd
import random
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--metadata', default='data/metadata.csv', help='CSV with column obj_id')
parser.add_argument('--out-dir', default='data/splits', help='where to write small splits')
parser.add_argument('--total', type=int, default=2000, help='total images in subset')
parser.add_argument('--seed', type=int, default=42, help='random seed (reproducible)')
parser.add_argument('--train-ratio', type=float, default=0.8, help='train/val split ratio')
args = parser.parse_args()

df = pd.read_csv(args.metadata)
if 'obj_id' not in df.columns:
    raise SystemExit("metadata.csv must include 'obj_id' column")

ids = df['obj_id'].astype(str).tolist()
if len(ids) < args.total:
    raise SystemExit(f"You requested {args.total} but metadata has only {len(ids)} ids")

random.seed(args.seed)
sampled = random.sample(ids, args.total)

n_train = int(round(len(sampled) * args.train_ratio))
train = sampled[:n_train]
val = sampled[n_train:]

outdir = Path(args.out_dir)
outdir.mkdir(parents=True, exist_ok=True)

pd.DataFrame({'obj_id': sampled}).to_csv(outdir / 'small_all_ids.csv', index=False)
pd.DataFrame({'obj_id': train}).to_csv(outdir / 'small_train_ids.csv', index=False)
pd.DataFrame({'obj_id': val}).to_csv(outdir / 'small_val_ids.csv', index=False)

print(f"Wrote {len(train)} train and {len(val)} val ids to {outdir}")
