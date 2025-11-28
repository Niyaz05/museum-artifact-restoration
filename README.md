# Artifact Damage Simulation & Restoration

This repository contains tools to simulate physical damage on artifact images (stains, mold, scratches, occlusions, dust), precompute damaged train/val/test sets, and train a ResNet34-based U-Net to restore images (repair/inpainting). It also includes inference utilities and helpers for dataset handling, evaluation (PSNR/SSIM), and saving checkpoints.

**Key Features**
- High-fidelity damage simulator for museum-grade artifacts (`src/damage_sim_museum.py` / `src/damage_simulation.py`).
- Precompute damaged datasets from clean images with masks (`scripts/precompute_damage.py`).
- Training pipeline using a ResNet34 UNet (`src/train_unet.py`) with optional perceptual loss.
- Inference utility to restore single images (`src/inference.py`).
- Support for precomputed masks and masked losses during training.

**Requirements**
- Python 3.8+ recommended
- See `requirements.txt` for full dependency list (torch, torchvision, numpy, pillow, opencv-python, scikit-image, diffusers, etc.).

**Project Layout**
- `data/`
  - `clean/` — original clean images (named by `obj_id`, e.g. `12345.jpg`).
  - `damaged/` — generated damaged images (organized `train/`, `val/`, `test/`).
  - `splits/` — CSVs listing `obj_id` values used for splits (e.g. `train_ids.csv`, `val_ids.csv`, `small_*` variants).
  - `metadata.csv` — (optional) original metadata CSV with column `obj_id`.
- `scripts/` — convenience scripts:
  - `create_small_split.py` — create a smaller subset of ids for quick experiments.
  - `check_one_file.py` — basic file existence checks.
  - `precompute_damage.py` — generate damaged images and masks for splits.
  - `small_check.py`, `find_missing_files.py`, `place_damaged_for_training.py`, etc.
- `src/` — main source code:
  - `train_unet.py` — training loop and checkpointing.
  - `inference.py` — single-image restore CLI.
  - `dataset.py` — `ArtifactDataset` and padding helpers.
  - `unet_model.py` — `ResNet34UNet` model implementation.
  - `utils.py` — PSNR/SSIM and image helpers.
  - `damage_sim_museum.py` (or `damage_simulation.py`) — damage simulator.
- `api/` — scripts for retrieving images from remote sources (e.g. `api/api.py` which contains API requests to fetch images from the MET website using `obj_id` values).
- `outputs/` — training outputs and checkpoints (e.g. `final_best/best_model.pth`, `metrics.csv`, `model_epoch_*.pth`).

**Quick Start — Installation**
Install dependencies (example using a virtualenv):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you use Conda, create and activate an environment and install the packages accordingly.

**Prepare data**
1. Place clean images under `data/clean/` named by id (e.g. `0001.jpg` or `12345.jpg`).
  Special: If you don't have local clean images, you can use `api/api.py` in the `api/` folder to fetch images from the MET website (or other APIs) using `obj_id` values. Run that script to populate `data/clean/` before creating splits. Ensure your `metadata.csv` or split CSVs contain the same `obj_id` identifiers used by the API.
2. Prepare CSV splits under `data/splits/` or generate a small subset for experiments:

```powershell
python scripts\create_small_split.py --metadata data\metadata.csv --out-dir data\splits --total 2000
```

**Precompute damaged images (optional)**
Use the simulator to create damaged train/val/test sets and write masks:

```powershell
python scripts\precompute_damage.py --clean-dir data\clean --train-csv data\splits\train_ids.csv --val-csv data\splits\val_ids.csv --out-root data\damaged --strength strong --max-types 4 --write-mask
```

This will produce `data/damaged/train/*.jpg` and corresponding `*_mask.png` files.

**Train**
Minimal training example (uses `data/damaged` for precomputed damaged images). If you prefer to simulate damage on the fly, run with `--damaged-dir` omitted and `simulate=True` in the dataset (see code comments).

```powershell
python src\train_unet.py --clean-dir data\clean --train-csv data\splits\train_ids.csv --val-csv data\splits\val_ids.csv --damaged-dir data\damaged\train --damaged-dir-val data\damaged\val --out-dir outputs\exp1 --epochs 40 --batch-size 4 --lr 3e-5 --num-workers 4
```

Important training flags (see `src/train_unet.py` for full list):
- `--pretrained` use ImageNet-pretrained ResNet34 backbone
- `--use-perceptual` enable perceptual (LPIPS-like) loss via CombinedLoss
- `--lam-l1`, `--lam-perc` control loss weights
- `--val-interval` how often to run validation

Checkpoints and metrics are written to `--out-dir` (e.g. `outputs/exp1/`). The best model is also written to `best_model.pth` when validation improves.

**Inference**
Restore a single image using a checkpoint:

```powershell
python src\inference.py --inp_path path\to\damaged.jpg --out_path path\to\restored.png --ckpt_path outputs\exp1\best_model.pth
```

The script pads images to be multiples of 32, runs the model, unpads and saves the output.

**Dataset, masks, and simulation**
- `src/dataset.py`'s `ArtifactDataset` supports two modes:
  - `simulate=True` (requires the simulator present): damage is generated on-the-fly
  - `simulate=False` and `--damaged-dir` provided: expects precomputed damaged images and optional masks named `<obj_id>_mask.*`.
- When masks are available, the training loop uses masked L1 loss over the damaged regions (recommended when you have accurate masks).

**Model Architecture**
- `src/unet_model.py` implements a ResNet34 encoder + UNet-style decoder (`ResNet34UNet`). The network outputs RGB images in 0..1 range and handles mild spatial mismatches.

**Evaluation**
- Utilities for PSNR and a robust SSIM are in `src/utils.py` (`psnr`, `safe_ssim`). Metrics are computed during validation and appended to `metrics.csv` in the output directory.

**Troubleshooting & Tips**
- GPU: training uses CUDA if available. Ensure `torch` + CUDA are installed matching your GPU.
- File naming: the dataset expects `obj_id` values in CSV to match filenames in `data/clean/` and `data/damaged/`. If filenames have suffixes (`_v1`, `_final`) the dataset has fallback glob lookup, but it's more robust to keep a consistent naming scheme.
- If you see `FileNotFoundError` for damaged images, check the `data/damaged` folder and CSV ids with `scripts/check_one_file.py`.

**Useful Scripts**
- `scripts\create_small_split.py` — generate small `small_*` CSVs for quick debug runs.
- `scripts\check_one_file.py` — basic checks to find a specific object id in damaged/train.
- `scripts\precompute_damage.py` — generate damaged images and masks for splits.
- `scripts\small_check.py` — project-specific checks (see file for details).

**Reproducibility**
- Seeds: `create_small_split.py`, `precompute_damage.py`, and dataset simulation support seeding for reproducibility. When precomputing damaged images, `--base-seed` and `seed` arguments allow stable generation.

**Dataset**
- check out the [Dataset](https://drive.google.com/drive/folders/1afB_2CwkfCRSbVh_Mth-VZGagd-LeGvp?usp=sharing)

**Outputs**

-This is the compairson between first epoch and final epoch 
<p float="left">
  <img src="https://github.com/user-attachments/assets/bb636bbd-260e-44f9-b218-a472c4619104" width="300" height="600" />

  <img src="https://github.com/user-attachments/assets/8cd9f10a-f1f2-4514-b483-4cdc0618df39" width="300" height="600" />
</p>

**Acknowledgements / Notes**
- This project aims to simulate and repair artifact damage for research and restoration assistance; tune the simulator and loss weights to match your domain and target artifact characteristics.

