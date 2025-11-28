# src/train_unet.py
import os
import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import time
import csv
from typing import List, Tuple, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import torchvision.utils as vutils
from PIL import Image
import numpy as np

# local
from dataset import ArtifactDataset, unpad
from unet_model import ResNet34UNet
from utils import tensor_to_uint8, ensure_dir, psnr, safe_ssim
from losses import CombinedLoss

# ---------------- collate (with mask support) ----------------
def collate_batch(samples: List[Tuple[torch.Tensor, torch.Tensor, Any, Tuple[int,int,int,int], Tuple[int,int], Optional[torch.Tensor]]]):
    if len(samples) == 0:
        return None
    hs = [s[0].shape[1] for s in samples]
    ws = [s[0].shape[2] for s in samples]
    H_max = max(hs)
    W_max = max(ws)

    inp_list, tgt_list, obj_ids, new_pad_infos, orig_sizes = [], [], [], [], []
    mask_list = []
    has_mask = False

    for inp, tgt, obj_id, pad_info, orig_size, mask in samples:
        c, h, w = inp.shape
        pad_h = H_max - h
        pad_w = W_max - w
        pad_top_extra = pad_h // 2
        pad_bottom_extra = pad_h - pad_top_extra
        pad_left_extra = pad_w // 2
        pad_right_extra = pad_w - pad_left_extra
        pad_vals = (int(pad_left_extra), int(pad_right_extra), int(pad_top_extra), int(pad_bottom_extra))

        safe_reflect_vert = (pad_top_extra < h) and (pad_bottom_extra < h)
        safe_reflect_horz = (pad_left_extra < w) and (pad_right_extra < w)
        use_reflect = safe_reflect_vert and safe_reflect_horz

        if use_reflect:
            inp_p = F.pad(inp, pad_vals, mode='reflect')
            tgt_p = F.pad(tgt, pad_vals, mode='reflect')
        else:
            try:
                mean_val = float(inp.mean().cpu().item())
            except Exception:
                mean_val = 0.0
            inp_p = F.pad(inp, pad_vals, mode='constant', value=mean_val)
            tgt_p = F.pad(tgt, pad_vals, mode='constant', value=mean_val)

        # masks
        if mask is None:
            mask_p = None
        else:
            has_mask = True
            m = mask
            if not isinstance(m, torch.Tensor):
                m = torch.tensor(m, dtype=torch.float32)
            if m.ndim == 2:
                m = m[None, ...]
            mask_p = F.pad(m.float(), pad_vals, mode='constant', value=0.0)

        pt, pb, pl, pr = pad_info
        pt_new = int(pt + pad_top_extra)
        pb_new = int(pb + pad_bottom_extra)
        pl_new = int(pl + pad_left_extra)
        pr_new = int(pr + pad_right_extra)

        inp_list.append(inp_p)
        tgt_list.append(tgt_p)
        obj_ids.append(obj_id)
        new_pad_infos.append((pt_new, pb_new, pl_new, pr_new))
        orig_sizes.append(orig_size)
        mask_list.append(mask_p)

    batch_inp = torch.stack(inp_list, dim=0)
    batch_tgt = torch.stack(tgt_list, dim=0)

    if has_mask:
        for i, m in enumerate(mask_list):
            if m is None:
                mask_list[i] = torch.zeros((1, H_max, W_max), dtype=torch.float32)
            else:
                if not isinstance(m, torch.Tensor):
                    mask_list[i] = torch.tensor(m, dtype=torch.float32)
        mask_batch = torch.stack(mask_list, dim=0)
    else:
        mask_batch = None

    return batch_inp, batch_tgt, obj_ids, new_pad_infos, orig_sizes, mask_batch

# ---------------- helpers ----------------
def save_unpadded_sample(epoch_dir, obj_id, inp_tensor, out_tensor, tgt_tensor, pad_info, orig_size):
    ensure_dir(epoch_dir)
    inp_arr = tensor_to_uint8(inp_tensor)
    out_arr = tensor_to_uint8(out_tensor)
    tgt_arr = tensor_to_uint8(tgt_tensor)
    inp_unp = unpad(inp_arr, pad_info)
    out_unp = unpad(out_arr, pad_info)
    tgt_unp = unpad(tgt_arr, pad_info)
    oh, ow = orig_size
    import cv2
    if inp_unp.shape[:2] != (oh, ow):
        inp_unp = cv2.resize(inp_unp, (ow, oh))
        out_unp = cv2.resize(out_unp, (ow, oh))
        tgt_unp = cv2.resize(tgt_unp, (ow, oh))
    Image.fromarray(inp_unp).save(os.path.join(epoch_dir, f"{obj_id}_input.png"))
    Image.fromarray(out_unp).save(os.path.join(epoch_dir, f"{obj_id}_output.png"))
    Image.fromarray(tgt_unp).save(os.path.join(epoch_dir, f"{obj_id}_target.png"))

def write_metrics_csv(path, row=None, header=False):
    ensure_dir(os.path.dirname(path))
    write_header = header and not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['epoch','train_loss','val_psnr','val_ssim','epoch_time_s'])
        if row is not None:
            if not hasattr(row, '__iter__') or isinstance(row, (str, bytes)):
                row = [row]
            writer.writerow(row)

# ---------------- training ----------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    if torch.cuda.is_available():
        try:
            print("GPU:", torch.cuda.get_device_name(0))
            print("CUDA total_memory:", torch.cuda.get_device_properties(0).total_memory)
        except Exception:
            pass

    torch.backends.cudnn.benchmark = True

    # Datasets: pass damaged dir(s) (may be same or different)
    train_ds = ArtifactDataset(
        clean_dir=args.clean_dir,
        metadata_csv=args.train_csv,
        damaged_dir=args.damaged_dir,
        ext=args.ext,
        simulate=False
    )
    val_ds = ArtifactDataset(
        clean_dir=args.clean_dir,
        metadata_csv=args.val_csv,
        damaged_dir=(args.damaged_dir_val if args.damaged_dir_val else args.damaged_dir),
        ext=args.ext,
        simulate=False
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size), shuffle=False,
                            num_workers=max(0, args.num_workers), pin_memory=True, collate_fn=collate_batch)

    model = ResNet34UNet(pretrained=args.pretrained).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=3)

    # Criterion: CombinedLoss if perceptual else L1
    if args.use_perceptual:
        try:
            criterion = CombinedLoss(lam_l1=args.lam_l1, lam_perc=args.lam_perc, lam_edge=args.lam_edge, lam_tv=args.lam_tv).to(device)
        except TypeError:
            criterion = CombinedLoss(lam_perc=args.lam_perc, lam_l1=args.lam_l1).to(device)
    else:
        criterion = None  # we'll use functional l1 or masked l1

    ensure_dir(args.out_dir)
    metrics_path = os.path.join(args.out_dir, 'metrics.csv')
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_ssim = -1.0
    write_metrics_csv(metrics_path, None, header=True)

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            inp, tgt, obj_ids, pad_infos, orig_sizes, mask_batch = batch
            inp = inp.to(device)
            tgt = tgt.to(device)
            if mask_batch is not None:
                mask_batch = mask_batch.to(device)

            opt.zero_grad()
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.amp.autocast(device_type, enabled=torch.cuda.is_available()):
                out = model(inp)
                # ensure same spatial size
                if out.shape[2:] != tgt.shape[2:]:
                    tgt_resized = torch.nn.functional.interpolate(tgt, size=out.shape[2:], mode='bilinear', align_corners=False)
                    if mask_batch is not None and mask_batch.shape[2:] != out.shape[2:]:
                        mask_resized = torch.nn.functional.interpolate(mask_batch, size=out.shape[2:], mode='nearest')
                    else:
                        mask_resized = mask_batch
                else:
                    tgt_resized = tgt
                    mask_resized = mask_batch

                # masked L1 if mask provided
                if mask_resized is not None:
                    # mask_resized shape: [B,1,H,W], out/tgt_resized: [B,3,H,W]
                    eps = 1e-6
                    diff = (out - tgt_resized).abs()  # B,3,H,W
                    # expand mask to channels
                    mask_c = mask_resized.repeat(1, out.shape[1], 1, 1)
                    masked_sum = (diff * mask_c).sum()
                    denom = (mask_c.sum() * 1.0) + eps
                    l1_masked = masked_sum / denom
                    if args.use_perceptual and hasattr(criterion, 'perc'):
                        perc = criterion.perc(out, tgt_resized)
                        loss = args.lam_l1 * l1_masked + args.lam_perc * perc
                    else:
                        loss = args.lam_l1 * l1_masked
                else:
                    # full-image loss
                    if args.use_perceptual and hasattr(criterion, 'perc'):
                        perc = criterion.perc(out, tgt_resized)
                        l1 = torch.nn.functional.l1_loss(out, tgt_resized)
                        loss = args.lam_l1 * l1 + args.lam_perc * perc
                    else:
                        loss = torch.nn.functional.l1_loss(out, tgt_resized)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), out_shape=str(out.shape))

        avg_loss = total_loss / max(1, len(train_loader))
        scheduler.step(avg_loss)

        # checkpoint
        ckpt_path = os.path.join(args.out_dir, f"model_epoch_{epoch}.pth")
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch}, ckpt_path)

        # validation
        avg_val_psnr = 0.0
        avg_val_ssim = 0.0
        val_count = 0
        val_errors = []
        if (epoch % args.val_interval) == 0:
            model.eval()
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                    try:
                        inp_v, tgt_v, obj_v, pad_v, orig_v, mask_v = batch
                        inp_v = inp_v.to(device)
                        tgt_v = tgt_v.to(device)
                        if mask_v is not None:
                            mask_v = mask_v.to(device)

                        out_v = model(inp_v)
                        if out_v.shape[2:] != tgt_v.shape[2:]:
                            tgt_v = torch.nn.functional.interpolate(tgt_v, size=out_v.shape[2:], mode='bilinear', align_corners=False)
                            if mask_v is not None and mask_v.shape[2:] != out_v.shape[2:]:
                                mask_v = torch.nn.functional.interpolate(mask_v, size=out_v.shape[2:], mode='nearest')

                        bs = out_v.shape[0]
                        for i in range(bs):
                            try:
                                out_np = tensor_to_uint8(out_v[i].cpu())
                                tgt_np = tensor_to_uint8(tgt_v[i].cpu())
                                pad_info = pad_v[i]
                                out_unp = unpad(out_np, pad_info)
                                tgt_unp = unpad(tgt_np, pad_info)
                                if out_unp.shape != tgt_unp.shape:
                                    import cv2
                                    out_unp = cv2.resize(out_unp, (tgt_unp.shape[1], tgt_unp.shape[0]))
                                p = psnr(tgt_unp, out_unp)
                                s = safe_ssim(tgt_unp, out_unp)
                                avg_val_psnr += p
                                avg_val_ssim += (s if s >= 0 else 0.0)
                                val_count += 1
                                if args.max_val_samples > 0 and val_count >= args.max_val_samples:
                                    break
                            except Exception as ee:
                                val_errors.append(str(ee))
                        if args.max_val_samples > 0 and val_count >= args.max_val_samples:
                            break
                    except Exception as e:
                        val_errors.append(str(e))
            print(f"Validation finished: samples={val_count}, errors={len(val_errors)}, avg_psnr={avg_val_psnr:.4f}, avg_ssim={avg_val_ssim:.4f}")
        else:
            print(f"Skipping validation this epoch (val_interval={args.val_interval})")

        if val_count > 0:
            avg_val_psnr = avg_val_psnr / val_count
            avg_val_ssim = avg_val_ssim / val_count
        else:
            avg_val_psnr = 0.0
            avg_val_ssim = 0.0

        epoch_time = time.time() - t0
        print(f"Epoch {epoch} train_loss {avg_loss:.6f} val_psnr {avg_val_psnr:.4f} val_ssim {avg_val_ssim:.4f} epoch_time_s {epoch_time:.1f}")
        write_metrics_csv(os.path.join(args.out_dir, 'metrics.csv'), [epoch, avg_loss, avg_val_psnr, avg_val_ssim, int(epoch_time)], header=False)

        # previews (full-res forward)
        with torch.no_grad():
            try:
                inp0, tgt0, fn0, pad0, orig0, mask0 = next(iter(train_loader))
                inp0_full = inp0.to(device)
                out0 = model(inp0_full)
                bs = inp0_full.size(0)
                grid_list = []
                for i in range(bs):
                    grid_list.extend([inp0_full[i].cpu(), out0[i].cpu(), tgt0[i].cpu()])
                grid = vutils.make_grid(torch.stack(grid_list), nrow=3)
                vutils.save_image(grid, os.path.join(args.out_dir, f'preview_epoch_{epoch}.png'))
                epoch_dir = os.path.join(args.out_dir, f'epoch_{epoch}')
                for i in range(min(args.samples_per_epoch, bs)):
                    save_unpadded_sample(epoch_dir, fn0[i], inp0[i].cpu(), out0[i].cpu(), tgt0[i].cpu(), pad0[i], orig0[i])
            except Exception as e:
                print("Preview save failed:", e)

        if avg_val_ssim > best_val_ssim:
            best_val_ssim = avg_val_ssim
            best_path = os.path.join(args.out_dir, 'best_model.pth')
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch, 'val_ssim': avg_val_ssim}, best_path)
            print(f"Saved NEW best model at epoch {epoch} val_ssim={avg_val_ssim:.4f}")

    print("Training complete. Best val SSIM:", best_val_ssim)

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean-dir', default='data/clean')
    parser.add_argument('--train-csv', default='data/splits/train_ids.csv')
    parser.add_argument('--val-csv', default='data/splits/val_ids.csv')
    parser.add_argument('--ext', default='jpg')
    parser.add_argument('--out-dir', default='outputs')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--max-damage-types', type=int, default=2)
    parser.add_argument('--samples-per-epoch', type=int, default=4)
    parser.add_argument('--use-perceptual', action='store_true')
    parser.add_argument('--lam-perc', type=float, default=0.8)
    parser.add_argument('--lam-l1', type=float, default=1.0)
    parser.add_argument('--lam-edge', type=float, default=1.0)
    parser.add_argument('--lam-tv', type=float, default=0.0)
    parser.add_argument('--val-interval', type=int, default=1)
    parser.add_argument('--max-val-samples', type=int, default=500, help='0 = all')
    parser.add_argument('--resize-max', type=int, default=0, help='deprecated - avoid resizing')
    # new damaged dir args:
    parser.add_argument('--damaged-dir', type=str, default=None, help='Folder with precomputed damaged TRAIN images')
    parser.add_argument('--damaged-dir-val', type=str, default=None, help='Folder with precomputed damaged VAL images (optional)')
    args = parser.parse_args()

    # backwards compat: support metadata_csv names if present
    args.train_csv = getattr(args, 'train_csv', args.__dict__.get('metadata_csv', 'data/splits/train_ids.csv'))
    args.val_csv = getattr(args, 'val_csv', args.__dict__.get('val_csv', 'data/splits/val_ids.csv'))

    train(args)
