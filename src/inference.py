# src/inference.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import argparse
import torch
from PIL import Image
import numpy as np
from unet_model import ResNet34UNet
from dataset import pad_to_multiple_of_32, unpad
from utils import tensor_to_uint8, ensure_dir
from torchvision import transforms


def load_model(ckpt_path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet34UNet(pretrained=False).to(device)
    st = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(st.get('model', st))
    model.eval()
    return model, device


def unet_restore(inp_path, out_path, ckpt_path):
    model, device = load_model(ckpt_path)

    img = Image.open(inp_path).convert('RGB')
    arr = np.array(img)
    arr_p, pad_info = pad_to_multiple_of_32(arr)

    inp = transforms.ToTensor()(arr_p).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(inp)

    out_np = tensor_to_uint8(out[0])
    out_unp = unpad(out_np, pad_info)

    ensure_dir(Path(out_path).parent)
    Image.fromarray(out_unp).save(out_path)
    print("Saved", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--ckpt_path", required=True)
    args = parser.parse_args()

    unet_restore(args.inp_path, args.out_path, args.ckpt_path)
