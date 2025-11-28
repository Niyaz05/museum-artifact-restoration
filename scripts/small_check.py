# src/check_simulator.py
from PIL import Image
import numpy as np
from damage_sim_museum import simulate_damage_strong
import os

os.makedirs('outputs/check_sim', exist_ok=True)
sample = 'data/clean'  # or path to one image
# pick a few images
from glob import glob
files = glob('data/clean/*.jpg')[:8]
for i,f in enumerate(files):
    im = np.array(Image.open(f).convert('RGB'))
    dam, mask = simulate_damage_strong(im, seed=i, strength='strong', max_types=4)
    Image.fromarray(dam).save(f'outputs/check_sim/{i}_dam.png')
    Image.fromarray(im).save(f'outputs/check_sim/{i}_clean.png')
print('Wrote outputs/check_sim/* - inspect for NaNs and proper damage.')
