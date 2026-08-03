from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from skimage import data
from skimage.transform import radon

from core import ANGLES, CROP_CENTERS, CROP_SIDE, SEED, SIZE, vessel_patch

def main():
    root = Path(__file__).parent.parent
    output = root / "data"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    retina = data.retina().astype(np.float64) / 255.0
    green = retina[..., 1]
    truth = np.stack([vessel_patch(green, center) for center in CROP_CENTERS])
    clean = []
    noisy = []
    angle_grid = np.arange(180, dtype=float)
    for image in truth:
        sinogram = radon(image, theta=angle_grid, circle=True, preserve_range=True)
        noise_sd = 0.008 * np.std(sinogram)
        noisy_sinogram = sinogram + rng.normal(0, noise_sd, size=sinogram.shape)
        clean.append(sinogram[:, ANGLES.astype(int)])
        noisy.append(noisy_sinogram[:, ANGLES.astype(int)])
    np.savez_compressed(
        output / "retina.npz",
        truth=truth,
        clean_sinogram=np.stack(clean),
        noisy_sinogram=np.stack(noisy),
        angles_deg=ANGLES,
    )
    source = {
        "dataset": "skimage.data.retina",
        "source": "Mikael Haggstrom, Medical gallery of Mikael Haggstrom 2014, WikiJournal of Medicine 1(2), doi:10.15347/wjm/2014.008",
        "license": "CC0 1.0 Universal Public Domain Dedication",
        "crop_centers_row_col": [list(center) for center in CROP_CENTERS],
        "crop_side": CROP_SIDE,
        "output_size": SIZE,
        "seed": SEED,
        "angles_deg": ANGLES.astype(int).tolist(),
    }
    (output / "source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
