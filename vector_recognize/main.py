import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import (
    label,
    regionprops,
)
from skimage.io import imread
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent


def first_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


save_path = BASE_DIR

DEBUG_8B = False
VLINES_B8_THRESHOLD = 0.2
ECCENTRICITY_0A_THRESHOLD = 0.63
ASPECT_DASH_THRESHOLD = 2.5


def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2, shape[1] + 2), dtype=bool)
    new_image[1:-1, 1:-1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)
    return np.max(labeled) - 1


def classificator(region):
    holes = count_holes(region)

    if holes == 2:  # B,8
        vlines = (np.sum(region.image, 0) == region.image.shape[0]).sum()
        vlines = vlines / region.image.shape[1]
        decision = "B" if vlines > VLINES_B8_THRESHOLD else "8"
        if DEBUG_8B:
            print(f"label={region.label} holes=2 vlines={vlines:.4f} -> {decision}")
        return decision

    if holes == 1:  # A,0
        if region.eccentricity > ECCENTRICITY_0A_THRESHOLD:
            return "0"
        return "A"

    # holes == 0: 1,W,X,*,-,/
    if region.image.sum() / region.image.size == 1.0:
        return "-"

    shape = region.image.shape
    aspect = shape[1] / shape[0]
    if aspect > ASPECT_DASH_THRESHOLD:
        return "-"

    aspect2 = np.min(shape) / np.max(shape)
    if aspect2 > 0.9:
        return "*"

    vlines = (np.sum(region.image, 0) == region.image.shape[0]).sum()
    hlines = (np.sum(region.image, 1) == region.image.shape[1]).sum()
    if vlines > 0 and hlines > 0:
        return "1"

    labeled = label(np.logical_not(region.image))
    bays = 0
    for r in regionprops(labeled):
        if r.area > 3:
            bays += 1
    if bays == 2:
        return "/"
    if bays == 4:
        return "X"
    if bays == 5:
        return "W"
    return "?"


if len(sys.argv) > 1:
    arg_path = Path(sys.argv[1]).expanduser()
    if arg_path.is_absolute():
        input_candidate = arg_path
    else:
        input_candidate = first_existing_path(Path.cwd() / arg_path, BASE_DIR / arg_path)
else:
    input_candidate = BASE_DIR / "alphabet.png"

input_path = first_existing_path(
    input_candidate,
    BASE_DIR / "alphabet.png",
    Path("alphabet.png"),
    BASE_DIR / "alphabet-small.png",
)
image = imread(str(input_path))[:, :, :-1]
abinary = image.mean(2) > 0

alabeled = label(abinary)
aprops = regionprops(alabeled)

results = {}

image_path = save_path / "out"
image_path.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(5, 7))

for region in aprops:
    symbol = classificator(region)

    results[symbol] = results.get(symbol, 0) + 1

    plt.cla()
    plt.title(f"Class - '{symbol}'")
    plt.imshow(region.image)
    plt.savefig(image_path / f"image_{region.label}.png")

print(results)

plt.imshow(abinary)
plt.show()
