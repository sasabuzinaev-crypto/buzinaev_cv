import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent


def first_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2, shape[1] + 2), dtype=bool)
    new_image[1:-1, 1:-1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)
    return np.max(labeled) - 1


def extract_features(region):
    holes = count_holes(region)

    h, w = region.image.shape

    density = region.image.sum() / region.image.size
    aspect = w / h
    aspect2 = min(h, w) / max(h, w)

    vlines = (np.sum(region.image, 0) == h).sum() / w
    hlines = (np.sum(region.image, 1) == w).sum() / h

    labeled = label(np.logical_not(region.image))
    bays = sum(1 for r in regionprops(labeled) if r.area > 3)

    return np.array([
        holes,
        region.eccentricity,
        aspect,
        aspect2,
        density,
        vlines,
        hlines,
        bays
    ], dtype=float)

def distance(a, b):
    return np.linalg.norm(a - b)


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


reference_vectors = {}
reference_labels_order = [
    "A", "B", "8", "0", "1", "W", "X", "/", "*", "-"
]
for region, label_name in zip(aprops, reference_labels_order):
    reference_vectors[label_name] = extract_features(region)

results = {}

image_path = BASE_DIR / "out"
image_path.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(5, 7))

for region in aprops:
    f = extract_features(region)

    best_label = None
    best_dist = float("inf")

    for label_name, ref_vec in reference_vectors.items():
        d = distance(f, ref_vec)
        if d < best_dist:
            best_dist = d
            best_label = label_name

    results[best_label] = results.get(best_label, 0) + 1

    plt.cla()
    plt.title(f"Class - '{best_label}'")
    plt.imshow(region.image)
    plt.savefig(image_path / f"image_{region.label}.png")

print(results)

plt.imshow(abinary)
plt.show()