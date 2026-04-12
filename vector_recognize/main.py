import matplotlib.pyplot as plt
import numpy as np
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


save_path = BASE_DIR

DEBUG_8B = False
MIRROR_8B_THRESHOLD = None


def mirror_diff(image: np.ndarray) -> float:
    image = image.astype(bool)
    on = int(image.sum())
    if on == 0:
        return 1.0

    w = image.shape[1]
    half = w // 2

    left = image[:, :half]
    right = image[:, w - half:]
    right = np.fliplr(right)

    mismatch = np.logical_xor(left, right).sum()
    return float(mismatch) / float(on)


def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2, shape[1] + 2), dtype=bool)
    new_image[1:-1, 1:-1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)
    return np.max(labeled) - 1


def extractor(region):
    cy, cx = region.centroid_local
    cy /= region.image.shape[0]
    cx /= region.image.shape[1]

    perimeter = region.perimeter / region.image.size
    holes = count_holes(region)

    vlines = np.mean(region.image, axis=0).sum()
    hlines = np.mean(region.image, axis=1).sum()

    eccentricity = region.eccentricity
    aspect = region.image.shape[1] / region.image.shape[0]

    return np.array([
        region.area / region.image.size,
        cy, cx,
        perimeter,
        holes,
        vlines,
        hlines,
        eccentricity,
        aspect
    ])


def classificator(region, templates):
    if count_holes(region) == 2:
        diff = mirror_diff(region.image)
        thr = MIRROR_8B_THRESHOLD if MIRROR_8B_THRESHOLD is not None else 0.25
        decision = "8" if diff <= thr else "B"

        if DEBUG_8B:
            print(
                f"label={region.label} holes=2 mirror_diff={diff:.4f} "
                f"thr={thr:.4f} -> {decision}"
            )
        return decision

    features = extractor(region)

    result = ""
    min_d = float("inf")

    for symbol, t in templates.items():
        d = np.linalg.norm(t - features)
        if d < min_d:
            min_d = d
            result = symbol

    return result


template_path = first_existing_path(
    BASE_DIR / "alphabet-small.png",
    BASE_DIR / "alphabet_small.png",
    Path("alphabet-small.png"),
)
template = imread(str(template_path))[:, :, :-1]
template = template.sum(2)

binary = template != 765
labeled = label(binary)
props = regionprops(labeled)

templates = {}
mirror_templates = {}

for region, symbol in zip(
    props,
    ["8", "O", "A", "B", "1", "W", "X", "*", "/", "-"]
):
    templates[symbol] = extractor(region)
    mirror_templates[symbol] = mirror_diff(region.image)

if ("8" in mirror_templates) and ("B" in mirror_templates):
    MIRROR_8B_THRESHOLD = (
        mirror_templates["8"] + mirror_templates["B"]
    ) / 2.0

if len(sys.argv) > 1:
    arg_path = Path(sys.argv[1]).expanduser()
    input_candidate = (arg_path if arg_path.is_absolute() else (BASE_DIR / arg_path)).resolve()
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
    symbol = classificator(region, templates)

    results[symbol] = results.get(symbol, 0) + 1

    plt.cla()
    plt.title(f"Class - '{symbol}'")
    plt.imshow(region.image)

    plt.savefig(image_path / f"image_{region.label}.png")

print(results)
print(props[1])

plt.imshow(abinary)
plt.show()
