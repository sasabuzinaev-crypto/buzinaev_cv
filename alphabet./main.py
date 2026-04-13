import argparse
import json
import os
from pathlib import Path

import numpy as np
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.transform import resize

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.getenv("TMPDIR", "/tmp")) / "matplotlib"))

import matplotlib.pyplot as plt  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent

SYMBOLS_ORDER = ["A", "B", "8", "0", "1", "W", "X", "*", "-", "/", "P", "D"]
DEFAULT_TEMPLATE_NAME = "alphabet_ext.png"
DEFAULT_INPUT_NAME = "symbols.png"
TEMPLATE_VECTOR_SIZE = 64
INPUT_THRESHOLD_DEFAULT = 10.0


def first_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] >= 3:
        image = image[:, :, :3]
        return image.mean(axis=2)
    return image.astype(np.float32)


def binarize_template(gray: np.ndarray) -> np.ndarray:
    # alphabet_ext.png: blue symbols on white background (symbol pixels are darker)
    return gray < 200


def binarize_input(gray: np.ndarray, threshold: float = 10.0) -> np.ndarray:
    # symbols.png: colored symbols on black background
    return gray > threshold


def pad_to_square(image: np.ndarray, value: float = 0.0) -> np.ndarray:
    h, w = image.shape[:2]
    side = int(max(h, w))
    pad_top = (side - h) // 2
    pad_bottom = side - h - pad_top
    pad_left = (side - w) // 2
    pad_right = side - w - pad_left
    return np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=value)


def vectorize_symbol(mask: np.ndarray, size: int = TEMPLATE_VECTOR_SIZE) -> np.ndarray:
    square = pad_to_square(mask.astype(np.float32), value=0.0)
    resized = resize(square, (size, size), order=1, mode="constant", anti_aliasing=False, preserve_range=True)
    vec = (resized > 0.5).astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def extract_templates(template_path: Path) -> dict[str, np.ndarray]:
    raw = imread(str(template_path))
    gray = to_gray(raw)
    binary = binarize_template(gray)

    labeled = label(binary)
    regions = regionprops(labeled)
    regions = sorted(regions, key=lambda r: r.bbox[1])  # left-to-right in one row

    if len(regions) != len(SYMBOLS_ORDER):
        raise RuntimeError(
            f"Unexpected templates count: got {len(regions)}, expected {len(SYMBOLS_ORDER)} from {template_path}"
        )

    templates: dict[str, np.ndarray] = {}
    for symbol, region in zip(SYMBOLS_ORDER, regions, strict=True):
        templates[symbol] = vectorize_symbol(region.image)
    return templates


def count_holes(region) -> int:
    # same idea as vector_recognize: number of background components inside the glyph
    shape = region.image.shape
    padded = np.zeros((shape[0] + 2, shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = region.image
    inverted = np.logical_not(padded)
    labeled = label(inverted)
    return int(np.max(labeled) - 1)


def template_best(templates: dict[str, np.ndarray], region_image: np.ndarray, candidates: list[str] | None = None) -> str:
    vec = vectorize_symbol(region_image)
    if candidates is None:
        candidates = list(templates.keys())
    matrix = np.stack([templates[s] for s in candidates], axis=0)
    scores = matrix @ vec
    return candidates[int(np.argmax(scores))]


def classificator(region, templates: dict[str, np.ndarray]) -> str:
    holes = count_holes(region)

    if holes == 2:  # B,8
        # vector_recognize uses vertical-line heuristic; here template match is more stable on noisy crops
        return template_best(templates, region.image, candidates=["B", "8"])

    if holes == 1:  # A,0,P,D (use templates to disambiguate)
        return template_best(templates, region.image, candidates=["A", "0", "P", "D"])

    # holes == 0: 1,W,X,*,-,/
    shape = region.image.shape
    aspect = shape[1] / shape[0]
    if aspect > 2.5:
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
    # fallback to full-template match (keeps vector_recognize-style logic but removes '?' for this dataset)
    return template_best(templates, region.image)


def recognize(
    input_path: Path,
    templates: dict[str, np.ndarray],
    threshold: float = INPUT_THRESHOLD_DEFAULT,
    out_dir: Path | None = None,
    save_images: bool = True,
) -> tuple[dict[str, int], int]:
    raw = imread(str(input_path))
    gray = to_gray(raw)
    binary = binarize_input(gray, threshold=threshold)

    labeled = label(binary)
    regions = regionprops(labeled)

    counts: dict[str, int] = {}

    if save_images:
        if out_dir is None:
            out_dir = BASE_DIR / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(5, 7))

    for region in regions:
        symbol = classificator(region, templates)
        counts[symbol] = counts.get(symbol, 0) + 1

        if save_images and out_dir is not None:
            plt.cla()
            plt.title(f"Class - '{symbol}'")
            plt.imshow(region.image)
            plt.axis("off")
            plt.savefig(out_dir / f"image_{region.label}.png", bbox_inches="tight", pad_inches=0.02)

    counts = {k: v for k, v in counts.items() if v > 0}
    return counts, len(regions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Symbol recognition (vector_recognize-style) for symbols.png.")
    parser.add_argument("--templates", type=Path, default=None, help="Path to alphabet_ext.png")
    parser.add_argument("--input", type=Path, default=None, help="Path to symbols.png")
    parser.add_argument("--json", action="store_true", help="Print result as JSON")
    parser.add_argument("--threshold", type=float, default=INPUT_THRESHOLD_DEFAULT, help="Binarization threshold")
    parser.add_argument("--out", type=Path, default=None, help="Output dir for extracted symbol images")
    parser.add_argument("--no-save", action="store_true", help="Do not save symbol crops")
    parser.add_argument("--show", action="store_true", help="Show binary image (matplotlib)")
    args = parser.parse_args()

    template_path = first_existing_path(
        (args.templates if args.templates is not None else BASE_DIR / DEFAULT_TEMPLATE_NAME),
        BASE_DIR / DEFAULT_TEMPLATE_NAME,
    )
    input_path = first_existing_path(
        (args.input if args.input is not None else BASE_DIR / DEFAULT_INPUT_NAME),
        BASE_DIR / DEFAULT_INPUT_NAME,
    )

    templates = extract_templates(template_path)
    result, total = recognize(
        input_path,
        templates,
        threshold=args.threshold,
        out_dir=args.out,
        save_images=(not args.no_save),
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        print(result)
    print(total)

    if args.show:
        raw = imread(str(input_path))
        gray = to_gray(raw)
        binary = binarize_input(gray, threshold=args.threshold)
        plt.figure(figsize=(8, 8))
        plt.imshow(binary, cmap="gray")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
