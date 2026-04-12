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

def count_holes(region):
    shape=region.image.shape 
    new_image=np.zeros((shape[0]+2,shape[1]+2))
    new_image[1:-1,1:-1]=region.image
    new_image=np.logical_not(new_image)
    labeled=label(new_image)
    return np.max(labeled)-1

def classificator(region):
    holes=count_holes(region)
    if holes ==2: #B,8
        vlines=(np.sum(region.image,0)==region.image.shape[0]).sum()
        vlines= vlines/region.image.shape[1]
        if vlines>0.2:
            return "B"
        else:
            return "8"
    elif holes==1: #A,O
        if region.eccentricity>0.58:
            return "O"
        else:
            return "A"
    else: #1,W,X,*,-,/
        if region.image.sum()/region.image.size==1.0:
            return "-"
        shape=region.image.shape
        aspect=np.min(shape)/np.max(shape)
        if aspect>0.9:
            return "*"
        vlines=(np.sum(region.image,0)==region.image.shape[0]).sum()
        hlines=(np.sum(region.image,1)==region.image.shape[1]).sum()
        if vlines>0 and hlines>0:
            return "1"
        labeled=label(np.logical_not(region.image))
        bays=0
        for r in regionprops(labeled):
            if r.area>3:
                bays+=1
        if bays==2:
            return "/"
        elif bays==4:
            return "X"
        elif bays==5:
            return "W"

save_path = BASE_DIR

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
abinary = image.mean(2)>0
alabeled = label(abinary)
print(np.max(alabeled))
aprops = regionprops(alabeled)

result = {}
image_path = save_path / "out_tree"
image_path.mkdir(exist_ok=True)
plt.figure(figsize=(5,7))
for region in aprops:
    symbol = classificator(region)
    if symbol not in result:
        result[symbol] = 0
    result[symbol] += 1
    plt.cla()
    plt.title(f"Class - '{symbol}'")
    plt.imshow(region.image)
    plt.savefig(image_path / f"image_{region.label}.png")
print(result)
print(f"{1.0-result.get('?',0)/len(aprops)}")
