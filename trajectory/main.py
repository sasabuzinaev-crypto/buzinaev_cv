from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

path_of_dir = Path("trajectory/out")
files = sorted(path_of_dir.iterdir(), key=lambda n: int(n.stem.split("_")[1]))

a = []
for file in files:
    img = np.load(file).astype(bool)
    lbl = label(img)
    props = regionprops(lbl)
    if not props:
        continue
    cy, cx = props[0].centroid
    a.append((cx, cy))

x, y = zip(*a)
plt.plot(x, y, "-o", ms=3)
plt.gca().invert_yaxis()
plt.axis("equal")
plt.show()
