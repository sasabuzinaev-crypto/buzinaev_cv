import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import opening
from skimage.measure import label
image = np.load("/Users/sashabuzinaevicloud.com/Downloads/wires/wires5.npy")
struct = np.ones((3,1))
process = opening(image, struct)
labeled_image=label(image)
labeled_process=label(process)
print(f"Original{np.max(labeled_image)}")
print(f"Processed{np.max(labeled_process)}")
plt.subplot(121)
list=[]
for i in range(1,np.max(labeled_image)+1):
    wire=labeled_image==i
    wire_opened = opening(wire, struct)
    labeled_wire = label(wire_opened)
    pieces = np.max(labeled_wire)
    if pieces == 1:
        status = "целый"
    else:
        status = f"порван на {pieces} части"
    print(status)
plt.imshow(labeled_image)
plt.subplot(122)
plt.imshow(opening(image, struct))
plt.show()