import numpy as np
import matplotlib.pyplot as plt
from skimage import draw

def hist(gray):
    H= np.zeros(256, dtype="int")
    for y in range(gray.shape[0]):
        for x in range(gray.shape[1]):
            val = gray[y,x]
            H[val] += 1
    return H
image=np.zeros((1000,1000),dtype="uint8")
image= np.random.randint(10,75,image.shape)
ys,xs =draw.disk((500,500),220)
image[ys,xs]=np.random.randint(100,110,len(ys))

ys,xs =draw.disk((800,800),200)
image[ys,xs]=np.random.randint(80,100,len(ys))
treshold=77
binary = image >treshold
plt.subplot(131)
plt.imshow(image,cmap="gray")
plt.subplot(132)
plt.imshow(hist(image))
plt.subplot(133)
plt.imshow(hist(binary))
plt.show()