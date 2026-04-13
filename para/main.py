import matplotlib.pyplot as plt
import numpy as np
def area(labeled,label):
    return (labeled==label).sum()
def centroid(labeled,label=1):
    y,x=np.where(labeled==label)
    return np.mean(y),np.mean(x)
def neighbours4(y,x):
    return (y,x+1),(y+1,x),(y,x-1),(y-1,x)
def neighboursX(y,x):
    return (y-1,x+1),(y+1,x+1),(y+1,x-1),(y-1,x-1)
def neighbours8(y,x):
    return neighbours4(y,x)+neighboursX(y,x)
def get_bounds(labeled,label,connectivity=neighbours4):
    pos=np.where(labeled==label)
    bounds=[]
    for y,x in zip(*pos):
        for yn,xn in connectivity(y,x):
            if yn<0 or yn>labeled.shape[0]-1:
                bounds.append((y,x))
                break
            elif xn<0 or xn>labeled.shape[1]-1:
                bounds.append((y,x))
                break
            elif labeled[yn,xn]!=label:
                bounds.append((y,x))
                break
    return bounds
def perimetr(labeled,label=1,connectivity=neighbours4):
    return len(get_bounds(labeled,label,connectivity))
def roundness(labeled,label,connectivity=neighbours4):
    A = area(labeled, label)
    P = perimetr(labeled, label, connectivity)
    return (P*P)/A
def distance(px1,px2):
    return ((px1[0]-px2[0])**2+(px1[1]-px2[1])**2)**0.5
def radial_distance(labeled,label,connectivity=neighbours8):
    cy,cx=get_bounds(labeled,label,connectivity)
    rd=0
    for y,x in bounds:
        rd+=distance((cy,cx),(y,x))
    return rd/ len(bounds)
def std_radial(labeled,label,connectivity=neighbours8):
    cy,cx=centroid(labeled,label)
    rd=radial_distance(labeled,label,connectivity)
    bounds= get_bounds(labeled,label,connectivity)
    sr=0
    for y,x in bounds:
        sr+=(distance((cy,cx),(y,x)) - rd)**2
    return (sr / len(bounds))**0.5
def std_circularity(labeled,label,connectivity=neighbours8):
    return (radial_distance(labeled,label,connectivity)/std_radial(labeled,label,connectivity))
    

labeled=np.zeros((16,16),dtype="int")
labeled[4:,:4]=1

labeled[3:10,8:]=2
labeled[[3,4,3],[8,8,9]]=0
labeled[[8,9,9],[8,8,9]]=0
labeled[[3,4,3],[-2,-1,-1]]=0
labeled[[9,8,9],[-2,-1,-1]]=0

labeled[12:-1,6:9]=3
copy=labeled.copy()
for i in range(1,np.max(labeled)+1):
    print(f"Area={area(labeled,i)}")
    print(f"per={perimetr(labeled,i)}")
    print(f"roundness={roundness(labeled,i)}")
    

    
    bounds = get_bounds(labeled, i,neighbours8)
    for y, x in bounds:
        copy[y, x] += 1
plt.imshow(copy)
plt.show()