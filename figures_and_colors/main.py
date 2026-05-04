import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.color import rgb2hsv

def find_clusters(sorted_vals: np.array):
    diffs = np.diff(sorted_vals)
    threshold_val = np.std(diffs) * 2
    boundary_idxs = np.where(diffs > threshold_val)
    return boundary_idxs[0] + 1

def determine_hue(obj_region, orig_img):
    center_y, center_x = obj_region.centroid
    hue_val = rgb2hsv(orig_img[int(center_y), int(center_x)])[0]
    
    hue_limits = {
        "red": 0.19202898,
        "orange": 0.30476192,
        "yellow": 0.41509435,
        "green": 0.60897434,
        "blue": 0.8333333
    }
    
    if 0.0 <= hue_val < hue_limits['red']:
        return "red"
    elif hue_val < hue_limits['orange']:
        return "orange"
    elif hue_val < hue_limits['yellow']:
        return "yellow"
    elif hue_val < hue_limits['green']:
        return "green"
    elif hue_val < hue_limits['blue']:
        return "blue"
    else:
        return "violet"
    
input_img = plt.imread("figures_and_colors/balls_and_rects.png")
gray_img = input_img.mean(axis=2)
binary_mask = gray_img > 0
labeled_img = label(binary_mask)
all_regions = regionprops(labeled_img)

circles_dict = {}
rects_dict = {}

for region in all_regions:
    color_name = determine_hue(region, input_img)
    if region.eccentricity == 0:
        if color_name not in circles_dict:
            circles_dict[color_name] = 0
        circles_dict[color_name] += 1
    else:
        if color_name not in rects_dict:
            rects_dict[color_name] = 0
        rects_dict[color_name] += 1

total_count = sum(circles_dict.values()) + sum(rects_dict.values())
print("Total objects detected:", total_count)
print("Circles:", circles_dict)
print("Rectangles:", rects_dict)