from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

from src.utils import get_class_dict
from src.data import Data

# # Load verificaiton data
# image_dir = "./data/verification_dataset_320x320/images"
# mask_dir = "./data/verification_dataset_320x320/masks"
# class_info= "./data/class_info.csv"

# class_dict, class_to_fuel, _ = get_class_dict(class_info)

# data = Data(image_dir, mask_dir, class_dict)

def find_corners(sample_idx):
    corners = []
    for sample in sample_idx:
        id, sub_id = [int(i) for i in sample.split("_")]
        corner_y = (id - 1) % 14 * 640 + (sub_id - 1) // 2 * 320
        corner_x = (id - 1) // 14 * 640 + (sub_id - 1 ) % 2 * 320
        corners.append((corner_x, corner_y))
    return corners

def stich(arrs, corners, save_to):
    main_image = np.empty((14*640, 30*640, 3), dtype="uint8")
    print("Begin stiching images...")
    for m, cor in enumerate(corners):
        print(m)
        main_image[cor[1]:cor[1]+320, cor[0]:cor[0]+320, :] = arrs[m,...]
    
    plt.imsave(save_to, main_image)
    
# Load predicted segmentaiton tiles
mask_dir = Path("./experiments/intermediate/run-02/verification_results") 
samples = list(mask_dir.glob("*.png"))
samples_idx = [s.stem for s in samples]

data = []
for sample in samples:
    image = ski.io.imread(sample)
    data.append(image.astype("uint8")[...,:-1])
data = np.array(data)

# Find tile correspondance 
corners = find_corners(samples_idx)

# Reconstruct mask
stich(data, corners, "mask.png")
