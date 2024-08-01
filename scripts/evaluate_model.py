from pathlib import Path
import os

from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.evaluate import make_predicitons, get_metrics
from src.data import Data
from src.utils import get_class_dict, Visualize

# Load saved model
path_to_model = "./experiments/intermediate/run-02/model.keras"
model = load_model(path_to_model)

# Load verificaiton data
image_dir = "./data/verification_dataset_320x320/images"
mask_dir = "./data/verification_dataset_320x320/masks"
class_info= "./data/class_info.csv"

class_dict, class_to_fuel, _ = get_class_dict(class_info)

data = Data(image_dir, mask_dir, class_dict)
X, y, sample_idx = data.get_batch()

# Perform segmentation
probs, y_pred = make_predicitons(model, X, batch_size=2)

# Calculate accuracy metrics
metrics = get_metrics(y, y_pred, class_to_fuel)
print("="*100)
print(pd.DataFrame(metrics))
print("="*100)

# Make save directory
save_dir = Path(path_to_model).parent / "verification_results"
os.mkdir(save_dir) if not save_dir.exists() else None

# Save segmentation
fuel_color = {
    "SurfaceFuels": "palegreen",
    "Canopy": "darkgreen",
    "Void": "gray"}
color_scheme = [fuel_color[f] for _, f in class_to_fuel.items()]

vis = Visualize(color_scheme)
for m in range(y_pred.shape[0]):
    vis.create_mask(y_pred[m,...], save_dir / f"{sample_idx[m]}.png")

'''

# Reconstruct main satellite image
def find_corners(sample_idx):
    corners = []
    for sample in sample_idx:
        id, sub_id = [int(i) for i in sample.split("_")]
        corner_y = (id - 1) % 14 * 640 + (sub_id - 1) // 2 * 320
        corner_x = (id - 1) // 14 * 640 + (sub_id - 1 ) % 2 * 320
        corners.append((corner_x, corner_y))
    return corners

def stich(arrs, corners, save_to):
    # main_image = np.zeros((28*640, 30*640, 3))
    main_image = np.empty((14*640, 30*640, 3))
    print("Begin stiching images...")
    for m, cor in enumerate(corners):
        print(m)
        main_image[cor[1]:cor[1]+320, cor[0]:cor[0]+320, :] = arrs[m,...]
        # if m % 10 == 0:
        #     plt.imsave(save_to, main_image)
    plt.imsave(save_to, main_image)
    
    # return main_image


corners = find_corners(sample_idx)
stich(X, corners, "mask.png")
# plt.imsave("test.png", main_image)
'''