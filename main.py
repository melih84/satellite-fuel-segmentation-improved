from pathlib import Path

import skimage as ski 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.networks import unet

data_root = "data/training_dataset/"
image_dir = data_root + "images/"
mask_dir = data_root + "masks/"

class_mapping = data_root + "class_mapping.csv"

df = pd.read_csv(class_mapping)
df.index = df["name"]
class_dict= df[["r", "g", "b"]].transpose().to_dict()

path_img = Path(image_dir)
path_lbl = Path(mask_dir)
img_idx = list(path_img.glob("*.jpg"))
lbl_idx = list(path_lbl.glob('*.png'))
print(f"Number of images: {len(img_idx)}")
print(f"Number of labels: {len(lbl_idx)}")

def get_class_label(label):
    label_dict = {}
    for cls, vals in class_dict.items():
        red_c = label[:,:,0] == vals["r"]
        green_c = label[:,:,1] == vals["g"]
        blue_c = label[:,:,2] == vals["b"]
        mask = red_c * green_c * blue_c
        label_dict[cls] = mask.astype("int")
    return label_dict

def get_batch(img_idx, lbl_idx, batch_size):
    img_batch = img_idx[:batch_size]
    lbl_batch = lbl_idx[:batch_size]
    X, y = [], []
    for img, lbl in zip(img_batch, lbl_batch):
        X_arr = ski.io.imread(img) / 255    # (640, 640, 3); normalize by 255
        lbl_rgb = ski.io.imread(lbl)
        lbl_dict = get_class_label(label=lbl_rgb)
        y_arr = np.array([arr for arr in lbl_dict.values()])  # (CLS, 640, 640)
        y_arr = np.moveaxis(y_arr, 0, -1)   # (640, 640, CLS)
        X.append(X_arr)
        y.append(y_arr)
           
    return np.array(X), np.array(y)     # (B, 640, 640, 3); (B, 640, 640, CLS)

X, y = get_batch(img_idx, lbl_idx, 8)

model = unet(input_size=(640,640,3), output_classes=3)
model.summary()
# hist = model.fit(X, y, batch_size=2, epochs=10)