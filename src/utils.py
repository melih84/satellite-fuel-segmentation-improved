from pathlib import Path
import os
import glob
import re


import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import pandas as pd


COLOR_CODE = {
    "orange": [255, 165, 0],
    "darkgreen": [1, 50, 32],
    "limegreen": [50, 205, 50],
    "springgreen": [0, 255, 127],
    "palegreen": [152, 251, 152],
    "black": [0, 0, 0],
    "gray": [119, 136, 153]
    }

class Visualize():
    def __init__(self, color_scheme):
        self.color_scheme = color_scheme
    
    def combine_class(self, masks):
        comb_mask = np.zeros(masks.shape[:2] + (3,)).astype("uint")
        n_class = masks.shape[-1]
        for cls in range(n_class):
            comb_mask += self._paint(masks[:,:,cls], self.color_scheme[cls])
        return comb_mask
    
    def display(self, image, ground_truth=None, predictions=None, save_to=None):
        n = 1
        n += 1 if ground_truth is not None else 0
        n += 1 if predictions is not None else 0
        fig, ax = plt.subplots(1, n)
        ax[0].imshow(image)
        ax[0].set_title("Input Image")
        i = 1
        if ground_truth is not None:
           actuals = self.combine_class(ground_truth)
           ax[i].imshow(actuals)
           ax[i].set_title("Ground Truth")
           i += 1
        if predictions is not None:   
            preds = self.combine_class(predictions)
            ax[i].imshow(preds)
            ax[i].set_title("Predicted Mask")
            i += 1
        
        [a.axis("off") for a in ax]
        fig.tight_layout()
        if save_to is not None:
            plt.savefig(save_to, dpi=600, bbox_inches="tight")
    
    def create_mask(self, prediction, save_to):
        _, ax = plt.subplots(1,1)
        ax.axis("off")

        preds = self.combine_class(prediction).astype("uint8")
        plt.imsave(save_to, preds)


    @staticmethod
    def _paint(mask, color):
        new_mask = np.zeros(mask.shape + (3,))
        for i in range(3):
            new_mask[:,:,i] = mask * COLOR_CODE[color][i]
        return new_mask.astype("uint")
    
class DataProcessor():
    def __init__(self, image_dir, mask_dir, class_dict):
        self.class_dict = class_dict
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_idx = []
        self.lbl_idx = []
        self._get_filenames()
        print(f"Number of images: {len(self.img_idx)}")
        print(f"Number of labels: {len(self.lbl_idx)}")
        print("="*10)   

    def _get_filenames(self):
        path_img = Path(self.image_dir)
        path_lbl = Path(self.mask_dir)
        self.img_idx = sorted(list(path_img.glob("*.jpg")))
        self.lbl_idx = sorted(list(path_lbl.glob('*.png')))

    def apply_split(self, save_dir: str, division=2):

        #TODO split labels and save images
        num_image = len(self.img_idx)
        curr_num = 1
        sub_images, sub_labels = [], []
        for img_path, lbl_path in zip(self.img_idx, self.lbl_idx):
            image = ski.io.imread(img_path)
            label = ski.io.imread(lbl_path)
            H, W, _ = image.shape
            h, w = int(H/division), int(W/division)
            count = 1
            for i in range(division):
                for j in range(division):
                    sub_images.append((img_path.stem + f"_{count:02}" + img_path.suffix, image[i*h:(i+1)*h, j*w:(j+1)*w, :]))
                    sub_labels.append((lbl_path.stem + f"_{count:02}" + lbl_path.suffix, label[i*h:(i+1)*h, j*w:(j+1)*w, :]))
                    count += 1
            print(f"Split {curr_num} / {num_image}", flush=True, end="\r")
            curr_num += 1
        
        print("")
        if save_dir is not None:
            image_dir = Path(save_dir) / "images"
            label_dir = Path(save_dir) / "masks"
            os.makedirs(image_dir) if not image_dir.exists() else None
            os.makedirs(label_dir) if not label_dir.exists() else None
            save_file = lambda dir, list:  [ski.io.imsave(dir / name, arr) for name, arr in list]
            save_file(image_dir, sub_images)
            save_file(label_dir, sub_labels)

        
        return sub_images, sub_labels
    

def get_class_dict(class_info_path):
    df = pd.read_csv(class_info_path)
    df.index = df["name"]
    class_dict= df[["r", "g", "b"]].transpose().to_dict()
    class_to_fuel = {i:cls for i, cls in enumerate(class_dict.keys())}
    fuel_to_class = {cls:i for i, cls in enumerate(class_dict.keys())}
    return (class_dict, class_to_fuel, fuel_to_class)


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path

def probs_to_one_hot(probs):
    y_pred = np.argmax(probs, axis=-1)
    n_class = probs.shape[-1]
    y_pred_one_hot = np.array([(y_pred == i).astype("int") for i in range(n_class)])
    y_pred_one_hot = np.moveaxis(y_pred_one_hot, 0, -1) 
    return y_pred_one_hot