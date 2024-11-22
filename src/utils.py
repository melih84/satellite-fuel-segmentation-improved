from pathlib import Path
import os
import glob
import re


import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import pandas as pd
from PIL import ImageColor
import cv2


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
            print(f"Split {curr_num} / {num_image}", flush=True, end="/r")
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

def one_hot_to_rgb(one_hots, color_ids):
    nc = one_hots.shape[-1]
    masks_list = []
    for mask in one_hots:
        mask_rgb = np.zeros(mask.shape[:2] + (3,))
        for cls in range(nc):
            for ch, val in enumerate(ImageColor.getcolor(color_ids[cls], "RGB")):
                mask_rgb[...,ch] += mask[...,cls] * val
        masks_list.append(mask_rgb)
    return masks_list


# PLOTS
def overlay_mask(img, msk):
    return cv2.addWeighted(img, 1, msk, .5, 0)

def segmentation_to_contour(img, msk):
    # msk: a flat gray-scale uint8 array
    RGBforLabel = {1:(0,255,0), 2:(0,255,0)}    # BUG the color changes from RGB to BRG
    msk = msk.astype(np.uint8)
    if np.all((img >= 0) & (img <= 1)):
        img = (img * 255).astype(np.uint8)
    contours, _ = cv2.findContours(msk, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        # Find mean colour inside this contour by doing a masked mean
        mask = np.zeros(msk.shape, np.uint8)
        cv2.drawContours(mask, [c],-1, 255, -1)
        mean, _, _, _ = cv2.mean(msk, mask=mask)
        # Get appropriate colour for this label
        label = 2 if mean > 1.0 else 1
        colour = RGBforLabel.get(label)
        cv2.drawContours(img, [c], -1, colour, 2)
    return img

def resize_to_nearest_multiple(image, num):
    # Resizes the dimensions of the image to a number closest do 'num'
     # Get the current dimensions
    height, width = image.shape[1:3]
    
    # Calculate the nearest multiples of"num
    new_height = round(height / num) * num
    new_width = round(width / num) * num
 
    # Resize the image
    resized_image = cv2.resize(image[0], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image


if __name__ == "__main__":
    image_dir = Path("inference")
    mask_dir = Path("runs/detect/exp_23/")
    overlay_dir = mask_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    images = list(image_dir.glob("*.png"))
    # masks = list(mask_dir.glob("*.png"))
    for image in images:
        name = image.stem
        img = cv2.imread(image)
        msk = cv2.imread(mask_dir / f"{name}.png", cv2.IMREAD_GRAYSCALE)
        img = segmentation_to_contour(img, msk)
        cv2.imwrite(overlay_dir/ f"{name}.png", img)
