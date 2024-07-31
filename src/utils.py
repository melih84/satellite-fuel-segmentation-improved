from pathlib import Path
import os

import numpy as np
import matplotlib.pyplot as plt
import skimage as ski


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