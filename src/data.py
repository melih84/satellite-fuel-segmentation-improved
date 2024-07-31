from pathlib import Path

import skimage as ski
from skimage.transform import resize
import numpy as np

class Data():
    def __init__(self, image_dir, mask_dir, class_dict):
        self.class_dict = class_dict
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self._get_filenames()
        print(f"Number of images: {len(self.img_idx)}")
        print(f"Number of labels: {len(self.lbl_idx)}")
        print("="*10)   
        
    def _get_filenames(self):
        path_img = Path(self.image_dir)
        path_lbl = Path(self.mask_dir)
        self.img_idx = sorted(list(path_img.glob("*.jpg")))
        self.lbl_idx = sorted(list(path_lbl.glob('*.png')))
     
    def get_class_label(self, label):
        label_dict = {}
        for cls, vals in self.class_dict.items():
            red_c = label[:,:,0] == vals["r"]
            green_c = label[:,:,1] == vals["g"]
            blue_c = label[:,:,2] == vals["b"]
            mask = red_c * green_c * blue_c
            label_dict[cls] = mask.astype("int")
        return label_dict

    def get_batch(self, batch_size, size=None):
        img_batch = self.img_idx[:batch_size]
        lbl_batch = self.lbl_idx[:batch_size]
        X, y = [], []
        for img, lbl in zip(img_batch, lbl_batch):
            image = ski.io.imread(img)
            lbl_rgb = ski.io.imread(lbl)
            lbl_dict = self.get_class_label(label=lbl_rgb)
            if size:
                image, lbl_dict = self.apply_resize(image, lbl_dict, size=size)
            X_arr = image / 255    # (640, 640, 3); normalize by 255
            y_arr = np.array([arr for arr in lbl_dict.values()])  # (CLS, 640, 640)
            y_arr = np.moveaxis(y_arr, 0, -1)   # (640, 640, CLS)

            print(f"Error in Label {lbl.name}") if (y_arr.sum(axis=-1) != 1).all() else None
            
            X.append(X_arr)
            y.append(y_arr)
        X, y = np.array(X), np.array(y)
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print("="*10)
        return X, y     # (B, 640, 640, 3); (B, 640, 640, CLS)

    def apply_resize(self, img, lbl_dict, size):
        small_img = resize(img, size, preserve_range=True, anti_aliasing=True).astype("int")
        small_lbl_dict = {cls: resize(lbl, size, order=0, preserve_range=True) for cls, lbl in lbl_dict.items()}
        return small_img, small_lbl_dict
    

class Processor():
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

    def apply_split(self, division=2):
        #TODO split labels and save images
        for img, lbl in zip(self.img_idx, self.lbl_idx):
            image = ski.io.imread(img)
            print(image.shape)
            H, W, _ = image.shape
            h, w = int(H/division), int(W/division)
            sub_images = [image[i*h:(i+1)*h,
                                j*w:(j+1)*w,
                                :]
                for i in range(division)
                for j in range(division)]
            break
        return sub_images

            