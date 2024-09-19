from pathlib import Path
import os

import skimage as ski
from skimage.transform import resize
import numpy as np
from PIL import ImageColor
import cv2

image_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff']

class Data():
    def __init__(self, image_dir, mask_dir, class_dict):
        self.class_dict = class_dict
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
        self._get_filenames()
        self.num_samples = len(self.img_idx)

        print(f"Number of images: {len(self.img_idx)}")
        print(f"Number of labels: {len(self.lbl_idx)}")
        print("="*10)   
        
    def _get_filenames(self):
        path_img = Path(self.image_dir)
        path_lbl = Path(self.mask_dir)
        self.img_idx = sorted(list(path_img.glob("*.jpg")))
        self.lbl_idx = sorted(list(path_lbl.glob('*.png')))
        self.sample_idx = [p.stem for p in self.img_idx]
     
    def get_class_label(self, label):
        label_dict = {}
        for cls, vals in self.class_dict.items():
            red_c = label[:,:,0] == vals["r"]
            green_c = label[:,:,1] == vals["g"]
            blue_c = label[:,:,2] == vals["b"]
            mask = red_c * green_c * blue_c
            label_dict[cls] = mask.astype("int")
        return label_dict

    def get_batch(self, batch_size=None, size=None):
        if batch_size is None:
            batch_size = self.num_samples
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
        return X, y, self.sample_idx[:batch_size]

    def apply_resize(self, img, lbl_dict, size):
        small_img = resize(img, size, preserve_range=True, anti_aliasing=True).astype("int")
        small_lbl_dict = {cls: resize(lbl, size, order=0, preserve_range=True) for cls, lbl in lbl_dict.items()}
        return small_img, small_lbl_dict
    



class LoadImagesAndMasks():
    def __init__(self, path, img_size=320, batch_size=4, augment=False,
                 color_ids=["#FFFFFF", "#000000"]):
        
        self.path = path if isinstance(path, Path) else Path(path)
        self.img_size = img_size
        self.augment = augment
        self.color_ids = color_ids

        img_path = self.path / "images"
        msk_path = self.path / "masks"
        img_filenames = [f for f in sorted(img_path.glob("*.*")) if f.name.split(".")[-1] in image_formats]
        msk_filenames = [f for f in sorted(msk_path.glob("*.*")) if f.name.split(".")[-1] in image_formats]
        
        assert len(img_filenames) == len(msk_filenames), f"Images and masks are inconsistent... {len(img_filenames)} images and {len(msk_filenames)} found"

        self.image_files, self.mask_files = img_filenames, msk_filenames
        n = len(self.image_files)
        self.indices = range(n)

    def __getitem__(self, index):
        index = self.indices[index]
        img_path = self.image_files[index]
        msk_path = self.mask_files[index]
        img = cv2.imread(img_path)
        msk_rgb = cv2.imread(msk_path)
        msk = self.rgb2class(msk_rgb)
        return img, msk_rgb, msk


    def rgb2class(self, rgb_mask):
        masks = []
        for hex_id in self.color_ids:
            r, g, b = ImageColor.getcolor(hex_id, "RGB")
            mask = (rgb_mask[:,:,0] == r) * (rgb_mask[:,:,1] == g) * (rgb_mask[:,:,2] == b)
            masks.append(mask)
        mask = np.array(masks)
        return np.moveaxis(mask, 0, -1)



        # msk_filenames, sample_ids = [f, f.stem for f in msk_path.glob("*.*")]
        # for f in img_filenames:
        #     if f.stem in sample_ids:
        #         continue
        #     else

        # label_filenames =list(msk_path.glob("*"))
        # assert filenames, f"No images found in ./{self.path}/images"
        # for f in filenames:
        #     if f.stem
        breakpoint()

        # path_img = Path(self.image_dir)
        # path_lbl = Path(self.mask_dir)
        # self.img_idx = sorted(list(path_img.glob("*.jpg")))
        # self.lbl_idx = sorted(list(path_lbl.glob('*.png')))
        # self.sample_idx = [p.stem for p in self.img_idx]      

if __name__ == "__main__":
    path = "./data/winter_conifer_alberta_320x320"
    loader = LoadImagesAndMasks(path)
    breakpoint()