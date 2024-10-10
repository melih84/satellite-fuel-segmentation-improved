from pathlib import Path
import os
import random

# from skimage.transform import resize
import numpy as np
from PIL import ImageColor
import cv2

import tensorflow.keras as keras

image_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff']

class Dataset:
    def __init__(self, path, img_size=320, split_ratio=1.0, shuffle=True, sample_size=None,
                 color_ids=["#FFFFFF", "#000000"]):
        
        self.path = path if isinstance(path, Path) else Path(path)
        self.img_size = img_size
        self.shuffle = shuffle
        self.color_ids = color_ids

        img_path = self.path / "images"
        msk_path = self.path / "masks"
        img_filenames = [f for f in sorted(img_path.glob("*.*")) if f.name.split(".")[-1] in image_formats]
        msk_filenames = [f for f in sorted(msk_path.glob("*.*")) if f.name.split(".")[-1] in image_formats]
        
        assert len(img_filenames) > 0, f"No images found in {img_path}"
        assert len(msk_filenames) > 0, f"No masks found in {msk_path}"
        assert len(img_filenames) == len(msk_filenames), f"Images and masks are inconsistent... {len(img_filenames)} images and {len(msk_filenames)} found"

        self.image_files, self.mask_files = img_filenames, msk_filenames

        
        n = int(len(self.image_files))
        self.indices = list(range(n))
        self.index_to_id = {i:f.stem for i, f in enumerate(img_filenames)}
        
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        self.indices = self.indices[:sample_size]
        n = int(len(self.indices))

        train_size = round(split_ratio * n)
        self.train_indices = self.indices[:train_size]
        self.valid_indices = self.indices[train_size:n]
        
        assert len(self.train_indices) > 0, f"Train set is empty."
        if split_ratio < 1.0:
            assert (len(self.valid_indices) > 0), f"Validation set is empty."



class DataGenerator(keras.utils.Sequence):
    def __init__(self, dataset: Dataset, task="train",
                 batch_size=4, shuffle_on_epoch=True, augment_dict=None):
            
            self.dataset = dataset
            self.shuffle = shuffle_on_epoch
            self.augment = augment_dict
            self.task = task

            if task == "train":
                self.indices = self.dataset.train_indices
            if task == "valid":
                self.indices = self.dataset.valid_indices
            if task == "test":
                self.indices = self.dataset.indices

            self.batch_size = min(batch_size, len(self.indices))
            self.ids = [self.dataset.index_to_id[i] for i in self.indices]
            self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(len(self.indices) / self.batch_size)

    def __getitem__(self, index):
        # Geneates one batch of data
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self._generate_data(batch_indices)
        return X, y

    def _generate_data(self, indices):
        X, y = [], []
        for index in indices:
            img_path = self.dataset.image_files[index]
            msk_path = self.dataset.mask_files[index]
            img = cv2.imread(img_path) / 255.
            msk_rgb = cv2.imread(msk_path)
            msk = self.rgb2class(msk_rgb)
            X.append(img), y.append(msk)
        return np.array(X), np.array(y)      

    def on_epoch_end(self):
        # Updates indexes after each epoch
        if self.shuffle and self.task != "test":
            np.random.shuffle(self.indices)

    def rgb2class(self, rgb_mask):
        masks = []
        for hex_id in self.dataset.color_ids:
            r, g, b = ImageColor.getcolor(hex_id, "RGB")
            mask = (rgb_mask[:,:,0] == r) * (rgb_mask[:,:,1] == g) * (rgb_mask[:,:,2] == b)
            masks.append(mask)
        mask = np.array(masks)
        return np.moveaxis(mask, 0, -1)
    
    def random_verical_flip(img, msk, p=.5):
        if random.random < p:
            img = cv2.flip(img, 0)
            msk = cv2.flip(msk, 0)
        return img, msk

class LoadImages(keras.utils.Sequence): # Inference
    def __init__(self, path):
        
        self.path = path if isinstance(path, Path) else Path(path)

        img_path = self.path
        img_filenames = [f for f in sorted(img_path.glob("*.*")) if f.name.split(".")[-1] in image_formats]
        
        assert len(img_filenames) > 0, f"No images found in {img_path}"

        self.image_files = img_filenames

        
        n = int(len(self.image_files))
        self.indices = list(range(n))
        self.index_to_id = {i:f.stem for i, f in enumerate(img_filenames)}
        self.nf = n

        self.ids = [self.index_to_id[i] for i in self.indices]

    def __len__(self):
        return self.nf

    def __getitem__(self, index):
        
        img_path = self.image_files[index]
        img = cv2.imread(img_path) / 255.
        assert img is not None, f"Image not found {img_path}"
        return img.reshape((1,) + img.shape)
        




if __name__ == "__main__":
    path = "./data/winter_conifer_alberta_320x320"
    dataset = Dataset(path)