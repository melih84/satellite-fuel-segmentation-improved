import pandas as pd

from src.utils import DataProcessor

image_dir = "/home/mazjam/projects/def-mgul/mazjam/satellite-fuel-segmentation/data/training_dataset/images"
mask_dir = "/home/mazjam/projects/def-mgul/mazjam/satellite-fuel-segmentation/data/training_dataset/masks"

class_info = "/home/mazjam/projects/def-mgul/mazjam/satellite-fuel-segmentation/data/class_info.csv"
df = pd.read_csv(class_info)
df.index = df["name"]
class_dict= df[["r", "g", "b"]].transpose().to_dict()

processor = DataProcessor(image_dir, mask_dir, class_dict)

new_dataset_dir = "/home/mazjam/projects/def-mgul/mazjam/satellite-fuel-segmentation/data/training_dataset_320x320"

processor.apply_split(new_dataset_dir, 2)