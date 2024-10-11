import argparse
from pathlib import Path
import time

import yaml
from tensorflow.keras.models import load_model
import cv2

from src.data import LoadImages
from src.utils import probs_to_one_hot, one_hot_to_rgb, increment_path

# TODO HARD CODED (to be fixed by storing class metadata in the saved .keras models) (maybe)
# class names
names = [ "conifer", "background" ]
# class hex color id
color_ids = [ "#FFFFFF", "#000000" ]

def detect():
    path, save_dir = opt.source, opt.save_dir
    
    dataset = LoadImages(path=path)
    for data, name in dataset:
        t1 = time.time()
        print(name, end=" ", flush=True)

        probs = model.predict(data)
        one_hots = probs_to_one_hot(probs)
        pred_masks = one_hot_to_rgb(one_hots, color_ids=color_ids)

        t2 = time.time()
        save_image(pred_masks, dataset.ids, save_dir)
        print(f"Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference")

def save_image(images, ids, save_dir=""):
    for image, id in zip(images, ids):
        path = Path(save_dir) / f"{id}.png"
        cv2.imwrite(path, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--task", default="detect")
    parser.add_argument('--project', default='runs/detect', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    opt = parser.parse_args()
    opt.save_dir = increment_path(Path(opt.project) / opt.name,
                                  exist_ok=opt.exist_ok,
                                  sep="_")  # increment run

    Path(opt.save_dir).mkdir(parents=True, exist_ok=True)
    model = load_model(opt.model)
    detect()