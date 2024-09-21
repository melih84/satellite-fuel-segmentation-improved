import argparse
from pathlib import Path

import yaml
from tensorflow.keras.models import load_model
import cv2

from src.data import Dataset, DataGenerator
from src.utils import probs_to_one_hot, one_hot_to_rgb, increment_path


def test(data_dict,
         img_size,
         batch_size,
         model,
         task,
         save_dir):
    
    path = data_dict[task]
    dataset = Dataset(path=path,
                      img_size=img_size,
                      shuffle=False)
    
    test_data = DataGenerator(dataset=dataset,
                              task="test",
                              batch_size=batch_size)
    
    probs = model.predict(test_data)
    one_hots = probs_to_one_hot(probs)
    pred_masks = one_hot_to_rgb(one_hots, color_ids=data_dict["color_ids"])
    save_image(pred_masks, test_data.ids, save_dir)

def save_image(images, ids, save_dir=""):
    for image, id in zip(images, ids):
        path = Path(save_dir) / f"{id}.png"
        cv2.imwrite(path, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--task", default="test")
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    opt = parser.parse_args()
    opt.save_dir = increment_path(Path(opt.project) / opt.name,
                                  exist_ok=opt.exist_ok,
                                  sep="_")  # increment run

    Path(opt.save_dir).mkdir(parents=True, exist_ok=True)
    model = load_model(opt.model)
    with open(opt.data, "r") as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)

    test(data_dict=data_dict,
         img_size=opt.image_size,
         batch_size=opt.batch_size,
         model=model,
         task=opt.task,
         save_dir=opt.save_dir)