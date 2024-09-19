import argparse

import yaml
from tensorflow.keras.models import load_model

from src.data import Dataset, DataGenerator
from src.utils import probs_to_one_hot


def test(data_dict,
         img_size,
         batch_size,
         model,
         task):
    
    path = data_dict[task]
    dataset = Dataset(path=path,
                      img_size=img_size,
                      shuffle=False)
    test_data = DataGenerator(dataset=dataset,
                              batch_size=batch_size)
    
    probs = model.predict(test_data)
    outputs = probs_to_one_hot(probs)
    breakpoint()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--task", default="test")

    opt = parser.parse_args()
    print(opt)

    model = load_model(opt.model)
    with open(opt.data, "r") as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)

    test(data_dict=data_dict,
         img_size=opt.image_size,
         batch_size=opt.batch_size,
         model=model,
         task=opt.task)