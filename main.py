import json
import random
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from src.networks import unet
from src.data import Data
from src.evaluate import make_predicitons, get_metrics
from src.utils import Visualize

# IMAGE_SIZE = (128, 128)
# IMAGE_SIZE = None

class_info = "data/class_info.csv"
df = pd.read_csv(class_info)
df.index = df["name"]
class_dict= df[["r", "g", "b"]].transpose().to_dict()

class_to_fuel = {i:cls for i, cls in enumerate(class_dict.keys())}
fuel_to_class = {cls:i for i, cls in enumerate(class_dict.keys())}

def main(configs):
    n_epcohs = configs.n_epochs
    batch_size = configs.batch_size
    image_size = configs.image_size
    if image_size is not None:
        image_size = (image_size, image_size) 

    train_root = "data/training_dataset/"
    train_data = Data(image_dir=train_root + "images/",
                    mask_dir=train_root + "masks/",
                    class_dict=class_dict)

    valid_root = "data/verification_dataset/"

    valid_data = Data(image_dir=valid_root + "images/",
                      mask_dir=valid_root + "masks/",
                      class_dict=class_dict)


    X_train, y_train = train_data.get_batch(batch_size=40, size=image_size)
    X_valid, y_valid = valid_data.get_batch(batch_size=10, size=image_size)


    model = unet(input_size=X_train.shape[1:], output_classes=3)
    model.summary()

#TODO add sample images to tensorboard
#TODO delete log dir if exists or incremet the log number
    log_dir = log_dir=root_path+study_id+"logs"
    tb_cb = TensorBoard(log_dir=log_dir,
                        write_graph=True,
                        update_freq=1)
    
    #TODO save best model
    es_cb = EarlyStopping(monitor="val_loss",
                          patience=50)

    hist = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=n_epcohs,
                    validation_data=(X_valid, y_valid),
                    callbacks=[tb_cb, es_cb])

    model.save(root_path+study_id+"model/")

    with open(root_path+study_id+"learning_history.json", "w") as f:
        json.dump(hist.history, f)
    
    # evaluations
    _, y_pred = make_predicitons(model, X_valid)
    metrics = get_metrics(y_valid, y_pred)
    with open(root_path+study_id+"metrics.json", "w") as f:
        json.dump(metrics, f)

    print(pd.DataFrame(metrics))
    #TODO add checkpoints

    n = min(X_valid.shape[0], 10)
    valid_samples = list(range(X_valid.shape[0]))
    random.shuffle(valid_samples)
    save_dir = os.path.join(root_path, study_id, "samples/valid/")
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None
    save_samples(X_valid, y_valid, y_pred, valid_samples[:n], save_dir)
    
    _, y_train_pred = make_predicitons(model, X_train)
    n = min(X_train.shape[0], 10)
    valid_samples = list(range(X_train.shape[0]))
    random.shuffle(valid_samples)
    save_dir = os.path.join(root_path, study_id, "samples/train/")
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None
    save_samples(X_train, y_train, y_train_pred, valid_samples[:n], save_dir)
    

def save_samples(X, y_true, y_pred, sample_list, save_to):
    fuel_color = {
        "SurfaceFuels": "palegreen",
        "Canopy": "darkgreen",
        "Void": "gray"}
    color_scheme = [fuel_color[f] for _, f in class_to_fuel.items()]
    viz = Visualize(color_scheme)
    for m in sample_list:
        viz.display(X[m,:,:,:],
                    y_true[m,:,:,:],
                    y_pred[m,:,:,:],
                    save_to=save_to+f"{m}.png")

    print()
    print(f"Sample images saved to {save_to}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--image_size", default=None, type=int)
    root_path = "experiments/"
    study_id = "study-00/"

    configs = parser.parse_args()
    main(configs)