import json
import random
import os
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

from src.networks import unet
from src.data import Data
from src.evaluate import make_predicitons, get_metrics
from src.utils import Visualize

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
    run_dir = configs.run_dir
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
    # model.summary()

    #TODO add sample images to tensorboard
    log_dir = log_dir=run_dir / "logs"
    tb_cb = TensorBoard(log_dir=log_dir,
                        write_graph=True,
                        update_freq=1)
    
    es_cb = EarlyStopping(monitor="val_loss",
                          patience=50)
    
    checkpoint_dir = run_dir / "checkpoints"
    mod_cb = ModelCheckpoint(filepath=str(checkpoint_dir) + "/checkpoint.model-{epoch:02d}-{val_loss:.2f}.keras",
                            monitor="val_loss",
                            mode="min",
                            save_best_only=True)

    hist = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=n_epcohs,
                    validation_data=(X_valid, y_valid),
                    callbacks=[tb_cb, es_cb, mod_cb])

    model.save(run_dir / "model")

    with open(run_dir / "learning_history.json", "w") as f:
        json.dump(hist.history, f)
    
    # evaluations
    _, y_pred = make_predicitons(model, X_valid)
    metrics = get_metrics(y_valid, y_pred, class_to_fuel)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)

    print(pd.DataFrame(metrics))
    #TODO add checkpoints

    n = min(X_valid.shape[0], 10)
    valid_samples = list(range(X_valid.shape[0]))
    random.shuffle(valid_samples)
    sample_dir = run_dir / "samples"
    save_dir = sample_dir / "validation"

    os.makedirs(save_dir) if not os.path.exists(save_dir) else None
    save_samples(X_valid, y_valid, y_pred, valid_samples[:n], save_dir)
    
    _, y_train_pred = make_predicitons(model, X_train)
    n = min(X_train.shape[0], 10)
    valid_samples = list(range(X_train.shape[0]))
    random.shuffle(valid_samples)
    save_dir = sample_dir / "train"
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
                    save_to=save_to / f"{m}.png")

    print()
    print(f"Sample images saved to {save_to}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--image_size", default=None, type=int)
    
    root_path = "experiments/"
    study_id = "study-00"
    study_dir = Path("./experiments") / study_id

    if not study_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run-')[1]) for folder in
                         study_dir.iterdir() if
                         str(folder.name).startswith('run-')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    
    run_dir = study_dir / f"run-{run_num:02}"
    os.makedirs(run_dir)
    configs = parser.parse_args()
    configs.run_dir = run_dir

    print("*"*20)
    print(f"STUDY-ID: {study_id} / RUN: {run_num}")
    print("*"*20)

    main(configs)