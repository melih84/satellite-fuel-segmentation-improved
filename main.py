import json

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from src.networks import unet
from src.data import Data
from src.evaluate import make_predicitons, get_metrics
from src.utils import Visualize

IMAGE_SIZE = (128, 128)

class_info = "data/class_info.csv"

def main():
    df = pd.read_csv(class_info)
    df.index = df["name"]
    class_dict= df[["r", "g", "b"]].transpose().to_dict()

    train_root = "data/training_dataset/"

    train_data = Data(image_dir=train_root + "images/",
                    mask_dir=train_root + "masks/",
                    class_dict=class_dict)

    valid_root = "data/verification_dataset/"

    valid_data = Data(image_dir=valid_root + "images/",
                      mask_dir=valid_root + "masks/",
                      class_dict=class_dict)


    X_train, y_train = train_data.get_batch(batch_size=4, size=IMAGE_SIZE)
    X_valid, y_valid = valid_data.get_batch(batch_size=2, size=IMAGE_SIZE)


    model = unet(input_size=X_train.shape[1:], output_classes=3)
    model.summary()

#TODO add sample images to tensorboard
    tb_cb = TensorBoard(log_dir=root_path+study_id+"logs",
                        write_graph=True,

                        update_freq=1)
    es_cb = EarlyStopping(monitor="val_loss",
                          patience=10)

    hist = model.fit(X_train, y_train,
                    batch_size=2,
                    epochs=3,
                    validation_data=(X_valid, y_valid),
                    callbacks=[tb_cb, es_cb])

    # model.save(study_id+"model/")
    with open(root_path+study_id+"learning_history.json", "w") as f:
        json.dump(hist.history, f)
    
    # evaluations
    probs, y_pred = make_predicitons(model, X_valid)
    metrics = get_metrics(y_valid, y_pred)
    with open(root_path+study_id+"metrics.json", "w") as f:
        json.dump(metrics, f)

    print(pd.DataFrame(metrics))
    # probs = model.predict(X_valid)

    # y_pred = np.argmax(probs, axis=-1)


if __name__ == "__main__":
    root_path = "experiments/"
    study_id = "study-00/"
    main()