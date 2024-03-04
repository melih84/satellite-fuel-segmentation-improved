import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.networks import unet
from src.data import Data

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


    X_train, y_train = train_data.get_batch(batch_size=32, size=IMAGE_SIZE)
    X_valid, y_valid = valid_data.get_batch(batch_size=4, size=IMAGE_SIZE)

    model = unet(input_size=X_train.shape[1,:], output_classes=3)
    model.summary()
    hist = model.fit(X_train, y_train,
                    batch_size=2,
                    epochs=20,
                    validation_data=(X_valid, y_valid))

    model.save(study_id+"model/")
    # probs = model.predict(X_valid)

    # y_pred = np.argmax(probs, axis=-1)

#TODO add tensorboard with sample images
#TODO calcualte ans store recall, precision, F1-score, IoU per class and the confusion matrix

if __name__ == "__main__":
    study_id = "experiments/study-00/"
    main()