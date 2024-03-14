import numpy as np

def make_predicitons(model, X):
    probs = model.predict(X)
    y_pred = np.argmax(probs, axis=-1)
    n_class = probs.shape[-1]
    y_pred_one_hot = np.array([(y_pred == i).astype("int") for i in range(n_class)])
    y_pred_one_hot = np.moveaxis(y_pred_one_hot, 0, -1) 
    return probs, y_pred_one_hot