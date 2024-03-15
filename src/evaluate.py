import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

def make_predicitons(model, X):
    probs = model.predict(X)
    y_pred = np.argmax(probs, axis=-1)
    n_class = probs.shape[-1]
    y_pred_one_hot = np.array([(y_pred == i).astype("int") for i in range(n_class)])
    y_pred_one_hot = np.moveaxis(y_pred_one_hot, 0, -1) 
    return probs, y_pred_one_hot

def get_metrics(ground_truth, prediction):
    n_class = ground_truth.shape[-1]
    pr, rc, f1, iou, px = [], [], [], [], []
    for i in range(n_class):
        y_true = ground_truth[:,:,:,i].flatten()
        y_pred = prediction[:,:,:,i].flatten()
        pr.append(precision_score(y_true, y_pred))
        rc.append(recall_score(y_true, y_pred))
        f1.append(f1_score(y_true, y_pred))
        iou.append(jaccard_score(y_true, y_pred))
        px.append(y_true.sum())
    return pr, rc, f1, iou, px