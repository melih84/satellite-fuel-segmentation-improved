import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

def make_predicitons(model, X, batch_size=None):
    probs = model.predict(X, batch_size=batch_size)
    y_pred = np.argmax(probs, axis=-1)
    n_class = probs.shape[-1]
    y_pred_one_hot = np.array([(y_pred == i).astype("int") for i in range(n_class)])
    y_pred_one_hot = np.moveaxis(y_pred_one_hot, 0, -1) 
    return probs, y_pred_one_hot

def get_metrics(ground_truth, prediction, class_to_fuel=None):
    n_class = ground_truth.shape[-1]
    pr, rc, f1, iou, ratio = {}, {}, {}, {}, {}
    for i in range(n_class):
        cls_id = (i if class_to_fuel is None else class_to_fuel[i])
        y_true = ground_truth[:,:,:,i].flatten()
        y_pred = prediction[:,:,:,i].flatten()
        pr[cls_id] = precision_score(y_true, y_pred)
        rc[cls_id] = recall_score(y_true, y_pred)
        f1[cls_id] = f1_score(y_true, y_pred)
        iou[cls_id] = jaccard_score(y_true, y_pred)
        ratio[cls_id] = y_true.sum() / y_true.size

    _mean = lambda metric: np.mean([m for _, m in metric.items()])
    pr["mean"] = _mean(pr)
    rc["mean"] = _mean(rc)
    f1["mean"] = _mean(f1)
    iou["mean"] = _mean(iou)
    ratio["mean"] = _mean(ratio)
    return {"precision": pr,
            "recall": rc,
            "f1-score": f1,
            "iou": iou,
            "class ratio": ratio}

