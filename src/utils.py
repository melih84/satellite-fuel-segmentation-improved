import numpy as np
import matplotlib.pyplot as plt

COLOR_CODE = {
    "orange": [255, 165, 0],
    "darkgreen": [1, 50, 32],
    "limegreen": [50, 205, 50],
    "black": [0, 0, 0]
    }

class Visualize():
    def __init__(self, color_scheme):
        self.color_scheme = color_scheme
    
    def combine_class(self, masks):
        comb_mask = np.zeros(masks.shape[:2] + (3,)).astype("uint")
        n_class = masks.shape[-1]
        for cls in range(n_class):
            comb_mask += self._paint(masks[:,:,cls], self.color_scheme[cls])
        return comb_mask
    
    def display(self, image, ground_truth=None, predictions=None):
        n = 1
        n += 1 if ground_truth is not None else 0
        n += 1 if predictions is not None else 0
        fig, ax = plt.subplots(1, n)
        ax[0].imshow(image)
        ax[0].set_title("Input Image")
        i = 1
        if ground_truth is not None:
           actuals = self.combine_class(ground_truth)
           ax[i].imshow(actuals)
           ax[i].set_title("Ground Truth")
           i += 1
        if predictions is not None:   
            preds = self.combine_class(predictions)
            ax[i].imshow(preds)
            ax[i].set_title("Predicted Mask")
            i += 1
        
        [a.axis("off") for a in ax]
        fig.tight_layout()
    
    @staticmethod
    def _paint(mask, color):
        new_mask = np.zeros(mask.shape + (3,))
        for i in range(3):
            new_mask[:,:,i] = mask * COLOR_CODE[color][i]
        return new_mask.astype("uint")