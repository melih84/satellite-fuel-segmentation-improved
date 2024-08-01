import skimage as ski
import matplotlib.pyplot as plt

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000

# Load image
path_to_image = "./data/verification_dataset_320x320/main.png"
image = ski.io.imread(path_to_image)

# Load mask
path_to_mask = "./experiments/intermediate/run-02/verification_results/mask.png"
mask = ski.io.imread(path_to_mask)
print(image.shape, mask.shape)

# Combine image & mask
alpha = 0.8
overlay = (image * alpha + mask * (1-alpha)).astype("uint8")

# Save
plt.imsave("overlay.png", overlay)