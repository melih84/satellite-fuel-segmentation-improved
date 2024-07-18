Project Compute Canada Cluster: **Graham**
# Bugs
 - Image and labels are not matching when running on CC. (Make sure images are selected by their names and all files are properly transfered)
 - The original image size run out of memery. 320 works fine.

# TODO
- Train a U-Net with size 320 x 320 for 100 epochs.
- Divide images into 4.
- Create a data generator for training 


# Note
- `requirment.txt` didn't work on Compute Canada.