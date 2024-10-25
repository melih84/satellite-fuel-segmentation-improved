# TODO (updated on 10/25/2024):
- Model:
    - Remove zero padding for valid padding
    - Add regularization
    - Add adata augmentaiton in Augmentor class (flip)
- Training:
    - Add the epoch, miou, and loss for to the saved models names
- Inference:
    - Handle & automated GDAL on the cluster (document)
    - Create world files for png binary masks instead of tiff (.png->.pgw) 
- Experiment:
    - Train the modle on both Edmonton & Moncton dataset
- Labeling:
    - Use the current winter and summer models for pre-labeling (json output)



# Activate environemnt on Compute Canada:
`. bash/activate_env.sh` or `source bash/activate_env.sh`



Project Compute Canada Cluster: **Graham**
# Bugs
 - The training stops pre-maturely. 
 Error: ` W external/local_tsl/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 7.05GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.`




# Note

- `requirment.txt` didn't work on Compute Canada.