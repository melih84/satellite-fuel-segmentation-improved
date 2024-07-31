Project Compute Canada Cluster: **Graham**
# Bugs
 - The training stops pre-maturely. 
 Error: ` W external/local_tsl/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 7.05GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.`

# TODO
- Evalute the entire verfication set. See what kind of augmentation is needed during training.
- Create a data generator for training. Apply augmentation.


# Note
- model (study-00/run-30): trained with 100 320 x 320 images (re-sized)
- model (study-00/run-34): trained with 1000 320 x 320 images (split)
- model (intermediate/run-01): trained with 100 320 x 320 images (split)
- model (intermediate/run-02): trained with 1000 320 x 320 images (split)

- `requirment.txt` didn't work on Compute Canada.