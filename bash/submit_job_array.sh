#!/bin/bash
#SBATCH --array=1-6
#SBATCH --time=4:0:0
#SBATCH --mem-per-cpu=36GB
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#$SBATCH --mail-user=jamshid2@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o slurm/%N-%j.out

echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $SLURM_TASK_ID"
echo ""

echo "Activating the environment..."
module load cuda
source ~/.venv_tf/bin/activate
module load gcc opencv/4.10.0 python scipy-stack/2024a
echo "GPU device info:"
python ../compute-canada/tf_device_info.py
python detect.py inference/Moncton-full-tif-resized16/$SLURM_TASK_ID weights/winter_moncton_unet_v1.0_ep_30_val_miou_8056.keras --name "moncton"

echo "Task ID $SLURM_TASK_ID Job ID $SLURM_JOB_ID finished with exit code $? at: `date`"

inference/Moncton-full-tif-resized16/1