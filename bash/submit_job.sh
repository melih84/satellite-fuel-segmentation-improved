#!/bin/bash
#SBATCH --time=3:0:0
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
echo ""

echo "Activating the environment..."
module load cuda
source ~/.venv_tf/bin/activate
 module load gcc opencv/4.10.0 python scipy-stack/2024a
echo "GPU device info:"
python ../compute-canada/tf_device_info.py
python train.py data/winter_conifer_alberta_320x320/winter_conifer_alberta.yaml --epoch 100 --lr-decay

echo "Job ID $SLURM_JOB_ID finished with exit code $? at: `date`"