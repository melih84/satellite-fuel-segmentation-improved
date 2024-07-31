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
echo "GPU device info:"
python ../compute-canada/tf_device_info.py
python main.py --study-id "samples_1000" --n_epochs 100 --batch_size 4

echo "Job ID $SLURM_JOB_ID finished with exit code $? at: `date`"