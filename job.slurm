#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=nv-fys
#SBATCH --time=0-05:00:00
#SBATCH --gres=gpu:A100m40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --job-name="LONG_aug_interpolate"
#SBATCH --output=LONG_aug_interpolate2.out
#SBATCH --mail-user=jawold@stud.ntnu.no
#SBATCH --mail-type=ALL

module purge
module load Anaconda3/2022.05
source /cluster/apps/eb/software/Anaconda3/2022.05/etc/profile.d/conda.sh conda activate myenv
conda activate myenv
python run.py --train --cfg runs/LONG_aug_interpolate/config.ini 