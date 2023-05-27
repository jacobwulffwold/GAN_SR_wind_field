#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=nv-fys
#SBATCH --time=1-15:00:00
#SBATCH --gres=gpu:A100m40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --array=1-14
#SBATCH --job-name=GAN_training_Z_handling
#SBATCH --output=%a_Z_handling.out
#SBATCH --mail-user=jawold@stud.ntnu.no
#SBATCH --mail-type=ALL

module purge
module load Anaconda3/2022.05
source /cluster/apps/eb/software/Anaconda3/2022.05/etc/profile.d/conda.sh conda activate myenv
conda activate myenv
python run.py --train --cfg config/wind_field_GAN_3D_config_cluster.ini --slurm_array_id $SLURM_ARRAY_TASK_ID