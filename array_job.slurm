#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=nv-fys
#SBATCH --time=0-00:20:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --array=0-2
#SBATCH --job-name=test_best_4
#SBATCH --output=%a_test_best_4.out
#SBATCH --mail-user=jawold@stud.ntnu.no
#SBATCH --mail-type=ALL

module purge
module load Anaconda3/2022.05
source /cluster/apps/eb/software/Anaconda3/2022.05/etc/profile.d/conda.sh conda activate myenv
conda activate myenv
python run.py --train --cfg config/wind_field_GAN_3D_config_cluster.ini --slurm_array_id $SLURM_ARRAY_TASK_ID