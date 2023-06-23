#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=nv-fys
#SBATCH --time=0-00:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --array=0-8
#SBATCH --job-name=norm_size
#SBATCH --output=%a_norm_size.out
#SBATCH --mail-user=jawold@stud.ntnu.no
#SBATCH --mail-type=ALL

module purge
module load Anaconda3/2022.05
source /cluster/apps/eb/software/Anaconda3/2022.05/etc/profile.d/conda.sh conda activate myenv
conda activate myenv
python run.py --train --cfg ./runs/lr200k_STD/config.ini --slurm_array_id $SLURM_ARRAY_TASK_ID