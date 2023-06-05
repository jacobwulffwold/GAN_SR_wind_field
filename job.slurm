#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=nv-fys
#SBATCH --time=0-01:00:00
#SBATCH --gres=gpu:A100m40:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=5
#SBATCH --mem=15G
#SBATCH --job-name="1h_param_search"
#SBATCH --output=1h_param_search.out
#SBATCH --mail-user=jawold@stud.ntnu.no
#SBATCH --mail-type=ALL

module purge
module load Anaconda3/2022.05
source /cluster/apps/eb/software/Anaconda3/2022.05/etc/profile.d/conda.sh conda activate myenv
conda activate myenv
python run.py --param_search --cfg config/wind_field_GAN_3D_config_cluster.ini --slurm_array_id 2