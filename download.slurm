#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=nv-fys
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --job-name="download_data_for_bessaker_wind_farm"
#SBATCH --output=download_check.out
#SBATCH --mail-user=jawold@stud.ntnu.no
#SBATCH --mail-type=ALL

module purge
module load Anaconda3/2022.05
source /cluster/apps/eb/software/Anaconda3/2022.05/etc/profile.d/conda.sh conda activate myenv
conda activate myenv
python run.py --download --cfg config/wind_field_GAN_3D_config_cluster.ini