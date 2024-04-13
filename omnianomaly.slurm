#!/bin/bash

#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:04:00
#SBATCH --nice=0

#SBATCH --job-name=train_MSL
#SBATCH --output=/mmfs1/home/alexeyy/storage/logs/output_amath.out

#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexeyy@uw.edu

echo "Running Apptainer"
apptainer run --nv --bind /mmfs1/home/alexeyy/storage/School/Research/submodules/OmniAnomaly:/app/omnianomaly /mmfs1/home/alexeyy/storage/School/Research/submodules/OmniAnomaly/omnianomaly.sif --dataset='mypkg_DSN_1k' --max_epoch=20 --batch_size=256
echo "Finished running Apptainer"

