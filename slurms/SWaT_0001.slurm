#!/bin/bash

#SBATCH --account=amath
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --time=1-0
#SBATCH --nice=0

#SBATCH --job-name=mypkg_SWaT_0001
#SBATCH --output=/mmfs1/home/alexeyy/storage/logs/mypkg_SWaT_0001_%j.out

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alexeyy@uw.edu

ds=mypkg_SWaT

echo "Running Apptainer"
apptainer run --nv --bind /mmfs1/home/alexeyy/storage/School/Research/submodules/OmniAnomaly:/app/omnianomaly /mmfs1/home/alexeyy/storage/School/Research/submodules/OmniAnomaly/omnianomaly.sif --level=0.0001 --dataset="$ds" --z_dim=3 --nf_layers=20 --max_epoch=20 --initial_lr=0.001 --batch_size=1400 --test_batch_size=1400 --save_dir="$ds" --restore_dir="$ds"
echo "Finished running Apptainer"

