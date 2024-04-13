#!/bin/bash

#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=1
#SBATCH --time=1-0
#SBATCH --nice=0

#SBATCH --job-name=train_omnianomaly
#SBATCH --output=/mmfs1/home/alexeyy/storage/logs/train_omnianomaly_amath_%j.out

#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexeyy@uw.edu

echo "Running Apptainer"
apptainer run --nv --bind /mmfs1/home/alexeyy/storage/School/Research/submodules/OmniAnomaly:/app/omnianomaly /mmfs1/home/alexeyy/storage/School/Research/submodules/OmniAnomaly/omnianomaly.sif --nf_layers=50 --z_dim=10  --dataset='mypkg_DSN_1k' --max_epoch=20 --batch_size=1400 --test_batch_size=1400 --save_dir='model_hyak_amath' --restore_dir='model_hyak_amath'
echo "Finished running Apptainer"

