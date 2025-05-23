#!/bin/bash
#SBATCH -p free
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=4:00:00

module load anaconda/2022.05
. ~/.mycondaconf
conda activate bubble-sciml 
module load gcc/11.2.0

python scripts/boxkit_dataset.py
