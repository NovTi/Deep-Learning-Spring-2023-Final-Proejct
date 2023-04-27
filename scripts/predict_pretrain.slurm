#!/bin/bash

#SBATCH --job-name=Pred_PT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=70GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,v100,rtx8000

# job info
config_path=$1
exp_name=$2
exp_id=$3

# Singularity path
ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
cd /scratch/$USER/NYU/DL/AE-Seg
python -m src.predict_pretrain --config ${config_path} --exp_name ${exp_name} --exp_id ${exp_id}
"
