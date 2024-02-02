#!  /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=11:30:00
#SBATCH --job-name=tensor
#SBATCH --cluster=htc
#SBATCH --gres=gpu:1
#SBATCH --partition=short


module load Anaconda3
module load CUDA/11.8.0
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/apps/system/easybuild/software/CUDA/11.8.0/
source activate /data/math-dewi-nn/ball5622/dewi-tf2-gpu
python run.py


conda deactivate
