#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition gpgpuB
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=kdb19 # required to send email notifcations - please replace 'your_username' with your college login name or email address
# script to run the training on the Imperial DoC Slurm Cluster (necessary for Longformer for larger GPUs)
configName="longformer-fce.json"
trainData="data/processed/fce/json/train.json"
evalData="data/processed/fce/json/dev.json"

source /homes/${USER}/.bashrc
conda activate dissertation
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
/usr/bin/nvidia-smi
cd /vol/bitbucket/${USER}/document-classification-transformers

#Add the path to your python script below, making sure the file exists, and any other commands
python main.py --config config/$configName --eval $evalData --train $trainData
