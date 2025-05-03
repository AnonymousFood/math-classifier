#!/bin/bash
### Some common partitions
##SBATCH --partition=gpu_a100
##SBATCH --partition=sooner_gpu_test_ada
##SBATCH --partition=sooner_gpu_test
##SBATCH --partition=gpu_a100,sooner_gpu_test,gpu
#SBATCH --partition=gpu,sooner_gpu_test,gpu_a100
##SBATCH --partition=debug_gpu
#
#SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=64
##SBATCH --cpus-per-task=32
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
##SBATCH --mem=32G
##SBATCH --mem=16G
##SBATCH --time=12:00:00
#SBATCH --time=02:00:00
##SBATCH --time=01:00:00
##SBATCH --time=00:10:00
#SBATCH --job-name=encoder
##SBATCH --mail-user=ADD YOUR OWN EMAIL
##SBATCH --mail-type=ALL
#SBATCH --chdir=/scratch/cs529324/final
#SBATCH --output=results/exp_%j_stdout.txt
#SBATCH --error=results/exp_%j_stderr.txt
##SBATCH --array=0-8    # the double ## means that this line is ignored
##SBATCH --array=0-4    # the double ## means that this line is ignored
#SBATCH --array=0    # the double ## means that this line is ignored
##SBATCH --array=1-4    # the double ## means that this line is ignored

#################################################

# NOTE, make sure you change your directory!

#---------------------------
# Setting yaml files to test
#---------------------------

#CONFIG_FILES=("config.yaml" "config1.yaml" "config2.yaml" "config3.yaml" "config4.yaml" "config5.yaml" "config6.yaml" "config7.yaml" "config8.yaml")
CONFIG_FILES=("config.yaml" "config1.yaml" "config2.yaml" "config3.yaml")
#CONFIG_FILES=("config.yaml")
CONFIG=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}

python ./models/encoder.py --config ./configs/"$CONFIG"
#python ./models/encoder.py --config ./configs/config.yaml
