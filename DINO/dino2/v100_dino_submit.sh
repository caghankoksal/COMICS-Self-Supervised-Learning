#!/bin/bash
# 
# CompecTA (c) 2018
# 
# You should only work under the /scratch/users/<username> directory.
#
# Jupyter job submission script
#
# TODO:
#   - Set name of the job below changing "JupiterNotebook" value.
#   - Set the requested number of nodes (servers) with --nodes parameter.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter. (Total accross all nodes)
#   - Select the partition (queue) you want to run the job in:
#     - short : For jobs that have maximum run time of 120 mins. Has higher priority.
#     - mid   : For jobs that have maximum run time of 1 day..
#     - long  : For jobs that have maximum run time of 7 days. Lower priority than short.
#     - longer: For testing purposes, queue has 31 days limit but only 3 nodes.
#   - Set the required time limit for the job with --time parameter.
#     - Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input/output file names below.
#   - If you do not want mail please remove the line that has --mail-type and --mail-user. If you do want to get notification emails, set your email address.
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch jupyter_submit.sh
#
# -= Resources =-
#

#SBATCH --job-name=dino
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=4
#SBATCH --partition ai 
#SBATCH --account=ai 
#SBATCH --qos=ai -c 1
#SBATCH --mem=130G -c 1
#SBATCH --constraint=tesla_v100 
#SBATCH --gres=gpu:3
#SBATCH --time=168:00:00
#SBATCH --output=jupyter-%J.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ckoksal20@ku.edu.tr

# Please read before you run: http://login.kuacc.ku.edu.tr/#h.3qapvarv2g49

################################################################################
##################### !!! DO NOT EDIT BELOW THIS LINE !!! ######################
################################################################################

# Load Anaconda
echo "======================="
echo "Loading Anaconda Module..."
module load anaconda/3.6
module load julia/latest
module load python/3.7.4
source activate detectron2
echo "======================="
python -m torch.distributed.launch --nproc_per_node=3 main_dino.py --arch vit_small --data_path /path/to/imagenet/train --output_dir /kuacc/users/ckoksal20/DINO/dino/results

