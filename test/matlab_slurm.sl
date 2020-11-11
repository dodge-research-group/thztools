#!/bin/bash -l
#SBATCH --account=def-jsdodge #adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=0-20:00     #adjust this to match the walltime of your job
#SBATCH --nodes=1      
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12  #adjust this if you are using PCT
#SBATCH --mem-per-cpu=5000         #adjust this according to your the memory requirement per node you need
#SBATCH --job-name=biasfit
#SBATCH --output=./logs/%x-%j.out
#SBATCH --mail-user=jsdodge@sfu.ca #adjust this to match your email address
#SBATCH --mail-type=ALL

#Load the appropriate matlab module
module load matlab/2020a

srun matlab -nodisplay -r "oldpath=path; path(oldpath,'../lib'); biasfit; exit;"
