#!/bin/bash

# Job name
##SBATCH --job-name=schlight

# Define the files which will contain the Standard and Error output
##SBATCH --output=/home/aolza/scratch/projects/imagination_perception/outputs/M8_%x.out
##SBATCH --error=/home/aolza/scratch/projects/imagination_perception/outputs/M8_%x.err

# Number of tasks that compose the job
#SBATCH --ntasks=1

# Advanced use
#SBATCH --cpus-per-task=4
# #SBATCH --threads-per-core=2
# #SBATCH --ntasks-per-core=2

# Required memory (Default 2GB)
#SBATCH --mem-per-cpu=4G

# Select one partition
# ML-CPU // Cola de trabajos en CPUs con AVX-512 y (VNNI Vector Neural Network Instructions)
## ML-GPU // Cola de trabajos en la GPU 
## GENERIC //Trabajos genericos que no requieran TF2 o PyTorch

#Uso de ML-CPU
#SBATCH --partition=ML-CPU

#Uso de GENERIC
##SBATCH --partition=GENERIC

#Uso de ML-GPU (Si hay trabajos multi-gpu el numero puede variar de 1 a 4)
# #SBATCH --partition=ML-GPU
# #SBATCH --gres=gpu:1

# If you are using arrays, specify the number of tasks in the array
##SBATCH --array=1-XX

#Ejemplo: En el caso de lanzar algo con python hay que incluir priscilla exec.
#         En el caso de binarios es necesario. 
  
echo   "priscilla exec python3 $1 $2 $3 $4 $5 $6 $7 $8"      
        priscilla exec python3 $1 $2 $3 $4 $5 $6 $7 $8

# Example: sbatch raw.sh raw.py imaginacion_fusiform 0
