#!/bin/bash
#SBATCH -n 1                                                        # Number of cores
#SBATCH -N 1                                                        # Ensure that all cores are on one machine
#SBATCH -t 1-0:00                                                   # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p blackhole # Partition to submit to
#SBATCH --mem=8000                                                  # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o /n/home11/aricarte/slurm_output/%j.out          # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/home11/aricarte/slurm_output/%j.err          # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=ALL                                             # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=angelo.ricarte@cfa.harvard.edu                  # Email to which notifications will be sent

#If you're not Angelo, please change the email settings.
export OMP_NUM_THREADS=1

module load Anaconda/5.0.1-fasrc02
module load hdf5

#This is Angelo's conda environment.
source activate opencv

python computeProfiles.py bondi_multizone_050423_onezone_bflux0_1e-8_2d_n4
python computeProfiles.py bondi_multizone_050423_bflux0_1e-8_2d_n4 
python computeProfiles.py bondi_multizone_042723_bflux0_1e-4_32^3
python computeProfiles.py bondi_multizone_042723_bflux0_1e-4_64^3
python computeProfiles.py bondi_multizone_050123_bflux0_1e-4_96^3
python computeProfiles.py bondi_multizone_050523_bflux0_1e-4_128^3_n3_noshort
python computeProfiles.py bondi_multizone_050123_bflux0_0_64^3
python computeProfiles.py bondi_multizone_050823_bflux0_0_64^3_nojit
python computeProfiles.py bondi_multizone_050123_bflux0_2e-8_32^3_n8
python computeProfiles.py bondi_multizone_050223_bflux0_2e-8_64^3_n8
python computeProfiles.py bondi_multizone_050423_bflux0_2e-8_96^3_n8_test_faster_rst
