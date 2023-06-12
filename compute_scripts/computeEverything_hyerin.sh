#!/bin/bash
#SBATCH -n 1                                                        # Number of cores
#SBATCH -N 1                                                        # Ensure that all cores are on one machine
#SBATCH -t 1-0:00                                                   # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p blackhole # Partition to submit to
#SBATCH --mem=8000                                                  # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ./%j.out          # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./%j.err          # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=ALL                                             # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=hyerin.cho@cfa.harvard.edu                  # Email to which notifications will be sent

#If you're not Angelo, please change the email settings.
export OMP_NUM_THREADS=1

source ~/venv3/bin/activate

#python computeProfiles.py 052523_bflux_n8_32^3 
#python computeProfiles.py 060223_bflux0_n8_32^3_nlim10k
#python computeProfiles.py production_runs/bondi_bz2e-8_1e8_nonlim
#python computeProfiles.py production_runs/bondi_bz2e-8_1e8_128
#python computeProfiles.py bondi_multizone_030723_bondi_128^3
#python computeProfiles.py bondi_multizone_052523_gizmo_n8_64^3_noshock
#python computeProfiles.py bondi_multizone_050423_bflux0_2e-8_96^3_n8_test_faster_rst
#python computeProfiles.py production_runs/gizmo_extg_1e8
#python computeProfiles.py bondi_multizone_050423_onezone_bflux0_1e-8_2d_n4
#python computeProfiles.py bondi_multizone_050123_onezone_bflux0_1e-4_64^3
python computeProfiles.py 061023_bflux0_n8_gamma3
