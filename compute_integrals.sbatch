#!/bin/bash
#SBATCH -J HNL_amtospheric_integrals
#SBATCH -c 4 # Number of cores
#SBATCH -p shared
#SBATCH --mem 12000
#SBATCH -t 1-08:00:00 # Maximum execution time (D-HH:MM:SS)
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jbook/atmospheric_HNL/integrals%A.out # Standard output
#SBATCH -e /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jbook/atmospheric_HNL/integrals%A.err # Standard error

#SBATCH --mail-user=$USER@g.harvard.edu  
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


source /n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/setup.sh

CMD="source /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jbook/i3/build/env-shell.sh python /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jbook/atmospheric_HNL/get_hnl_integrals.py" 
echo $CMD
$CMD
