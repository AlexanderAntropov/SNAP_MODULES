#!/bin/bash
rm dump.phase
rm melting_count.txt
melt_path=$1
# lmp_serial -in $melt_path/melting.in -var StartFile $2 -var seed `bash -c 'echo $RANDOM'`
mpirun -np 128 lmp_mpi -in $melt_path/melting.in -var StartFile $2 -var seed `bash -c 'echo $RANDOM'` > lammps_melting.log

### Script using OVITO to calculate the number of solid phase atoms
### Separate script because ovito is in another conda env
conda run -n ovito python $melt_path/melting_counter.py