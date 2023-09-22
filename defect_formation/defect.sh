#!/bin/bash
lmp_serial -in $1 -var StartFile $2 -var cell_scale $3 -var label $4  > lammps_defect_formation.log
#mpirun -np 4 lmp_mpi -in $1 