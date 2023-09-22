#!/bin/bash
rm defect_migration.temp
mpirun -np 15 lmp_mpi -partition 15x1 -in $1 -var init $2 -var final $3 -var nebid $4 -var cell_scale $5  | awk '{ print $7  }' | tail -n 1 >> defect_migration.temp

rm screen.*
rm log.lammps.*
rm dump.neb.*