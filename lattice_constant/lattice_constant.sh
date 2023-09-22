#!/bin/bash
lmp_serial -in $1 > lammps_lattice_constant.log
#mpirun -np 4 lmp_mpi -in $1 