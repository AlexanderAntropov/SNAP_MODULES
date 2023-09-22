from multiprocessing import allow_connection_pickling
import sys
import numpy as np
from monty.serialization import loadfn
from maml.utils import pool_from, convert_docs
import json
import os
import random
import subprocess
import glob, shutil

# local environment descriptors imports
from maml.describers import BispectrumCoefficients
from sklearn.decomposition import PCA

# machine learning interatomic potentials imports
from maml.base import SKLModel
from maml.apps.pes import SNAPotential
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

import pandas as pd

# materials properties prediction imports
from pymatgen.core import Structure, Lattice
from maml.apps.pes import LatticeConstant, ElasticConstant, NudgedElasticBand, DefectFormation

# disable logging information
import logging
logging.disable(logging.CRITICAL)
import warnings

from lammps import lammps

class LAMMPS_tests:
      def __init__(self, my_folder):
           self.my_folder = my_folder

      def find_lattice_constant(self):
            subprocess.call([self.my_folder+"/lattice_constant/lattice_constant.sh", self.my_folder+"/lattice_constant/lattice_constant.in" ])
            table = np.loadtxt("lattice_constant.temp")
            i = np.argmin(table[:,1])
            lat = table[i][0]
            self.lattice_constant = lat

      def calculate_energy(self, StartFile, origin_cell_size, label = ""):
            cell_scale = str(self.lattice_constant/origin_cell_size)
            subprocess.call([self.my_folder+"/defect_formation/defect.sh", self.my_folder+"/defect_formation/defect.in", StartFile, cell_scale, label])
            form_energy = np.loadtxt("defect_info.temp")
            return(form_energy)

      def calculate_migration_energy(self, StartFile, FinishFile, neb_id, origin_cell_size, label = ""):
            cell_scale = str(self.lattice_constant/origin_cell_size)
            #mpirun -np 15 lmp_mpi -partition 15x1 -in $neb_path/neb.in -var init $2 -var final $3 -var nebid $4 -var cell_scale $5 | awk '{ print $7  }' | tail -n 1 >> defect_migration.temp
            subprocess.call([self.my_folder+"/neb/defect_migration.sh", self.my_folder+"/neb/neb.in", StartFile, FinishFile, str(neb_id), str(cell_scale), label])
            file3 = open("defect_migration.temp", "r")
            # считываем все строки
            mig_en = float(file3.readlines()[0])
            return(mig_en)

      def evaluate_melting(self, StartFile):
            melting_path = self.my_folder+"/melting"
            subprocess.call([melting_path+"/test_melt.sh", melting_path, StartFile])
            file_m = open("melting_count.temp", "r")
            solid = float(file_m.readlines()[0])
            liquid = 1 - solid
            return(liquid)

      def check_lost_atoms(self):
            file3 = open("log.lammps", "r")
            # считываем все строки
            lines = file3.readlines()
            file3.close
            if "Lost atoms" in lines[-2]:
                  return(True)
            else:
                  return(False)

