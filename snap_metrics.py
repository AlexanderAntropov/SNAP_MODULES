from multiprocessing import allow_connection_pickling
import sys
import numpy as np
import matplotlib.pyplot as plt
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

def find_lattice_constant():
      lmp = lammps(cmdargs=[ "-screen", os.devnull,  "-nocite"])
      lmp.file('../../SNAP_MODULES/lattice_constant.in')
      with open('log.lammps') as file:
            lines = [line.rstrip() for line in file]
      info = [line for line in lines if (("ENERGY" in line) and("${fe}" not in line))]
      info2 = [line.split() for line in info]
      df = pd.DataFrame(info2, columns=["pff", 'lat', 'en'])
      df['lat'] = df['lat'].astype(float) ;  df['en'] = df['en'].astype(float)
      lat = df[df['en'] == df['en'].min()].lat
      return(float(lat))



def calculate_defect_energy(latt, def_type):
      file1 = open("calc.in", "w")
      file1.write("variable a equal "+ str(latt) +"\n")

      file1.write("units		metal \n\
                  atom_style	atomic \n\
                  boundary    p p p \n\
                  lattice		fcc $a \n\
                  region		box block 0 2 0 2 0 2 \n\
                  create_box	2 box \n\
                  create_atoms 1 box \n\
                  lattice		fcc $a origin 0.5 0.0 0.0 \n\
                  create_atoms 2 box")
      file1.write("\n")
      if def_type=="Uvac":
            file1.write("group del id 1 # 865 \n\
                        delete_atoms group del \n")
      elif def_type=="Nvac":
             file1.write("group del id 35 # 865 \n\
                        delete_atoms group del \n")           
      elif def_type=="Uint":
            file1.write("create_atoms 1 single 0.25 0.25 0.25 \n")
      elif def_type=="Nint":
            file1.write("create_atoms 2 single 0.25 0.25 0.25 \n")
      elif def_type=="UDD":
            file1.write("group del id 1  \n\
                        delete_atoms group del  \n\
                        create_atoms 1 single 1.9 1.9 1.9 \n\
                        create_atoms 1 single 0.1 0.1 0.1 \n")
      
      file1.write( 'neighbor	4.0 bin \n\
                  neigh_modify	every 50 delay 0 check no \n\
                  timestep	0.001 \n\
                  group metal type 1 \n\
                  group nitro type 2 \n\
                  mass 1 238.02 \n\
                  mass 2 14.0067 \n\
                  pair_style snap \n\
                  pair_coeff * * SNAPotential.snapcoeff SNAPotential.snapparam U N \n\
                  thermo 1 \n\
                  fix mom all momentum 10 linear 1 1 1 \n ')
      file1.write("dump myDump all custom 1 " + def_type+"_relax.dump  id type xu yu zu \n")
      file1.write('thermo_style custom step pe \n\
                  minimize 1.0e-11 1.0e-11 100 1000 \n\
                  run		0 \n\
                  variable fe equal pe \n\
                  print "ENERGY_and_MSD ${fe} " file "defect_info.temp"  ' )
      file1.close()
      lmp = lammps(cmdargs=["-log", "none", "-screen", os.devnull,  "-nocite"])
      lmp.file('calc.in')
      with open('defect_info.temp') as file:
            lines = [line.rstrip() for line in file]
      info = lines[0].split()
      return(float(info[1]))



def evaluate_defects(latt = 4.85):
      defects = {}
      keys = ['ideal', 'Uvac', 'Nvac', 'Uint', 'UDD', 'Nint']
      for key in keys:
            defects[key] = calculate_defect(latt, key)

      chempot = defects['ideal']/64
      defects['ideal'] = defects['ideal'] - 64*chempot
      defects['Uvac'] = defects['Uvac'] - 63*chempot
      defects['Nvac'] = defects['Nvac'] - 63*chempot
      defects['Uint'] = defects['Uint'] - 65*chempot
      defects['UDD'] = defects['UDD'] - 65*chempot
      defects['Nint'] = defects['Nint'] - 65*chempot

      reference = {'ideal' : 0.00, 'Uvac' : 3.25, "Nvac" : 1.88, 'Uint' : 7.07, 'Nint' : 3.17, 'UDD' : 5.00}
      for defect_list in [defects, reference]:
            defect_list["UFP"] = defect_list['Uvac'] + defect_list['Uint']
            defect_list["NFP"] = defect_list['Nvac'] + defect_list['Nint']
            defect_list["UFP_DD"] = defect_list['Uvac'] + defect_list['UDD']
            defect_list["SD"] = defect_list['Uvac'] + defect_list['Nvac']
            defect_list["ASD"] = defect_list['Uint'] + defect_list['Nint']
            defect_list["ASD_DD"] = defect_list['UDD'] + defect_list['Nint']
      
      diff = {}
      for key in defect_list:
        diff[key]  = defect_list[key] - reference[key]   
      RMSE_DEF=0.0 
      for key in ['UFP','NFP', 'UFP_DD', 'SD', 'ASD', 'ASD_DD']:
            RMSE_DEF+=((defects[key]-reference[key]))**2    
      print(defects)
      print(reference)
      df = pd.DataFrame.from_dict(defects, orient='index', columns = ["def"])
      df['ref'] = reference.values()
      df['diff'] = (df['def'] - df['ref']).round(decimals=2)
      df['def'].round(decimals=2)
      metric_def=df.loc[['UFP','NFP','UFP_DD', 'SD', 'ASD', 'ASD_DD']]
      print(metric_def)
      rms = np.sqrt(mean_squared_error(metric_def['def'], metric_def['ref']))
      print(rms)
      return(rms)