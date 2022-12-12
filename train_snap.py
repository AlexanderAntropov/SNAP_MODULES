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
warnings.simplefilter('ignore')
import time
from monty.os.path import which
#print(which("lmp_serial"))

def load_folder(folder, verbose = False):
    ############### <<< LOAD DATA #####################
    if folder[-1]!="/": folder=folder+"/"
    filelist_in_folder = os.listdir(folder)
    datalist={filename : loadfn(folder+filename)[0] for filename in filelist_in_folder}

    if verbose: print("Loaded {} , {} files (configurations)".format(folder, len(datalist)))

    structures = [datalist[filename]['structure'] for filename in datalist]
    energies = [datalist[filename]['data']['energy_per_atom']*len(datalist[filename]['structure']) for filename in datalist]     ###    if 'outputs' in vaspdata[0] : energies = [d['outputs']['energy'] for d in vaspdata]
    forces = [datalist[filename]['data']['forces'] for filename in datalist]                                    ###  if 'outputs' in vaspdata[0] : forces = [d['outputs']['forces'] for d in vaspdata]
    stresses = [datalist[filename]['data']['virial_stress'] for filename in datalist]
    stresses = [[stress/10.0 for stress in tenz] for tenz in stresses] ### maml wants GPa

    if verbose: print(" # of structures in data: {}\n".format(len(structures)),
        "# of energies in data: {}\n".format(len(energies)),
        "# of forces in data: {}\n".format(len(forces)))

    feature_types = []
    feature_filename = []
    for num, configuration in enumerate(structures):
        if len(configuration)>1 : feature_types.append("en")
        else : feature_types.append("en0")
        feature_filename.append(filelist_in_folder[num])

        for LatticeCite in configuration:
            for i in range(3) : 
                feature_types.append(str(LatticeCite.species.elements[0]))
                feature_filename.append(filelist_in_folder[num])
        for i in range(6) :  
            feature_types.append("st") 
            feature_filename.append(filelist_in_folder[num])
    folder = folder[:-1]
    #Возвращает кортеж соответствующих списков - структуры, энергии, силы, давления и список типов фичей, полученные из всех структур в папке
    return({'structures': structures, 'energies': energies, 'forces': forces, 'stresses': stresses, 'feature_types' : feature_types, 'feature_folder' : folder, 'feature_filename' : feature_filename})


def weights_in_set(set, set_weight, elem_weight_list, energy_weight, stress_weight, name_l, en0_weight = 20.0):
    en_weight_default = energy_weight/0.1 ### Ошибка в 0.1 эВ/атом равносильна ошибке в 1 эВ/А
    st_weight_default = stress_weight/100 ### Ошибка в 100 GPa равносильна ошибке в 1 эВ/А , 6 компонент
    f_weight_default = 1 ## 

    weights_in_set = []
    en_count = set["feature_types"].count("en0")+set["feature_types"].count("en")
    st_count = set["feature_types"].count("st")
    f_count = (len(set["feature_types"]) - en_count - st_count)/en_count
    for feature_type in set["feature_types"]:
        weight = 1.0
        if feature_type == "en0" : weight =  en0_weight   *   en_weight_default*set_weight/en_count
        if feature_type == "en" : weight =  en_weight_default*set_weight
        for i in range(len(name_l)):
            if feature_type == name_l[i] : weight = f_weight_default*elem_weight_list[i]*set_weight/f_count
        if feature_type == "st" : weight = st_weight_default*set_weight/6.0
        weights_in_set.append(weight)
    return(weights_in_set)

def concatenate_sets(list_of_sets):
        structures = [a for set in list_of_sets for a in set['structures']]
        energies = [a for set in list_of_sets for a in set['energies']]
        forces = [a for set in list_of_sets for a in set['forces']]
        stresses = [a for set in list_of_sets for a in set['stresses']]
        feature_types = [a for set in list_of_sets for a in set['feature_types']]
        folder = [set['feature_folder'] for set in list_of_sets for a in set['feature_types']]
        filenames = [a for set in list_of_sets for a in set['feature_filename']]
        return({'structures': structures, 'energies': energies, 'forces': forces, 'stresses': stresses, 'feature_types' : feature_types, 'feature_folder' : folder, 'feature_filename' : filenames})

class SNAP_model:
    def set_parameters(self, name_l,r_l,w_l, twojmax , quadratic = False):
        self.file_param=open("param", "a")
        self.n_threads = 128
        self.rcutfac = 5.00
        self.name_l = name_l
        self.r_l = r_l
        self.w_l = w_l
        self.quadratic = quadratic
        self.twojmax = twojmax

    def update_weights(self, energy_weight, stress_weight , elem_weight_list, en0_weight=20.0):
        self.energy_weight = energy_weight
        self.stress_weight = stress_weight 
        self.elem_weight_list = elem_weight_list
        self.en0_weight = en0_weight

    def load_train_data(self, folder_list, set_weight_list, verbose = False):
        self.list_of_train_sets = [] 
        for folder in folder_list:
            self.list_of_train_sets.append(load_folder(folder, verbose))
        self.set_weight_list = set_weight_list
        if len(self.list_of_train_sets)!=len(self.set_weight_list) : 
            print ("Error len(self.list_of_train_sets)!=len(self.set_weight_list) ")
            quit() 

    def fit_SNAP(self):
        
        self.weights_all = []
        for s, set in enumerate(self.list_of_train_sets):
            self.weights_all+=weights_in_set(set, self.set_weight_list[s], self.elem_weight_list, self.energy_weight, self.stress_weight, self.name_l, en0_weight=self.en0_weight)


        self.train_set = concatenate_sets(self.list_of_train_sets)

        self.element_profile = {}
        for i in range(len(self.name_l)):
            self.element_profile.update({self.name_l[i]: {'r': self.r_l[i], 'w': self.w_l[i]}})


        warnings.filterwarnings("ignore")

        per_force_describer = BispectrumCoefficients(rcutfac=self.rcutfac, twojmax=self.twojmax, 
                                                        element_profile=self.element_profile, 
                                                        quadratic=self.quadratic, 
                                                        pot_fit=True, 
                                                        include_stress=True, 
                                                        n_jobs=self.n_threads, verbose=False)
        tm=time.time()
        ml_model = LinearRegression()
        skl_model = SKLModel(describer=per_force_describer, model=ml_model)
        self.snap = SNAPotential(model=skl_model)
        # Train the potential with lists of structures, energies, forces
        self.snap.train(self.train_set['structures'], self.train_set['energies'], self.train_set['forces'], self.train_set['stresses'], include_stress=True, sample_weight=self.weights_all)
        self.snap.write_param()


def evaluate_testdata(snap_model, test_set, label = ""): 
    test_structures, test_energies, test_forces, test_stresses = test_set['structures'], test_set['energies'], test_set['forces'], test_set['stresses']
    test_feature_types, test_folder, test_filenames = test_set['feature_types'], test_set['feature_folder'], test_set['feature_filename']

    df_orig, df_predict = snap_model.snap.evaluate(test_structures, test_energies, test_forces, test_stresses, include_stress=True)
    orig = df_orig['y_orig'] / df_orig['n']
    predict = df_predict['y_orig'] / df_predict['n']

    test_weights = weights_in_set(test_set, 1, snap_model.elem_weight_list, snap_model.energy_weight, snap_model.stress_weight, snap_model.name_l)

    evaluate_info_dataframe = pd.DataFrame({'orig': orig , 'predict': predict , 'weight': test_weights , 'feature_type' : test_feature_types, 'folder' : test_folder, 'filename' : test_filenames})
    evaluate_info_dataframe.to_csv('orig_vs_predict_'+label, sep=" ")

    return(evaluate_info_dataframe)

def count_RMSE(evaluate_info_dataframe):
    df = evaluate_info_dataframe
    name_l = list(df.feature_type.unique()) ; name_l.remove("en")
    if ("en0" in name_l) : name_l.remove("en0")
    name_l.remove("st")   
    
    RMSE_f = {}
    for elem in name_l:
        RMSE_f.update({elem : mean_squared_error(df.orig[df.feature_type == elem], df.predict[df.feature_type == elem], squared=False) })
    RMSE_e = mean_squared_error(df.orig[df.feature_type == "en"], df.predict[df.feature_type == "en"], squared=False)
    RMSE_s = mean_squared_error(df.orig[df.feature_type == "st"], df.predict[df.feature_type == "st"], squared=False) # mean_squared_error(original_stress, weighted_predict_stress)
    RMSE_f_total = mean_squared_error(df.orig[df['feature_type'].apply(lambda s: s in name_l )], df.predict[df['feature_type'].apply(lambda s: s in name_l )], squared=False)

    RMSE_ml = mean_squared_error(df.orig, df.predict, squared=False, sample_weight = df.weight)

    print("F {:.3f} // EN {:.3f} // ST {:.3f} // All = {:.3f}".format(RMSE_f_total, RMSE_e, RMSE_s, RMSE_ml ), end = " // ")
    print("F by Elem ", end = '')
    print(RMSE_f, end = '   ')
    return (RMSE_f, RMSE_f_total, RMSE_e, RMSE_s, RMSE_ml)

def metrika_1(evaluate_info_dataframe):
    df = evaluate_info_dataframe
    name_l = list(df.feature_type.unique()) ; name_l.remove("en")
    if ("en0" in name_l) : name_l.remove("en0")
    name_l.remove("st")    

    RMSE_e = mean_squared_error(df.orig[df.feature_type == "en"], df.predict[df.feature_type == "en"], squared=False)
    RMSE_f_total = mean_squared_error(df.orig[df['feature_type'].apply(lambda s: s in name_l )], df.predict[df['feature_type'].apply(lambda s: s in name_l )], squared=False)
    RMSE_s = mean_squared_error(df.orig[df.feature_type == "st"], df.predict[df.feature_type == "st"], squared=False)

    force_relative_err = RMSE_f_total/df.orig[df['feature_type'].apply(lambda s: s in name_l )].abs().mean()
    stress_relative_err = RMSE_s/df.orig[df.feature_type == "st"].abs().mean()
    energy_relative_err = RMSE_e/(df.orig[df.feature_type == "en"].max() - df.orig[df.feature_type == "en"].min())

    return(np.sqrt((force_relative_err**2 + stress_relative_err**2 +  energy_relative_err**2)/3))





if __name__ == '__main__' :

    name_l = ['U', 'N']
    r_l = [ 0.5,0.4]
    w_l = [1.0,1.0]

    energy_weight = 1
    stress_weight = 1

    folder_list = ['train_ideal_all']#,'train_single']
    set_weight_list = [1]

    elem_weight_list = [1.0,1.0]

    for qua in [True, False]:
        for twj in [6,8]:
            print(qua)    
            print(twj)   
            tm = time.time()
            snap_model = SNAP_model()
            snap_model.set_parameters(name_l, r_l, w_l, quadratic=qua, twojmax=twj)       
            snap_model.update_weights(energy_weight=energy_weight,stress_weight=stress_weight, elem_weight_list=elem_weight_list, en0_weight=100)
            snap_model.load_train_data(folder_list, set_weight_list=set_weight_list)
            snap_model.fit_SNAP()

            train_set = concatenate_sets(snap_model.list_of_train_sets)
            dataframe = evaluate_testdata(snap_model,snap_model.train_set)
            count_RMSE(dataframe)
            print(metrika_1(dataframe))
            print(time.time()-tm)

            potential_name = "UN_"+str(int(snap_model.quadratic))+"_twj"+str(snap_model.twojmax)
            os.rename("SNAPotential.snapparam", potential_name+".snapparam")
            os.rename("SNAPotential.snapcoeff", potential_name+".snapcoeff")


import glob, os
import shutil
for f in glob.glob("tmp*"):
    shutil.rmtree(f, ignore_errors=True)
