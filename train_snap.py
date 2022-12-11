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
print(which("lmp_serial"))

def load_folder(folder):
    ############### <<< LOAD DATA #####################
    if folder[-1]!="/": folder=folder+"/"
    filelist_in_folder = os.listdir(folder)
    datalist=[loadfn(folder+filename) for filename in filelist_in_folder]
    print("Loaded {} , {} files (configurations)".format(folder, len(datalist)))
    vaspdata = [a for b in datalist for a in b] ### распрямление массива делаем список из списка списков

    structures = [d['structure'] for d in vaspdata]
    if 'data' in vaspdata[0] : energies = [d['data']['energy_per_atom']*len(d['structure']) for d in vaspdata]     ###    if 'outputs' in vaspdata[0] : energies = [d['outputs']['energy'] for d in vaspdata]
    if 'data' in vaspdata[0] : forces = [d['data']['forces'] for d in vaspdata]                                    ###  if 'outputs' in vaspdata[0] : forces = [d['outputs']['forces'] for d in vaspdata]
    stresses = [d['data']['virial_stress'] for d in vaspdata]
    stresses = [[d/10.0 for d in tenz] for tenz in stresses] ### maml wants GPa

    print(" # of structures in data: {}\n".format(len(structures)),
        "# of energies in data: {}\n".format(len(energies)),
        "# of forces in data: {}\n".format(len(forces)))

    feature_types = []
    for configuration in structures:
        if len(configuration)>1 : feature_types.append("en")
        else : feature_types.append("en0")

        for LatticeCite in configuration:
            for i in range(3) : feature_types.append(str(LatticeCite.species.elements[0]))
        for i in range(6) :  feature_types.append("st")
    #Возвращает кортеж соответствующих списков - структуры, энергии, силы, давления и список типов фичей, полученные из всех структур в папке
    return({'structures': structures, 'energies': energies, 'forces': forces, 'stresses': stresses, 'feature_types' : feature_types})


def weights_in_set(set, set_weight, elem_weight_list, energy_weight, stress_weight, name_l):
    en_weight_default = energy_weight/0.1 ### Ошибка в 0.1 эВ/атом равносильна ошибке в 1 эВ/А
    st_weight_default = stress_weight/100 ### Ошибка в 100 GPa равносильна ошибке в 1 эВ/А , 6 компонент
    f_weight_default = 1 ## 

    weights_in_set = []
    en_count = set["feature_types"].count("en0")+set["feature_types"].count("en")
    st_count = set["feature_types"].count("st")
    f_count = (len(set["feature_types"]) - en_count - st_count)/en_count
    for feature_type in set["feature_types"]:
        weight = 1.0
        if feature_type == "en0" : weight =  20   *   en_weight_default*set_weight/en_count
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
        return({'structures': structures, 'energies': energies, 'forces': forces, 'stresses': stresses, 'feature_types' : feature_types})

class SNAP_model:
    def set_parameters(self, name_l,r_l,w_l):
        self.file_param=open("param", "a")
        self.n_threads = 128
        self.rcutfac = 5.00
        self.name_l = name_l
        self.r_l = r_l
        self.w_l = w_l

    def update_weights(self, energy_weight, stress_weight , elem_weight_list):
        self.energy_weight = energy_weight
        self.stress_weight = stress_weight 
        self.elem_weight_list = elem_weight_list

    def load_train_data(self, folder_list, set_weight_list):
        list_of_sets = [] 
        for folder in folder_list:
            list_of_sets.append(load_folder(folder))
        self.list_of_sets = list_of_sets
        self.set_weight_list = set_weight_list
        if len(self.list_of_sets)!=len(self.set_weight_list) : 
            print ("Error len(self.list_of_sets)!=len(self.set_weight_list) ")
            quit() 

    def fit_SNAP(self, quadratic=True):
        
        self.weights_all = []
        for s, set in enumerate(self.list_of_sets):
            self.weights_all+=weights_in_set(set, self.set_weight_list[s], self.elem_weight_list, self.energy_weight, self.stress_weight, self.name_l)

        structures = [a for set in self.list_of_sets for a in set['structures']]
        energies = [a for set in self.list_of_sets for a in set['energies']]
        forces = [a for set in self.list_of_sets for a in set['forces']]
        stresses = [a for set in self.list_of_sets for a in set['stresses']]

        self.element_profile = {}
        for i in range(len(self.name_l)):
            self.element_profile.update({self.name_l[i]: {'r': self.r_l[i], 'w': self.w_l[i]}})
        print(self.element_profile)


        warnings.filterwarnings("ignore")

        per_force_describer = BispectrumCoefficients(rcutfac=self.rcutfac, twojmax=6, 
                                                        element_profile=self.element_profile, 
                                                        quadratic=quadratic, 
                                                        pot_fit=True, 
                                                        include_stress=True, 
                                                        n_jobs=self.n_threads, verbose=True)
        tm=time.time()
        ml_model = LinearRegression()
        skl_model = SKLModel(describer=per_force_describer, model=ml_model)
        self.snap = SNAPotential(model=skl_model)
        # Train the potential with lists of structures, energies, forces
        self.snap.train(structures, energies, forces, stresses, include_stress=True, sample_weight=self.weights_all)
        print(time.time()-tm)
        self.snap.write_param()


    def evaluate_testdata(self, test_set): 
        test_structures, test_energies, test_forces, test_stresses, test_feature_types = test_set['structures'], test_set['energies'], test_set['forces'], test_set['stresses'], test_set['feature_types']
        df_orig, df_predict = self.snap.evaluate(test_structures, test_energies, test_forces, test_stresses, include_stress=True)
        orig = df_orig['y_orig'] / df_orig['n']
        predict = df_predict['y_orig'] / df_predict['n']

        test_weights = weights_in_set(test_set, 1, self.elem_weight_list, self.energy_weight, self.stress_weight, self.name_l)

        ### WRITE GRAPH FILES ###        
        energy_indices = [i for i, feature_type in enumerate(test_feature_types) if (feature_type == "en" or feature_type == "en0")]
        forces_indices = [i for i, feature_type in enumerate(test_feature_types) if (feature_type in self.name_l)]
        stress_indices = [i for i, feature_type in enumerate(test_feature_types) if (feature_type == "st")]
        file_fl_test = open('forces_linear_test.test_set', "w")
        file_el_test = open('energies_linear_test.test_set', "w")
        file_st_test = open('stress_linear_test.test_set', "w")
        for index in forces_indices: file_fl_test.write(str(orig[index])+" "+str(predict[index])+"\n")
        for index in energy_indices: file_el_test.write(str(orig[index])+" "+str(predict[index])+"\n")
        for index in stress_indices: file_st_test.write(str(orig[index])+" "+str(predict[index])+"\n")
        file_fl_test.close() ; file_el_test.close() ; file_st_test.close()
        
        original_energy = orig[energy_indices] ; original_forces = orig[forces_indices] ; original_stress = orig[stress_indices]
        predict_energy = predict[energy_indices] ; predict_forces = predict[forces_indices] ; predict_stress = predict[stress_indices]

        RMSE_f = mean_squared_error(original_forces, predict_forces, squared=False)
        RMSE_e = mean_squared_error(original_energy, predict_energy, squared=False)
        RMSE_s = mean_squared_error(original_stress, predict_stress, squared=False) # mean_squared_error(original_stress, weighted_predict_stress)

        RMSE_ml = mean_squared_error(orig, predict, squared=False, sample_weight = test_weights)

        print("RMSE F, EN, ST, RMSE_ml = {:.3f} {:.3f} {:.3f} {:.3f}".format(RMSE_f, RMSE_e, RMSE_s, RMSE_ml ))
        return ((RMSE_f, RMSE_e, RMSE_s, RMSE_ml))


if __name__ == '__main__' :

    name_l = ['U', 'N']
    r_l = [ 0.4,0.6,0.5]
    w_l = [1.0,1.0,1.0]
    elem_weight_list = [1.0,1.0,1.0]
    energy_weight = 5.55
    stress_weight = 3.33

    folder_list = ['train_ideal_1']#,'train_single']
    set_weight_list = [1]


    snap_model = SNAP_model()
    snap_model.set_parameters(name_l, r_l, w_l)
    snap_model.update_weights(energy_weight=energy_weight,stress_weight=stress_weight, elem_weight_list=elem_weight_list)
    snap_model.load_train_data(folder_list, set_weight_list=set_weight_list)
    tm=time.time()
    snap_model.fit_SNAP()
    print(time.time()-tm)

    train_set = concatenate_sets(snap_model.list_of_sets)
    test_set = load_folder("test_UN")
    snap_model.evaluate_testdata(train_set)
    snap_model.evaluate_testdata(test_set)

