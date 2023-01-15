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
warnings.simplefilter('ignore')
import time
from monty.os.path import which

from snap_metrics import *
from lammps import lammps

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

def load_list_of_sets(folder_list, verbose = False):
    list_of_train_sets = [] 
    for folder in folder_list:
        list_of_train_sets.append(load_folder(folder, verbose))
    return(list_of_train_sets)

class SNAP_model:
    def set_parameters(self, name_l,r_l,w_l, twojmax , quadratic = False, rcutfac = 0.5):
        self.file_param=open("param", "a")
        self.n_threads = 128
        self.rcutfac = rcutfac
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

    def update_set_weight_list(self, set_weight_list):
        self.set_weight_list = set_weight_list   

    def write_features(self, set, label):
        per_force_features = self.per_force_describer.transform(set['structures'])
        per_force_features.to_csv('train_features_dataframe_'+label, sep=",")

    def write_snap_to_files(self):
        self.snap.write_param()

    def fit_SNAP(self, list_of_train_sets):

        if len(list_of_train_sets)!=len(self.set_weight_list) : 
            print ("Error len(self.list_of_train_sets)!=len(self.set_weight_list) ")
            quit() 

        self.weights_all = []
        for s, set in enumerate(list_of_train_sets):
            self.weights_all+=weights_in_set(set, self.set_weight_list[s], self.elem_weight_list, self.energy_weight, self.stress_weight, self.name_l, en0_weight=self.en0_weight)


        self.train_set = concatenate_sets(list_of_train_sets)

        self.element_profile = {}
        for i in range(len(self.name_l)):
            self.element_profile.update({self.name_l[i]: {'r': self.r_l[i], 'w': self.w_l[i]}})


        warnings.filterwarnings("ignore")

        self.per_force_describer = BispectrumCoefficients(rcutfac=self.rcutfac, twojmax=self.twojmax, 
                                                        element_profile=self.element_profile, 
                                                        quadratic=self.quadratic, 
                                                        pot_fit=True, 
                                                        include_stress=True, 
                                                        n_jobs=self.n_threads, verbose=False)
        tm=time.time()
        ml_model = LinearRegression()
        skl_model = SKLModel(describer=self.per_force_describer, model=ml_model)
        self.snap = SNAPotential(model=skl_model)
        # Train the potential with lists of structures, energies, forces
        self.snap.train(self.train_set['structures'], self.train_set['energies'], self.train_set['forces'], self.train_set['stresses'], include_stress=True, sample_weight=self.weights_all)
        for f in glob.glob("tmp*"):
            shutil.rmtree(f, ignore_errors=True)
        self.write_snap_to_files()
        self.fit_info_str = " En_w {:.3f} St_w {:.3f} // r_l {} w_l {} elem_w {} // folders {}  ///  ".format(self.energy_weight, self.stress_weight, str([round(i, 2) for i in self.r_l]), str([round(i, 2) for i in self.w_l]), str([round(i, 2) for i in self.elem_weight_list]), str([round(i, 2) for i in self.set_weight_list]) )

    def set_test_set(self, list_of_test_sets):
        self.list_of_test_sets = list_of_test_sets
        self.test_set = concatenate_sets(list_of_test_sets)

    def evaluate_testdata(self, test_set, label = ""): 
        test_structures, test_energies, test_forces, test_stresses = test_set['structures'], test_set['energies'], test_set['forces'], test_set['stresses']
        test_feature_types, test_folder, test_filenames = test_set['feature_types'], test_set['feature_folder'], test_set['feature_filename']

        df_orig, df_predict = self.snap.evaluate(test_structures, test_energies, test_forces, test_stresses, include_stress=True)
        orig = df_orig['y_orig'] / df_orig['n']
        predict = df_predict['y_orig'] / df_predict['n']

        test_weights = weights_in_set(test_set, 1, self.elem_weight_list, self.energy_weight, self.stress_weight, self.name_l)

        evaluate_info_dataframe = pd.DataFrame({'orig': orig , 'predict': predict , 'weight': test_weights , 'feature_type' : test_feature_types, 'folder' : test_folder, 'filename' : test_filenames})
        evaluate_info_dataframe.to_csv('orig_vs_predict_'+label, sep=" ")

        self.test_dataframe = evaluate_info_dataframe

    def count_RMSE_predictors(self):
        df = self.test_dataframe
        name_l = list(df.feature_type.unique()) ; name_l.remove("en")
        if ("en0" in name_l) : name_l.remove("en0")
        name_l.remove("st")   
        
        self.RMSE_f = {}
        for elem in name_l:
            self.RMSE_f.update({elem : round(mean_squared_error(df.orig[df.feature_type == elem], df.predict[df.feature_type == elem], squared=False), 3) })
        self.RMSE_e = mean_squared_error(df.orig[df.feature_type == "en"], df.predict[df.feature_type == "en"], squared=False)
        self.RMSE_s = mean_squared_error(df.orig[df.feature_type == "st"], df.predict[df.feature_type == "st"], squared=False) # mean_squared_error(original_stress, weighted_predict_stress)
        self.RMSE_f_total = mean_squared_error(df.orig[df['feature_type'].apply(lambda s: s in name_l )], df.predict[df['feature_type'].apply(lambda s: s in name_l )], squared=False)

        self.force_relative_err = self.RMSE_f_total/df.orig[df['feature_type'].apply(lambda s: s in name_l )].abs().mean()
        self.stress_relative_err = self.RMSE_s/df.orig[df.feature_type == "st"].abs().mean()
        self.energy_relative_err = self.RMSE_e/1 ### характерный порядок энергий, в рамках которого предсказываем - 1 эВ (df.orig[df.feature_type == "en"].max() - df.orig[df.feature_type == "en"].min())

        self.RMSE_predictors = mean_squared_error(df.orig, df.predict, squared=False, sample_weight = df.weight)
        self.RMSE_predictors_info_str = " F {:.3f} (by elem {}) // EN {:.3f} // ST {:.3f} // All = {:.3f} ".format(self.RMSE_f_total, str(self.RMSE_f), self.RMSE_e, self.RMSE_s, self.RMSE_predictors )


    ############### ФУНКЦИИ ДЛЯ РАСЧЕТА ДЕФЕКТОВ ##################

    def evaluate_defects(self):
        self.lattice_constant = find_lattice_constant()
        defects = {}
        keys = ['ideal', 'Uvac', 'Nvac', 'Uint', 'UDD', 'Nint']
        for key in keys:
            defects[key] = calculate_defect_energy(self.lattice_constant, key)

        chempot = defects['ideal']/64
        defects['ideal'] = defects['ideal'] - 64*chempot
        defects['Uvac'] = defects['Uvac'] - 63*chempot
        defects['Nvac'] = defects['Nvac'] - 63*chempot
        defects['Uint'] = defects['Uint'] - 65*chempot
        defects['UDD'] = defects['UDD'] - 65*chempot
        defects['Nint'] = defects['Nint'] - 65*chempot

        reference = {'ideal' : 0.00, 'Uvac' : 3.25, "Nvac" : 1.88, 'Uint' : 7.07, 'Nint' : 3.17, 'UDD' : 5.00}

        self.defect_dataframe = pd.DataFrame({"def" : pd.Series(defects), "ref" : pd.Series(reference)}, index = pd.Series(defects).index)
        
        self.defect_dataframe.loc["UFP"]=self.defect_dataframe.loc["Uvac"]+self.defect_dataframe.loc["Uint"]
        self.defect_dataframe.loc["NFP"]=self.defect_dataframe.loc["Nvac"]+self.defect_dataframe.loc["Nint"]
        self.defect_dataframe.loc["UFP_DD"]=self.defect_dataframe.loc["Uvac"]+self.defect_dataframe.loc["UDD"]
        self.defect_dataframe.loc["SD"]=self.defect_dataframe.loc["Uvac"]+self.defect_dataframe.loc["Nvac"]
        self.defect_dataframe.loc["ASD"]=self.defect_dataframe.loc["Uint"]+self.defect_dataframe.loc["Nint"]
        self.defect_dataframe.loc["ASD_DD"]=self.defect_dataframe.loc["UDD"]+self.defect_dataframe.loc["Nint"]

        self.defect_dataframe['diff'] = (self.defect_dataframe['def'] - self.defect_dataframe['ref']).round(decimals=2)
        self.defect_dataframe['def'].round(decimals=2)
        self.neutral_def=self.defect_dataframe.loc[['UFP','NFP','UFP_DD', 'SD', 'ASD', 'ASD_DD']]
        self.RMSE_def = np.sqrt(mean_squared_error(self.neutral_def['def'], self.neutral_def['ref']))
        self.RMSE_defects_info_str = " // RMSE_DEF = {:.3f} // ".format(self.RMSE_def)

class DE_algo:
    def set_de_parameters(self, mut=0.8, crossp=0.7, popsize=20, its=10000):
        self.mut = mut
        self.crossp=crossp
        self.popsize=popsize 
        self.its=its

    def set_name_l(self, name_l):
        self.name_l = name_l

    def set_snap_parameters(self, quadratic=True, twojmax=6):
        self.quadratic=quadratic
        self.twojmax=twojmax

    def set_train_test_data(self, list_of_train_sets, list_of_test_sets="nan",):
        self.list_of_train_sets = list_of_train_sets
        if list_of_test_sets=="nan": self.list_of_test_sets = list_of_train_sets
        else: self.list_of_test_sets = list_of_test_sets
        self.test_set = concatenate_sets(self.list_of_test_sets)

    def set_bounds(self, bounds_universal, bounds_per_element, bounds_folders):
        self.bounds_universal = bounds_universal
        self.bounds_per_element = bounds_per_element
        self.bounds_folders = bounds_folders

    def set_metric_func(self, metric_func):
        self.metric_func = metric_func

    def concatenate_bounds(self, log_folder_weight = False, log_universal_weights = False):
        def to_tuple(tuple_or_number):
            if type(tuple_or_number)==tuple: return tuple_or_number
            else: return (tuple_or_number, tuple_or_number)

        self.bounds = []
        self.bounds.append(to_tuple(self.bounds_universal['en_weight']))
        self.bounds.append(to_tuple(self.bounds_universal['st_weight']))

        for elem in self.name_l:
            self.bounds.append(to_tuple(self.bounds_per_element[elem]['r_c']))
            self.bounds.append(to_tuple(self.bounds_per_element[elem]['w_sna']))
            self.bounds.append(to_tuple(self.bounds_per_element[elem]['weight_of_elem_force']))   
        
        for bound in self.bounds_folders:
            self.bounds.append(to_tuple(bound))

    def unpack_de_parameter(self, de_parameter_list):
        de_parameter_list = list(de_parameter_list)
        self.energy_weight = de_parameter_list.pop(0)
        self.stress_weight = de_parameter_list.pop(0)
        self.r_l = []
        self.w_l = []
        self.elem_weight_list = []
        for elem in self.name_l:
            self.r_l.append(de_parameter_list.pop(0))
            self.w_l.append(de_parameter_list.pop(0))
            self.elem_weight_list.append(de_parameter_list.pop(0))

        self.set_weight_list = de_parameter_list

    def perform(self):
        self.de_fit_num = 0
        self.concatenate_bounds()
        print('SSSSSSSSSS')
        dimensions = len(self.bounds)
        pop = np.random.rand(self.popsize, dimensions)
        min_b, max_b = np.asarray(self.bounds).T
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + pop * diff
        print('SSSS')
        fitness = np.asarray([self.fit_function(ind) for ind in pop_denorm])
        best_idx = np.argmin(fitness)
        best = pop_denorm[best_idx]
        print("Population made")
        for i in range(self.its):
            for j in range(self.popsize):
                idxs = [idx for idx in range(self.popsize) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
                mutant = np.clip(a + self.mut * (b - c), 0, 1)
                cross_points = np.random.rand(dimensions) < self.crossp
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimensions)] = True
                trial = np.where(cross_points, mutant, pop[j])
                trial_denorm = min_b + trial * diff
                f = self.fit_function(trial_denorm)
                if f < fitness[j]:
                    fitness[j] = f
                    pop[j] = trial
                    if f < fitness[best_idx]:
                        best_idx = j
                        best = trial_denorm
            #yield best, fitness[best_idx]

    def fit_function(self, de_parameter_list):
        self.de_fit_num+=1
        self.unpack_de_parameter(de_parameter_list)
        self.snap_model = SNAP_model()
        self.snap_model.set_parameters(self.name_l, self.r_l, self.w_l, quadratic=self.quadratic, twojmax=self.twojmax)       
        self.snap_model.update_weights(energy_weight=self.energy_weight,stress_weight=self.stress_weight, elem_weight_list=self.elem_weight_list, en0_weight=100)
        self.snap_model.update_set_weight_list(self.set_weight_list)
        self.snap_model.fit_SNAP(self.list_of_train_sets)
        self.snap_model.set_test_set(self.list_of_test_sets)
        self.metric_result = self.metric_func(self.snap_model)
        self.de_output()
        return(self.metric_result)

    def de_output(self):
        f=open("DE_history.txt", "a")

        print(self.de_fit_num, end = ' ')
        print(self.snap_model.fit_info_str, end = ' ')
        print(self.snap_model.RMSE_predictors_info_str, end = ' ')
        print(self.snap_model.RMSE_defects_info_str, end = '\n')
        print(" MY METRIC = "+str(round(self.metric_result,2)), end = '\n')

        f.write(str(self.de_fit_num)+")) ")
        f.write(self.snap_model.fit_info_str+" ")
        f.write(self.snap_model.RMSE_predictors_info_str+" ")
        f.write(self.snap_model.RMSE_defects_info_str+"")
        f.write(" MY METRIC = "+str(round(self.metric_result,2))+"\n")
        f.close()

        f=open("DE_big_history.txt", "a")    
        sm = self.snap_model
        par_list = [self.de_fit_num] + sm.r_l + sm.w_l + sm.elem_weight_list + [sm.energy_weight] + [sm.stress_weight] + sm.set_weight_list
        all_info_row = par_list + [sm.RMSE_f_total, sm.RMSE_e, sm.RMSE_s, sm.RMSE_predictors ] + list(sm.neutral_def['def'])
        f.write(str(all_info_row)[1:-1]+"\n")
        f.close()


def check_bounds_consistance(obj_list, bounds):
    if len(obj_list)!=len(bounds): 
        print("Bounds len error") 
        quit()