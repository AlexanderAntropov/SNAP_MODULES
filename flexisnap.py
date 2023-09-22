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

from custom_regression import *

# local environment descriptors imports
from maml.describers import BispectrumCoefficients
from sklearn.decomposition import PCA

# machine learning interatomic potentials imports
from maml.base import SKLModel
from maml.apps.pes import SNAPotential
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge

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
from os.path import exists

from lammps import lammps


class List_of_Collections:
    """
    Different collections can have different weights, so the full training set is a list of collections, not the single one
    Attributes
    ----------
    coll_list : list of collection obj
    """
    def __init__(self, folder_list, verbose=False):
        """
        Parameters
        ----------
        folder_list : list of str
            Folders reffering to collections with .json files
        """
        self.folder_list = folder_list # List of folders containing .json files (list of str)
        self.coll_list = []  # literally a python list of Collection_of_configurations objects.
        for folder in self.folder_list:
            coll = Collection_of_configurations()
            coll.load_collection_from_folder(folder, verbose=verbose)
            self.coll_list.append(coll)

    def set_global_weights(self, elem_name_list, f_by_elem_weight_list, f_to_en_factor = 0.1, f_to_st_factor = 100):
        self.elem_name_list = elem_name_list
        self.f_by_elem_weight_list = f_by_elem_weight_list
        self.f_to_en_factor = f_to_en_factor
        self.f_to_st_factor = f_to_st_factor

    def set_list_weights(self, weight_f_list, weight_en_list, weight_st_list):
        self.weight_f_list = weight_f_list
        self.weight_en_list = weight_en_list
        self.weight_st_list = weight_st_list
        if len(self.coll_list)!=len(self.weight_f_list): print ("Error len(collection_list)!=len(coll_weight_f_list)") ; quit() 
        if len(self.coll_list)!=len(self.weight_en_list): print ("Error len(collection_list)!=len(coll_weight_en_list)") ; quit()
        if len(self.coll_list)!=len(self.weight_st_list): print ("Error len(collection_list)!=len(coll_weight_st_list)") ; quit()
        
    def put_weights_to_collections(self):
        for i, coll in enumerate(self.coll_list):
            coll.set_weights_in_coll(self.weight_f_list[i], self.weight_en_list[i], self.weight_st_list[i], self.elem_name_list, self.f_by_elem_weight_list, self.f_to_en_factor, self.f_to_st_factor)

    def create_full_collection_from_list(self):
        full_coll = Collection_of_configurations()
        full_coll.structures = [a for coll in self.coll_list for a in coll.structures]
        full_coll.energies = [a for coll in self.coll_list for a in coll.energies]
        full_coll.forces = [a for coll in self.coll_list for a in coll.forces]
        full_coll.stresses = [a for coll in self.coll_list for a in coll.stresses]
        full_coll.feature_types = [a for coll in self.coll_list for a in coll.feature_types]
        full_coll.source_folder = [a for coll in self.coll_list for a in coll.source_folder]
        full_coll.source_file = [a for coll in self.coll_list for a in coll.source_file]

        full_coll.weights = [a for coll in self.coll_list for a in coll.weights]
        return(full_coll)

class Collection_of_configurations:
    def load_collection_from_folder(self, folder, verbose = False):
           ############### <<< LOAD DATA #####################
        if folder[-1]!="/": folder=folder+"/"
        filelist_in_folder = os.listdir(folder)
        datalist={filename : loadfn(folder+filename)[0] for filename in filelist_in_folder}
        if verbose: print("Loaded {} , {} files (configurations)".format(folder, len(datalist)))

        self.structures = [datalist[filename]['structure'] for filename in datalist]
        self.energies = [datalist[filename]['data']['energy_per_atom']*len(datalist[filename]['structure']) for filename in datalist]     ###    if 'outputs' in vaspdata[0] : energies = [d['outputs']['energy'] for d in vaspdata]
        self.forces = [datalist[filename]['data']['forces'] for filename in datalist]                                    ###  if 'outputs' in vaspdata[0] : forces = [d['outputs']['forces'] for d in vaspdata]
        self.stresses = [datalist[filename]['data']['virial_stress'] for filename in datalist]
        self.stresses = [[stress/10.0 for stress in tenz] for tenz in self.stresses] ### maml wants GPa

        if verbose: print(" # of structures in data: {}\n".format(len(self.structures)),
            "# of energies in data: {}\n".format(len(self.energies)),
            "# of forces in data: {}\n".format(len(self.forces)))

        self.feature_types = []
        self.source_file = []
        for num, configuration in enumerate(self.structures):
            self.feature_types.append("en")
            self.source_file.append(filelist_in_folder[num])
            for LatticeCite in configuration:
                for i in range(3) : 
                    self.feature_types.append(str(LatticeCite.species.elements[0]))
                    self.source_file.append(filelist_in_folder[num])
            for i in range(6) :  
                self.feature_types.append("st") 
                self.source_file.append(filelist_in_folder[num])
        self.source_folder = [folder[:-1]]* len(self.feature_types)
        self.weights = [1.0] * len(self.feature_types)
        #Возвращает кортеж соответствующих списков - структуры, энергии, силы, давления и список типов фичей, полученные из всех структур в папке
        #return({'structures': structures, 'energies': energies, 'forces': forces, 'stresses': stresses, 'feature_types' : feature_types, 'feature_folder' : folder, 'source_file' : source_file})
    
    def set_weights_in_coll(self,  coll_weight_f, coll_weight_en, coll_weight_st, elem_name_list, f_by_elem_weight_list, f_to_en_factor = 0.1, f_to_st_factor = 100):
        #en_weight_default = coll_weight_en/f_to_en_factor ### Ошибка в 0.1 эВ/атом равносильна ошибке в 1 эВ/А
        #st_weight_default = coll_weight_st/f_to_st_factor ### Ошибка в 100 GPa равносильна ошибке в 1 эВ/А , 6 компонент
        #f_weight_default = 1 ## 

        self.weights = []
        en_count = self.feature_types.count("en")
        st_count = self.feature_types.count("st")
        f_count = len(self.feature_types) - en_count - st_count
        for feature_type in self.feature_types:
            weight = 1.0
            if feature_type == "en" : weight =  coll_weight_en/en_count/f_to_en_factor
            elif feature_type == "st" : weight = coll_weight_st/f_to_st_factor/st_count
            else:
                for i in range(len(elem_name_list)):
                    if feature_type == elem_name_list[i] : weight = coll_weight_f*f_by_elem_weight_list[i]/f_count
            self.weights.append(weight)

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
    source_file = []
    for num, configuration in enumerate(structures):
        if len(configuration)>1 : feature_types.append("en")
        else : feature_types.append("en0")
        source_file.append(filelist_in_folder[num])

        for LatticeCite in configuration:
            for i in range(3) : 
                feature_types.append(str(LatticeCite.species.elements[0]))
                source_file.append(filelist_in_folder[num])
        for i in range(6) :  
            feature_types.append("st") 
            source_file.append(filelist_in_folder[num])
    folder = folder[:-1]
    #Возвращает кортеж соответствующих списков - структуры, энергии, силы, давления и список типов фичей, полученные из всех структур в папке
    return({'structures': structures, 'energies': energies, 'forces': forces, 'stresses': stresses, 'feature_types' : feature_types, 'feature_folder' : folder, 'source_file' : source_file})


def weights_in_set(set, set_weight, set_weight_en, elem_weight_list, energy_weight, stress_weight, name_l, en0_weight = 20.0):
    en_weight_default = energy_weight/0.1 ### Ошибка в 0.1 эВ/атом равносильна ошибке в 1 эВ/А
    st_weight_default = stress_weight/100 ### Ошибка в 100 GPa равносильна ошибке в 1 эВ/А , 6 компонент
    f_weight_default = 1 ## 

    weights_in_set = []
    en_count = set["feature_types"].count("en0")+set["feature_types"].count("en")
    st_count = set["feature_types"].count("st")
    f_count = len(set["feature_types"]) - en_count - st_count
    for feature_type in set["feature_types"]:
        weight = 1.0
        if feature_type == "en0" : weight =  en0_weight   *   en_weight_default*set_weight/en_count
        if feature_type == "en" : weight =  en_weight_default*set_weight_en/en_count
        for i in range(len(name_l)):
            if feature_type == name_l[i] : weight = f_weight_default*elem_weight_list[i]*set_weight/f_count
        if feature_type == "st" : weight = st_weight_default*set_weight/6.0/st_count
        weights_in_set.append(weight)
    return(weights_in_set)

def concatenate_collections(list_of_collections):
        structures = [a for set in list_of_collections for a in set['structures']]
        energies = [a for set in list_of_collections for a in set['energies']]
        forces = [a for set in list_of_collections for a in set['forces']]
        stresses = [a for set in list_of_collections for a in set['stresses']]
        feature_types = [a for set in list_of_collections for a in set['feature_types']]
        folder = [set['feature_folder'] for set in list_of_collections for a in set['feature_types']]
        filenames = [a for set in list_of_collections for a in set['source_file']]
        return({'structures': structures, 'energies': energies, 'forces': forces, 'stresses': stresses, 'feature_types' : feature_types, 'feature_folder' : folder, 'source_file' : filenames})

class SNAP_Learning_Process:
    def __init__(self, my_folder):
        self.my_folder = my_folder

    def set_parameters(self, elem_name_list,r_l,w_l, twojmax , rcutfac = 0.5, quadratic = False, threads = 4, regularization = False, regular_attraction = None, target_regular_list = None ):
        self.regularization = regularization
        self.target_regular_list = target_regular_list 
        self.regular_attraction = regular_attraction
        self.file_param=open("param", "a")
        self.n_threads = threads
        self.rcutfac = rcutfac
        self.elem_name_list = elem_name_list
        self.r_l = r_l
        self.w_l = w_l
        self.quadratic = quadratic
        self.twojmax = twojmax

        self.RMSE_all_features_info_str = ""
        self.RMSE_defects_info_str = ""
        self.RMSE_mig_info_str = ""
        self.RMSE_stab_info_str = ""
        self.melting_info_str = ""
        self.lattice_constant = 0.0
        self.RMSE_f_total = 0.0
        self.RMSE_e = 0.0
        self.RMSE_s = 0.0
        self.RMSE_all_features = 0.0

    #def set_f_by_elem_weight_list(self, f_by_elem_weight_list):
    #    self.f_by_elem_weight_list = f_by_elem_weight_list


    def write_features(self, set, label):
        per_force_features = self.per_force_describer.transform(set['structures'])
        per_force_features.to_csv('train_features_dataframe_'+label, sep=",")

    def write_features_per_atom(self, set, label):
        self.per_atom_describer = BispectrumCoefficients(rcutfac=self.rcutfac, twojmax=self.twojmax, 
                                                        element_profile=self.element_profile, 
                                                        quadratic=self.quadratic, 
                                                        pot_fit=False, 
                                                        include_stress=False, 
                                                        n_jobs=self.n_threads, verbose=False)
        per_atom_features = self.per_atom_describer.transform(set['structures'])
        per_atom_features.to_csv('atom_features_'+label, sep=",")

    def write_nonquadratic_per_force_describers(self):
        per_force_describer_nonq = BispectrumCoefficients(rcutfac=self.rcutfac, twojmax=self.twojmax, 
                                                        element_profile=self.element_profile, 
                                                        quadratic=False, 
                                                        pot_fit=True, 
                                                        include_stress=True, 
                                                        n_jobs=self.n_threads, verbose=False)
        per_force_features_nonq = per_force_describer_nonq.transform(self.train_set['structures'])
        np.savetxt('nonquadratic_per_force_describers.txt', np.array(per_force_features_nonq) , delimiter=' ', fmt='%1.4e')

    def write_train_feature_types(self):
        file = open('feature_types_train.txt', 'w')
        for elem in self.train_set['feature_types']:
            file.write(str(elem +"\n"))
        file.close()

    def train_snap(self, list_of_collections):
        self.train_full_collection = list_of_collections.create_full_collection_from_list()

        self.element_profile = {}
        for i in range(len(self.elem_name_list)):
            self.element_profile.update({self.elem_name_list[i]: {'r': self.r_l[i], 'w': self.w_l[i]}})

        warnings.filterwarnings("ignore")
        self.per_force_describer = BispectrumCoefficients(rcutfac=self.rcutfac, twojmax=self.twojmax, 
                                                        element_profile=self.element_profile, 
                                                        quadratic=self.quadratic, 
                                                        pot_fit=True, 
                                                        include_stress=True, 
                                                        n_jobs=self.n_threads, verbose=False)
        
        if self.regularization:
            if isinstance(self.regular_attraction, int) or isinstance(self.regular_attraction, float):
                ml_model = Ridge(alpha = self.regular_attraction)
                print("Ridge Regression with constant L2 attraction coefficient for all dimensions")
            else:
                print("Custom FlexiSNAPLinearRegression with list of L2 attraction coefficient for all dimensions")
                ml_model = FlexiSNAPLinearRegression(regular_attraction=self.regular_attraction,  target_regular_list = self.target_regular_list)
        else:
            ml_model = LinearRegression()     
        skl_model = SKLModel(describer=self.per_force_describer, model=ml_model)
        self.snap = SNAPotential(model=skl_model)
        # Train the potential with lists of structures, energies, forces
        self.snap.train(self.train_full_collection.structures, self.train_full_collection.energies, self.train_full_collection.forces, self.train_full_collection.stresses, include_stress=True, sample_weight=self.train_full_collection.weights)
        #for f in glob.glob("tmp*"):
        #    shutil.rmtree(f, ignore_errors=True)
        self.snap.write_param()
        self.fit_info_str = "r_l {} w_l {} f_by_elem_w {} // f_list {} en_list {} st_list {}  ///  ".format(str([round(i, 2) for i in self.r_l]), str([round(i, 2) for i in self.w_l]), str([round(i, 2) for i in list_of_collections.f_by_elem_weight_list]), str([round(i, 2) for i in list_of_collections.weight_f_list]), str([round(i, 2) for i in list_of_collections.weight_en_list]), str([round(i, 2) for i in list_of_collections.weight_st_list]) )

    #def set_test_set(self, list_of_test_sets):
    #    self.list_of_test_sets = list_of_test_sets
    #    self.test_set = concatenate_sets(list_of_test_sets)

    def evaluate_test_data(self, list_of_collections_test, label = ""): 
        self.test_full_collection = list_of_collections_test.create_full_collection_from_list()

        df_orig, df_predict = self.snap.evaluate(self.test_full_collection.structures, self.test_full_collection.energies, self.test_full_collection.forces, self.test_full_collection.stresses, include_stress=True)
        orig = df_orig['y_orig'] / df_orig['n']
        predict = df_predict['y_orig'] / df_predict['n']

        self.test_dataframe = pd.DataFrame({'orig': orig , 'predict': predict , 'feature_type' : self.test_full_collection.feature_types, 'folder' : self.test_full_collection.source_folder, 'filename' : self.test_full_collection.source_file, 'weight' : self.test_full_collection.weights})
        self.test_dataframe.to_csv('orig_vs_predict_'+label, sep=" ")

    def count_RMSE_test_data(self):
        df = self.test_dataframe
        elem_name_list = list(df.feature_type.unique())
        elem_name_list.remove("en")
        elem_name_list.remove("st")   
        
        self.RMSE_f_by_elem = {}
        for elem in elem_name_list:
            self.RMSE_f_by_elem.update({elem : round(mean_squared_error(df.orig[df.feature_type == elem], df.predict[df.feature_type == elem], squared=False), 3) })
        self.RMSE_e = mean_squared_error(df.orig[df.feature_type == "en"], df.predict[df.feature_type == "en"], squared=False)
        self.RMSE_s = mean_squared_error(df.orig[df.feature_type == "st"], df.predict[df.feature_type == "st"], squared=False) # mean_squared_error(original_stress, weighted_predict_stress)
        self.RMSE_f_all = mean_squared_error(df.orig[df['feature_type'].apply(lambda s: s in elem_name_list )], df.predict[df['feature_type'].apply(lambda s: s in elem_name_list )], squared=False)

        self.RMSE_weighted = mean_squared_error(df.orig, df.predict, squared=False, sample_weight = df.weight)
        self.RMSE_test_info_str = " F {:.3f} (by elem {}) // EN {:.3f} // ST {:.3f} // All weighted = {:.3f} ".format(self.RMSE_f_all, str(self.RMSE_f_by_elem), self.RMSE_e, self.RMSE_s, self.RMSE_weighted )

    def ml_output(self):
        print(self.fit_info_str, end = ' ')
        print(self.RMSE_predictors_info_str, end = ' ')
        print(self.RMSE_defects_info_str, end = ' ')
        print(self.RMSE_mig_info_str, end = '\n')


class DE_algo:
    #https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/

    def __init__(self, number_of_replicas, my_number, mut=0.8, crossp=0.7, popsize=20):
        self.number_of_replicas = number_of_replicas
        self.my_number = my_number
        self.mut = mut
        self.crossp=crossp
        self.popsize=popsize 
        if self.popsize<4:
            raise IOError("Popsize_should_be>3")

    def set_elements(self, elem_name_list):
        self.elem_name_list =  elem_name_list

    def set_limits(self, r_l_limits, w_l_limits, f_by_elem_weight_list_limits, weight_f_list_limits, weight_en_list_limits, weight_st_list_limits, log_scale = False):
        self.r_l_limits = r_l_limits
        self.w_l_limits = w_l_limits
        self.f_by_elem_weight_list_limits = f_by_elem_weight_list_limits
        self.weight_f_list_limits = weight_f_list_limits
        self.weight_en_list_limits = weight_en_list_limits
        self.weight_st_list_limits = weight_st_list_limits
        self.log_scale = log_scale

        def to_tuple(tuple_or_number, log=False):
            if log:
                if type(tuple_or_number)==tuple: return (np.log10(tuple_or_number[0]), np.log10(tuple_or_number[1]))
                else: return (np.log10(tuple_or_number), np.log10(tuple_or_number))
            else:
                if type(tuple_or_number)==tuple: return tuple_or_number
                else: return (tuple_or_number, tuple_or_number)

        self.bounds = []
        for elem in self.r_l_limits:
            self.bounds.append(to_tuple(elem))
        for elem in self.w_l_limits:
            self.bounds.append(to_tuple(elem))

        for elem in self.f_by_elem_weight_list_limits:
            self.bounds.append(to_tuple(elem, log = self.log_scale))  
        
        for limit in self.weight_f_list_limits:
            self.bounds.append(to_tuple(limit, log = self.log_scale))
        for limit in self.weight_en_list_limits:
            self.bounds.append(to_tuple(limit, log = self.log_scale))
        for limit in self.weight_st_list_limits:
            self.bounds.append(to_tuple(limit, log = self.log_scale))

    def unpack_de_point(self, de_point):
        de_point = list(de_point)
        r_l = []
        w_l = []
        f_by_elem_weight_list = []
        if self.log_scale:
            for elem in self.elem_name_list:
                r_l.append(de_point.pop(0))
            for elem in self.elem_name_list:
                w_l.append(de_point.pop(0))
            for elem in self.elem_name_list:
                f_by_elem_weight_list.append(10**(de_point.pop(0)))
            k = int(len(de_point)/3)
            weight_f_list = [10**x for x in de_point[:k]]
            weight_en_list = [10**x for x in de_point[k:2*k]]
            weight_st_list = [10**x for x in de_point[2*k:]]
        else: 

            for elem in self.elem_name_list:
                r_l.append(de_point.pop(0))
            for elem in self.elem_name_list:
                w_l.append(de_point.pop(0))
            for elem in self.elem_name_list:
                f_by_elem_weight_list.append(de_point.pop(0))
            k = int(len(de_point)/3)
            weight_f_list = [x for x in de_point[:k]]
            weight_en_list = [x for x in de_point[k:2*k]]
            weight_st_list = [x for x in de_point[2*k:]]
        return((r_l , w_l, f_by_elem_weight_list, weight_f_list, weight_en_list, weight_st_list))

    def create_population(self):
        dimensions = len(self.bounds)
        pop = np.random.rand(self.popsize, dimensions)
        min_b, max_b = np.asarray(self.bounds).T
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + pop * diff

        self.fitness = np.asarray([self.try_function(self.unpack_de_point(ind)) for ind in pop_denorm])
        best_idx = np.argmin(self.fitness)
        self.best = pop_denorm[best_idx]
        np.savetxt("../prl_pop_"+str(self.my_number), pop_denorm , delimiter=' ', fmt='%1.6e')
        np.savetxt("../prl_fitness_"+str(self.my_number), self.fitness , delimiter=' ', fmt='%1.6e')
        print("Population created and counted")

        self.my_gen = 0
        file_gen = open("../prl_generation_"+str(self.my_number), 'w')
        file_gen.write(str(self.my_gen))
        file_gen.close()

    def one_step(self):
        dimensions = len(self.bounds)
        min_b, max_b = np.asarray(self.bounds).T

        file_gen = open("../prl_generation_"+str(self.my_number), "r")
        self.my_gen = int(file_gen.readlines()[0])
        file_gen.close()

        all_ready = False
        while not all_ready:
            all_ready = True
            for repl in range(self.number_of_replicas):
                repl_ready = False
                if exists("../prl_generation_"+str(repl)):
                    file_m = open("../prl_generation_"+str(repl), "r")
                    if int(file_m.readlines()[0]) >= self.my_gen: 
                        repl_ready = True
                    file_m.close()
                all_ready = bool(all_ready*repl_ready)
            print("waiting other replicas")
            time.sleep(5)

        self.fitness = np.loadtxt("../prl_fitness_"+str(self.my_number), delimiter=" ")
        pop_denorm = np.loadtxt("../prl_pop_"+str(self.my_number), delimiter=" ")
        all_population = pop_denorm
        for repl in range(self.number_of_replicas):
            if repl != self.my_number:
                read = np.loadtxt("../prl_pop_"+str(repl), delimiter=" ")
                all_population = np.concatenate((all_population, read ), axis = 0)                  

        for j in range(self.popsize):
            idxs = [idx for idx in range(len(all_population)) if idx != j]
            a, b, c = all_population[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + self.mut * (b - c), min_b, max_b)
            cross_points = np.random.rand(dimensions) < self.crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial_denorm = np.where(cross_points, mutant, pop_denorm[j])
            f = self.try_function(self.unpack_de_point(trial_denorm))
            if f < self.fitness[j]:
                self.fitness[j] = f
                pop_denorm[j] = trial_denorm

        self.my_gen+=1
        file_gen = open("../prl_generation_"+str(self.my_number), 'w')
        file_gen.write(str(self.my_gen))
        file_gen.close()
        np.savetxt("../prl_pop_"+str(self.my_number), pop_denorm , delimiter=' ', fmt='%1.6e')
        np.savetxt("../prl_fitness_"+str(self.my_number), self.fitness , delimiter=' ', fmt='%1.6e')
        #yield best, fitness[best_idx]
        print("Population changed")

def check_bounds_consistance(obj_list, bounds):
    if len(obj_list)!=len(bounds): 
        print("Bounds len error") 
        quit()

def number_of_linear_features(twojmax):
    Nlin=0
    for j1 in range(0,twojmax+1):
        for j2 in range(0,j1+1):
            for j in range(j1-j2,min(twojmax,j1+j2)+1,2):
                if (j>=j1): Nlin+=1
    return(int(Nlin))

def number_of_quad_features(twojmax):
    Nlin=number_of_linear_features(twojmax)
    Nquad = Nlin*(Nlin+1)/2
    return(int(Nquad))