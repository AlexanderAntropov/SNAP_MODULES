import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/aantropov/UN_SNAP_SUMMER_2022/MAML_Fast/SNAP_MODULES_FINAL')

import numpy as np
import flexisnap
from flexisnap import *

from snap_metrics import *

lmp_tests = LAMMPS_tests('/home/aantropov/UN_SNAP_SUMMER_2022/MAML_Fast/SNAP_MODULES_FINAL')



lmp_tests.find_lattice_constant()
print(lmp_tests.lattice_constant)

#UU_en = lmp_tests.calculate_defect_energy('defect_for_lammps/UU-dumbell.data', 4.9, label = "UU-d")
#UU_ideal = lmp_tests.calculate_defect_energy('defect_for_lammps/ideal.data', 4.9, label = "ideal")

Uvac_mig = lmp_tests.evaluate_migration('neb/Uvac1.dump', 'neb/Uvac2.coord', neb_id = 2, origin_cell_size = 4.9, label = "")

print(Uvac_mig)

twojmax=6

f_to_en_factor = 0.1 ### error of 0.1 eV/atom equals to error 1 eV/A
f_to_st_factor = 100 ### error of 100 GPa  equals to error 1 eV/A

elem_name_list = ['U', 'N']

bounds_universal = {"en_weight" :  1.0 , "st_weight" :  1.0 }

bounds_per_element = {"U" : { "r_c" : 4.5 , "w_sna" : 1.0 , "weight_of_elem_force" : 1.0 },
                    "N" : { "r_c" : 6.12 , "w_sna" : 0.46 , "weight_of_elem_force" : 0.51 },
}

#train_folder_list = [ '../train_single','../train_UNfluid','../train_Nfluid', '../train_Ufluid',  '../train_ideal_all', '../train_def_all', '../train_U4vac_3600','../train_N5vac_3100', '../train_DNPM' , '../train_cold_def_and_mig',  '../train_surf100',  '../train_surf110',  '../train_surf110_ledge',  '../train_Uint_anti', '../train_UNfluid_add_1eV']
#bounds_folders =    [0.1, 100.43, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
#bounds_folders_en = [0.1, 100.0, 0.1, 1.0, 25.67, 21.95, 1.99, 1.0, 1.0, 72.8, 1.0 , 1.0, 1.0, 5.0, 50.0]
train_folder_list = [ '../train_UNfluid','../train_Nfluid', '../train_Ufluid',  '../train_ideal_all']
bounds_folders_f =    [1.0, 10.0, 8.66, 2.8]
bounds_folders_en = [6.8, 0.1, 0.1, 8.2]
bounds_folders_st = [5.8, 5.1, 5.1, 5.0]

test_folder_list = [ '../train_UNfluid']
test_f = [1]
test_en = [1]
test_st = [1]
test_f_by_elem_w = [1,1]


check_bounds_consistance(elem_name_list, bounds_per_element)
check_bounds_consistance( train_folder_list, bounds_folders_f)
check_bounds_consistance( train_folder_list, bounds_folders_en)
check_bounds_consistance( train_folder_list, bounds_folders_st)

N_coeff_lin = number_of_linear_features(twojmax)
N_coeff_quad = number_of_quad_features(twojmax)

reg_param = 5.0
reg0 = np.array([0.0] * (N_coeff_lin+1) )
reg1 = np.array([reg_param] * N_coeff_quad)
reg = np.concatenate((reg0,reg1,reg0,reg1), axis=0)
reg = np.array(reg)


target_regular_coefs = np.array([0.0] * (((N_coeff_lin+1)+N_coeff_quad)*2) )

def my_metric_function(slp):
    slp.evaluate_test_collection(test_collection, label = "testset")
    slp.count_RMSE_all_features()
    RMSE_predictors = slp.RMSE_all_features
    #snap_model.evaluate_defects()
    #snap_model.evaluate_migration()
    #snap_model.evaluate_stability()
    #RMSE_def = snap_model.RMSE_def
    #RMSE_mig = snap_model.RMSE_mig
    #snap_model.evaluate_melting()
    #melting_pen = snap_model.melting_penalty
    #STAB_PEN = snap_model.stability_penalty
    my_metric = ((RMSE_predictors*5)**2) #+RMSE_def**2+RMSE_mig**2+melting_pen**2)
    return(my_metric)

def to_number(tuple_or_number):
    if type(tuple_or_number)==tuple: return np.mean(tuple_or_number)
    else: return tuple_or_number

r_l = []
w_l = []
f_by_elem_weight_list = []
for elem in elem_name_list:
    r_l.append(to_number(bounds_per_element[elem]['r_c']))
    w_l.append(to_number(bounds_per_element[elem]['w_sna']))
    f_by_elem_weight_list.append(to_number(bounds_per_element[elem]['weight_of_elem_force']))

energy_weight = to_number(bounds_universal['en_weight'])
stress_weight = to_number(bounds_universal['st_weight'])
weight_f_list = [to_number(x) for x in bounds_folders_f]
weight_en_list = [to_number(x) for x in bounds_folders_en]
weight_st_list = [to_number(x) for x in bounds_folders_st]



list_of_collections_train = List_of_Collections(train_folder_list)
list_of_collections_train.set_global_weights(elem_name_list, f_by_elem_weight_list, f_to_en_factor, f_to_st_factor)
list_of_collections_train.set_list_weights(weight_f_list, weight_en_list, weight_st_list)
list_of_collections_train.put_weights_to_collections()
print("QQQ")
list_of_collections_test = List_of_Collections(test_folder_list)
list_of_collections_test.set_global_weights(elem_name_list, test_f_by_elem_w, f_to_en_factor, f_to_st_factor)
list_of_collections_test.set_list_weights(test_f, test_en, test_st)
list_of_collections_test.put_weights_to_collections()
print("QQQ")
#test_collection = list_of_collections_test.create_full_collection_from_list()

slp = SNAP_Learning_Process('/home/aantropov/UN_SNAP_SUMMER_2022/MAML_Fast/SNAP_MODULES_FINAL')

slp.set_parameters(elem_name_list, r_l, w_l, quadratic=True, twojmax=twojmax, threads = 32, regular=reg, target_regular_coefs = target_regular_coefs)       

slp.train_snap(list_of_collections_train)

slp.evaluate_test_data(list_of_collections_test, label = "testset")
slp.count_RMSE_test_data()
RMSE_predictors = slp.RMSE_all_features

slp.ml_output()

#print(" MY METRIC = "+str(round(metric_result,2)), end = '\n')
#print("Latt. const. = "+str(round(snap_model.lattice_constant,2)), end = '\n')
#print(snap_model.defect_dataframe)
#print(snap_model.migration_dataframe)
#print(w_c)

import glob, os
import shutil
for f in glob.glob("tmp*"):
    shutil.rmtree(f, ignore_errors=True)
