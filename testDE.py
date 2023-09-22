import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/aantropov/UN_SNAP_SUMMER_2022/MAML_Fast/SNAP_MODULES_FINAL')

import numpy as np
import flexisnap
from flexisnap import *

from lammps_tests import *

#UU_en = lmp_tests.calculate_defect_energy('defect_for_lammps/UU-dumbell.data', 4.9, label = "UU-d")
#UU_ideal = lmp_tests.calculate_defect_energy('defect_for_lammps/ideal.data', 4.9, label = "ideal")

twojmax=6

f_to_en_factor = 0.1 ### error of 0.1 eV/atom equals to error 1 eV/A
f_to_st_factor = 100 ### error of 100 GPa  equals to error 1 eV/A

elem_name_list = ['U', 'N']

bounds_universal = {"en_weight" :  1.0 , "st_weight" :  1.0 }

bounds_per_element = {"U" : { "r_c" : 4.5 , "w_sna" : 1.0 , "weight_of_elem_force" : 1.0 },
                    "N" : { "r_c" : 6.12 , "w_sna" : 0.46 , "weight_of_elem_force" : 0.51 },
}

r_l_limits = [(4,5),(4,5)]
w_l_limits = [(0.1,1),(0.1,1)]
f_by_elem_weight_list_limits = [(0.1,1),(0.1,1)]

#train_folder_list = [ '../train_single','../train_UNfluid','../train_Nfluid', '../train_Ufluid',  '../train_ideal_all', '../train_def_all', '../train_U4vac_3600','../train_N5vac_3100', '../train_DNPM' , '../train_cold_def_and_mig',  '../train_surf100',  '../train_surf110',  '../train_surf110_ledge',  '../train_Uint_anti', '../train_UNfluid_add_1eV']
#bounds_folders =    [0.1, 100.43, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
#bounds_folders_en = [0.1, 100.0, 0.1, 1.0, 25.67, 21.95, 1.99, 1.0, 1.0, 72.8, 1.0 , 1.0, 1.0, 5.0, 50.0]
train_folder_list = [ '../train_UNfluid','../train_Nfluid', '../train_Ufluid',  '../train_ideal_all']
weight_f_list_limits =    [1.0, 10.0, 8.66, 2.8]
weight_en_list_limits = [6.8, 0.1, 0.1, 8.2]
weight_st_list_limits = [5.8, 5.1, 5.1, 5.0]

test_folder_list = [ '../train_UNfluid']
test_f = [1]
test_en = [1]
test_st = [1]
test_f_by_elem_w = [1,1]


check_bounds_consistance(elem_name_list, bounds_per_element)
check_bounds_consistance( train_folder_list, weight_f_list_limits)
check_bounds_consistance( train_folder_list, weight_en_list_limits)
check_bounds_consistance( train_folder_list, weight_st_list_limits)

N_coeff_lin = number_of_linear_features(twojmax)
N_coeff_quad = number_of_quad_features(twojmax)

reg_param = 5.0
reg0 = np.array([0.0] * (N_coeff_lin+1) )
reg1 = np.array([reg_param] * N_coeff_quad)
reg = np.concatenate((reg0,reg1,reg0,reg1), axis=0)
reg = np.array(reg)


target_regular_list = np.array([0.0] * (((N_coeff_lin+1)+N_coeff_quad)*2) )


list_of_collections_train = List_of_Collections(train_folder_list)

list_of_collections_test = List_of_Collections(test_folder_list)
list_of_collections_test.set_global_weights(elem_name_list, test_f_by_elem_w, f_to_en_factor, f_to_st_factor)
list_of_collections_test.set_list_weights(test_f, test_en, test_st)
list_of_collections_test.put_weights_to_collections()


slp = SNAP_Learning_Process('/home/aantropov/UN_SNAP_SUMMER_2022/MAML_Fast/SNAP_MODULES_FINAL')
lmp_tests = LAMMPS_tests('/home/aantropov/UN_SNAP_SUMMER_2022/MAML_Fast/SNAP_MODULES_FINAL')

def try_function(tuple):
    (r_l , w_l, f_by_elem_weight_list, weight_f_list, weight_en_list, weight_st_list) = tuple
    list_of_collections_train.set_global_weights(elem_name_list, f_by_elem_weight_list, f_to_en_factor, f_to_st_factor)
    list_of_collections_train.set_list_weights(weight_f_list, weight_en_list, weight_st_list)
    list_of_collections_train.put_weights_to_collections()
    slp.set_parameters(elem_name_list, r_l, w_l, quadratic=True, twojmax=twojmax, threads = 32, regularization = True, regular_attraction = reg, target_regular_list = target_regular_list)       
    slp.train_snap(list_of_collections_train)
    print(slp.fit_info_str)

    slp.evaluate_test_data(list_of_collections_test, label = "testset")
    slp.count_RMSE_test_data()
    print(slp.RMSE_test_info_str)

    lmp_tests.find_lattice_constant()
    print(lmp_tests.lattice_constant)

    U_ideal = lmp_tests.calculate_energy(StartFile = "defects/ideal.dump", origin_cell_size = 4.825, label = "U_int")
    U_int_en = lmp_tests.calculate_energy(StartFile = "defects/Uint.dump", origin_cell_size = 4.825, label = "U_int")
    formation_penalty = (7.2-(U_int_en-U_ideal))**2
    print("formation_energy = "+str(U_int_en-U_ideal))

    UU_mig_en = lmp_tests.calculate_migration_energy(StartFile = "neb/Uvac1.dump", FinishFile = "neb/Uvac2.coord", neb_id=2, origin_cell_size = 4.825, label = "UU-d")
    migration_penalty = (2.5-UU_mig_en)**2
    print("migration_energy = "+str(UU_mig_en))

    liquid_fraction = lmp_tests.evaluate_melting(StartFile = "melting/UN_two_phase_start.dump")
    if lmp_tests.check_lost_atoms():
        print("Lost atoms")
        melting_penalty = 100
    else:
        melting_penalty = (0.5-liquid_fraction)**2
        print("melting_penalty = "+str(melting_penalty))


    result = slp.RMSE_weighted**2 + formation_penalty +  migration_penalty + melting_penalty
    return(result)


de_algo = DE_algo(1,0,mut=0.6, crossp=0.5, popsize=4)
de_algo.try_function = try_function
de_algo.set_elements(elem_name_list)

de_algo.set_limits(r_l_limits, w_l_limits, f_by_elem_weight_list_limits, weight_f_list_limits, weight_en_list_limits, weight_st_list_limits, log_scale = False)


de_algo.create_population()
for i in range(100):
    de_algo.one_step()

import glob, os
import shutil
for f in glob.glob("tmp*"):
    shutil.rmtree(f, ignore_errors=True)
