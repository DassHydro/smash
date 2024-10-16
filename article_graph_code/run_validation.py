import smash
import numpy as np
import multiprocessing as mp
from preprocessing_arcmed import load_data
import pandas as pd

DESC_NAME = ["pente",
             "ddr",
             "karst2019_shyreg",
             "foret",
             "urbain",
             "resutilpot",
             "vhcapa",
             ]

def single_run_simulation(config):

    if(config[2] == 'with'):
        read_sm = True

    else:
        read_sm = False

    setup, mesh = load_data('/local/AIX/mettalbi/Projects/smash-maintenance-1.0.x/smash/factory/dataset/Arcmed/catchment_info.csv')

    setup['hydrological_module'] = 'gr4'
    setup['routing_module'] = 'lr'
    setup['start_time'] = '2019-08-01 00:00'
    setup['end_time'] = '2023-08-01 00:00'
    setup['read_sm'] = read_sm
    setup['sm_directory'] = '../../DATA/SMAP_SURFACE_organised'
    setup['prcp_directory'] = '../../DATA/prcp'
    setup['read_pet'] = True
    setup['pet_directory'] = '../../DATA/ETP/ETP-SFR-FRA-INTERA_L93'
    setup['read_qobs'] = True
    setup['qobs_directory'] = "../../DATA/regio_arcmed/QM_60M_extract_2024"
    setup['sm_format'] = 'tif'
    setup['sm_metric'] = config[3]
    setup['maxiter'] = 350
    setup['read_descriptor']= True
    setup['descriptor_directory']="../../DATA/descriptor"
    setup['descriptor_name']=DESC_NAME
    setup['sparse_storage'] = True
    
    model = smash.core.model.model.Model(setup, mesh)

    if(config[1] == 'full'):
        cal_code = mesh['code'].tolist()

    if (config[0] == 'ann'):
        model_calibrated = smash.io.read_model_ddt('results/models_calibrated/Arcmed/Arcmed_'+config[0]+'_'+config[1]+'_'+config[2]+'_sm_'+config[3]+'.hdf5')
        model.rr_parameters.values = model_calibrated['rr_parameters']['values']
        model.forward_run()

        smash.io.save_model_ddt(model, 'results/models_validated/Arcmed/Arcmed_'+config[0]+'_'+config[1]+'_'+config[2]+'_sm_'+config[3]+'_val.hdf5')

    else :
        model_calibrated = smash.io.read_model('results/models_calibrated/Arcmed/Arcmed_'+config[0]+'_'+config[1]+'_'+config[2]+'_sm_'+config[3]+'.hdf5')
        model.rr_parameters.values = model_calibrated.rr_parameters.values

        model.forward_run()

        smash.io.save_model(model, 'results/models_validated/Arcmed/Arcmed_'+config[0]+'_'+config[1]+'_'+config[2]+'_sm_'+config[3]+'_val.hdf5')


ncpu=24

mappings = ['distributed', 'ann']
cal_codes = ['regio']
sm_maps = ['with', 'without']
sm_metrics = ['spatial_bias_insensitive', 'rmse']

configs = []

for mapping in mappings:
    for cal_code in cal_codes:
        for map in sm_maps:
            if(map == 'with'):
                for sm_metric in sm_metrics:
                    configs.append([mapping, cal_code, map, sm_metric])
            else:
                    configs.append([mapping, cal_code, map, 'None'])

pool = mp.Pool(ncpu)
list_model = pool.map(single_run_simulation, [config for config in configs])
# # results, train_cost_df = reduction(list_model)
pool.close()
                    
