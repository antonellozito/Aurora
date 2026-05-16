import sys, os
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.signal
import pickle
import importlib


sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))+'/devices')


# LOAD EQUILIBRIUM

def equilibrium(device,shot,shotfile='EQH',time = 0.,exp='augd'):
    
    exp_data = importlib.import_module("exp_data_" + device + "_ELMs")
    
    equilibrium_file = exp_data.equilibrium(shot,shotfile=shotfile,time=time,exp=exp)
    
    print('Equilibrium data for device ' + device + ' for shot ' + str(shot) + ' loaded.')
        
    return equilibrium_file


# LOAD ELECTRON DENSITY

def ne(device,shot,time_inter_ELM_start,time_inter_ELM_end,radial_shift_kin,ELM_frequency,time_resolution,sim_time,exp_time_shift):

    exp_data = importlib.import_module("exp_data_" + device + "_ELMs")
    
    ne_dict = exp_data.ne(shot,time_inter_ELM_start,time_inter_ELM_end,radial_shift_kin,ELM_frequency,time_resolution,sim_time,exp_time_shift)
    
    print('Electron density data for device ' + device + ' for shot ' + str(shot) + ' loaded.')
    
    return ne_dict
        

# LOAD ELECTRON TEMPERATURE

def Te(device,shot,time_inter_ELM_start,time_inter_ELM_end,radial_shift_kin,ELM_frequency,time_resolution,sim_time,exp_time_shift):

    exp_data = importlib.import_module("exp_data_" + device + "_ELMs")
    
    Te_dict = exp_data.Te(shot,time_inter_ELM_start,time_inter_ELM_end,radial_shift_kin,ELM_frequency,time_resolution,sim_time,exp_time_shift)
    
    print('Electron temperature data for device ' + device + ' for shot ' + str(shot) + ' loaded.')
    
    return Te_dict


# LOAD ION TEMPERATURE

def Ti(device,shot,time_window_data,shotfile_core='CEZ',shotfile_edge='CMZ',exp='augd'):
    
    exp_data = importlib.import_module("exp_data_" + device + "_ELMs")
    
    Ti_dict = exp_data.Ti(shot,time_window_data,shotfile_core=shotfile_core,shotfile_edge=shotfile_edge,exp=exp)
    
    print('Ion temperature data for device ' + device + ' for shot ' + str(shot) + ' loaded.')
    
    return Ti_dict


# LOAD NEUTRAL MAIN GAS DENSITY

def n0(device,shot):
    
    exp_data = importlib.import_module("exp_data_" + device + "_ELMs")
    
    n0_dict = exp_data.n0(shot)
    
    print('Neutral gas density data for device ' + device + ' for shot ' + str(shot) + ' loaded.')
    
    return n0_dict


# LOAD CORE PLASMA IMPURITY DENSITIES

def nimp_core(device,shot_imp,imp,shotfile_imp,time_window,exp='augd',scaling_factor_core=1.0):
    
    exp_data = importlib.import_module("exp_data_" + device + "_ELMs")
    
    nimp_core_dict = exp_data.nimp_core(shot_imp,imp,shotfile_imp,time_window,exp=exp,scaling_factor_core=scaling_factor_core)
    
    print('Core impurity ion density data for device ' + device + ' for shot ' + str(shot_imp) + ' loaded.')
    
    return nimp_core_dict


# LOAD EDGE PLASMA IMPURITY DENSITIES

def nimp_edge(device,shot,time_inter_ELM_start,time_inter_ELM_end,scaling_factor_edge,ELM_frequency,time_resolution,sim_time,exp_time_shift):

    exp_data = importlib.import_module("exp_data_" + device + "_ELMs")
    
    nimp_edge_dict = exp_data.nimp_edge(shot,time_inter_ELM_start,time_inter_ELM_end,scaling_factor_edge,ELM_frequency,time_resolution,sim_time,exp_time_shift)
    
    print('Edge impurity ion density data for device ' + device + ' for shot ' + str(shot) + ' loaded.')
    
    return nimp_edge_dict


# WRITE OUTPUT TO DICTIONARY

def exp_data(device,
             shot_equ,
             shotfile_equ,
             time_equ,
             exp_equ,
             main_species_concentration,
             shot_kin,
             time_inter_ELM_start_kin,
             time_inter_ELM_end_kin,
             radial_shift_kin,
             time_window_data_Ti,
             shotfile_core_Ti,
             shotfile_edge_Ti,
             exp_Ti,
             shot_n0,
             shot_imp_core,
             imp,
             shotfile_imp_core,
             time_window_imp_core,
             exp_imp_core,
             scaling_factor_core,
             shot_imp_edge,
             time_inter_ELM_start_imp,
             time_inter_ELM_end_imp,
             scaling_factor_edge,
             ELM_frequency,
             time_resolution,
             sim_time,
             exp_time_shift):
    
    equilibrium_file = equilibrium(device,shot_equ,shotfile_equ,time_equ,exp_equ)
        
    ne_dict = ne(device,shot_kin,time_inter_ELM_start_kin,time_inter_ELM_end_kin,radial_shift_kin,ELM_frequency,time_resolution,sim_time,exp_time_shift)
    
    Te_dict = Te(device,shot_kin,time_inter_ELM_start_kin,time_inter_ELM_end_kin,radial_shift_kin,ELM_frequency,time_resolution,sim_time,exp_time_shift)        
                 
    Ti_dict = Ti(device,shot_kin,time_window_data_Ti,shotfile_core_Ti,shotfile_edge_Ti,exp_Ti)
    
    n0_dict = n0(device,shot_n0)
    
    nimp_core_dict = nimp_core(device,shot_imp_core,imp,shotfile_imp_core,time_window_imp_core,exp_imp_core,scaling_factor_core)
    
    nimp_edge_dict = nimp_edge(device,shot_imp_edge,time_inter_ELM_start_imp,time_inter_ELM_end_imp,scaling_factor_edge,ELM_frequency,time_resolution,sim_time,exp_time_shift)
    
    exp_data = {
            "device": device,
            "equilibrium": equilibrium_file,
            "kinetic_profiles_inter_ELM": {
                "rhop_ne": np.array(ne_dict['rho_inter_ELM']),
                "ne": np.array(ne_dict['ne_inter_ELM'])*1e-6,
                "rhop_Te": np.array(Te_dict['rho_inter_ELM']),
                "Te":  np.array(Te_dict['Te_inter_ELM']), 
                "rhop_Ti": np.array(Ti_dict['rho']),
                "Ti":  np.array(Ti_dict['Ti_median']),
                "rhop_n0": np.array(n0_dict['rho']),
                "n0": np.array(n0_dict['n0']),
            },
            "kinetic_profiles_ELM_cycle": {
                "time_ne": np.array(ne_dict['time_ELM_cycle']),
                "rhop_ne": np.array(ne_dict['rho_ELM_cycle']),
                "ne": np.array(ne_dict['ne_ELM_cycle'])*1e-6,
                "time_Te": np.array(Te_dict['time_ELM_cycle']),
                "rhop_Te": np.array(Te_dict['rho_ELM_cycle']),
                "Te": np.array(Te_dict['Te_ELM_cycle']),
            },
        }
    
    exp_data.update({
        "D_density_plasma": {
            "rhop": exp_data["kinetic_profiles_inter_ELM"]["rhop_ne"],
            "n_D": exp_data["kinetic_profiles_inter_ELM"]["ne"] * main_species_concentration ,
            },
    })
    
    exp_data.update({
        f"{imp}_density_plasma": {
            "rhop": nimp_core_dict['rho_imp'],
            f"n_{imp}": nimp_core_dict['n_imp']*1e-6 ,
            },
    })
    
    exp_data.update({
        f"{imp}_density_edge_inter_ELM": {
            "rhop_data":nimp_edge_dict['rho_inter_ELM_data'],
            f"n_{imp}_data": nimp_edge_dict['nimp_inter_ELM_data']*1e-6 ,
            "rhop_median":nimp_edge_dict['rho_inter_ELM_median'],
            f"n_{imp}_median": nimp_edge_dict['nimp_inter_ELM_median']*1e-6 ,
            },
    })
    
    exp_data.update({
        f"{imp}_density_edge_ELM_cycle": {
            "time":nimp_edge_dict['time_ELM_cycle'],
            "rhop":nimp_edge_dict['rho_ELM_cycle'],
            f"n_{imp}": nimp_edge_dict['nimp_ELM_cycle']*1e-6 ,
            },
    })
    
    exp_data.update({
        f"{imp}_density_edge_ELM_cycle": {
            "time":nimp_edge_dict['time_ELM_cycle'],
            "rhop":nimp_edge_dict['rho_ELM_cycle'],
            f"n_{imp}": nimp_edge_dict['nimp_ELM_cycle']*1e-6 ,
            },
    })
    
    return exp_data
