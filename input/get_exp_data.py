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
    
    exp_data = importlib.import_module("exp_data_" + device)
    
    equilibrium_file = exp_data.equilibrium(shot,shotfile=shotfile,time=time,exp=exp)
    
    print('Equilibrium data for device ' + device + ' for shot ' + str(shot) + ' loaded.')
        
    return equilibrium_file


# LOAD ELECTRON DENSITY

def ne(device,shot,time_window,time_window_data,interval_elm,time_points_elm,shotfile='IDA',exp='augd',shotfile_elm='ELM',exp_elm='augd'):

    exp_data = importlib.import_module("exp_data_" + device)
    
    ne_dict = exp_data.ne(shot,time_window,time_window_data,interval_elm,time_points_elm,shotfile=shotfile,exp=exp,shotfile_elm=shotfile_elm,exp_elm=exp_elm)
    
    print('Electron density data for device ' + device + ' for shot ' + str(shot) + ' loaded.')
    
    return ne_dict
        

# LOAD ELECTRON TEMPERATURE

def Te(device,shot,time_window,time_window_data,shotfile='IDA',exp='augd'):

    exp_data = importlib.import_module("exp_data_" + device)
    
    Te_dict = exp_data.Te(shot,time_window,time_window_data,shotfile=shotfile,exp=exp)
    
    print('Electron temperature data for device ' + device + ' for shot ' + str(shot) + ' loaded.')
    
    return Te_dict


# LOAD ION TEMPERATURE

def Ti(device,shot,time_window_data,shotfile_core='CEZ',shotfile_edge='CMZ',exp='augd'):
    
    exp_data = importlib.import_module("exp_data_" + device)
    
    Ti_dict = exp_data.Ti(shot,time_window_data,shotfile_core=shotfile_core,shotfile_edge=shotfile_edge,exp=exp)
    
    print('Ion temperature data for device ' + device + ' for shot ' + str(shot) + ' loaded.')
    
    return Ti_dict


# LOAD NEUTRAL MAIN GAS DENSITY

def n0(device,shot):
    
    exp_data = importlib.import_module("exp_data_" + device)
    
    n0_dict = exp_data.n0(shot)
    
    print('Neutral gas density data for device ' + device + ' for shot ' + str(shot) + ' loaded.')
    
    return n0_dict


# LOAD CORE PLASMA IMPURITY DENSITIES

def nimp_core(device,shot_imp_main,imp_main,shotfile_imp_main,
              shots_imp_background,imp_background,shotfiles_imp_background,
              time_windows_background,exp='augd'):
    
    exp_data = importlib.import_module("exp_data_" + device)
    
    nimp_core_dict = exp_data.nimp_core(shot_imp_main,imp_main,shotfile_imp_main,
                                        shots_imp_background,imp_background,shotfiles_imp_background,
                                        time_windows_background,exp=exp)
    
    print('Impurity ion density data for device ' + device + ' for shot ' + str(shot_imp_main) + ' loaded.')
    
    return nimp_core_dict


# LOAD NEUTRAL GAS IMPURITY DENSITY

def nimp_gas(device,shot,imp,shotfile_lines,shotfile_currents,
             shotfile_total_pressures,gauge_div,gauge_pump,impurity_pressure_drop,main_species_pressure_drop,
             time_window_pre_puff,time_window_correction,exp='augd'):
    
    exp_data = importlib.import_module("exp_data_" + device)
    
    nimp_gas_dict = exp_data.nimp_gas(shot,imp,shotfile_lines,shotfile_currents,
                 shotfile_total_pressures,gauge_div,gauge_pump,impurity_pressure_drop,main_species_pressure_drop,
                 time_window_pre_puff,time_window_correction,exp=exp)
    
    print('Impurity gas density data for device ' + device + ' for shot ' + str(shot) + ' loaded.')
    
    return nimp_gas_dict
    

# WRITE OUTPUT TO DICTIONARY

def exp_data(device,
             shot_equ,
             shotfile_equ,
             time_equ,
             exp_equ,
             main_species_concentration,
             shot_kin,
             time_window_ne,
             time_window_data_ne,
             shotfile_ne,
             exp_ne,
             time_window_Te,
             time_window_data_Te,
             shotfile_Te,
             exp_Te,
             time_window_data_Ti,
             shotfile_core_Ti,
             shotfile_edge_Ti,
             exp_Ti,
             shot_n0,
             shot_imp_main,
             imp_main,
             shotfile_imp_main,
             shots_imp_background,
             imp_background,
             shotfiles_imp_background,
             time_windows_background,
             exp_imp,
             shot_gas,
             shotfile_lines,
             shotfile_currents,
             shotfile_total_pressures,
             gauge_div,
             gauge_pump,
             impurity_pressure_drop,
             main_species_pressure_drop,
             time_window_pre_puff,
             time_window_correction,
             exp_imp_gas,
             shotfile_elm,
             interval_elm,
             time_points_elm,
             exp_elm):
    
    equilibrium_file = equilibrium(device,shot_equ,shotfile_equ,time_equ,exp_equ)
        
    ne_dict = ne(device,shot_kin,time_window_ne,time_window_data_ne,interval_elm,time_points_elm,shotfile_ne,exp_ne,shotfile_elm,exp_elm)
    
    Te_dict = Te(device,shot_kin,time_window_Te,time_window_data_Te,shotfile_Te,exp_Te)
    
    Ti_dict = Ti(device,shot_kin,time_window_data_Ti,shotfile_core_Ti,shotfile_edge_Ti,exp_Ti)
    
    n0_dict = n0(device,shot_n0)
    
    nimp_core_dict = nimp_core(device,shot_imp_main,imp_main,shotfile_imp_main,
                  shots_imp_background,imp_background,shotfiles_imp_background,
                  time_windows_background,exp_imp)
    
    nimp_gas_dict = nimp_gas(device,shot_gas,imp_main,shotfile_lines,shotfile_currents,
                 shotfile_total_pressures,gauge_div,gauge_pump,impurity_pressure_drop,main_species_pressure_drop,
                 time_window_pre_puff,time_window_correction,exp_imp_gas)
    
    # Extrapolate main impurity density towards the separatrix
    
    rho_ne = ne_dict['rho']
    ne_profiles = ne_dict['ne_median']
    rho_imp_main = nimp_core_dict['rho_imp_main']
    n_imp_main_profile = nimp_core_dict['n_imp_main']
    n_imp_main_unc = nimp_core_dict['n_imp_main_err']
    
    idx = np.argmin(np.abs(np.asarray(rho_ne) - rho_ne[rho_ne > rho_imp_main[0,-1]].min()))
    idx_IDA_last_point = np.argmin(np.abs(np.asarray(rho_ne) - rho_imp_main[0,-1]))
    rho_imp_main_extrap = np.append(rho_imp_main,rho_ne[idx:-1])
    n_imp_main_profile_extrap = np.zeros((n_imp_main_profile.shape[0],len(rho_imp_main_extrap)))
    n_imp_main_unc_extrap = np.zeros((n_imp_main_unc.shape[0],len(rho_imp_main_extrap)))

    ne_base = np.median(ne_profiles,axis=1)

    for i in range(0,n_imp_main_profile.shape[0]):
        n_imp_main_profile_extrap[i,0:n_imp_main_profile.shape[1]] = n_imp_main_profile[i,:]
        for j in range(n_imp_main_profile.shape[1],len(rho_imp_main_extrap)):
            n_imp_main_profile_extrap[i,j] = n_imp_main_profile[i,n_imp_main_profile.shape[1]-1] / (ne_base[idx_IDA_last_point] / ne_base[idx-n_imp_main_profile.shape[1]+j])
        n_imp_main_unc_extrap[i,0:n_imp_main_unc.shape[1]] = n_imp_main_unc[i,:]
        for j in range(n_imp_main_unc.shape[1],len(rho_imp_main_extrap)):
            n_imp_main_unc_extrap[i,j] = ( n_imp_main_unc[i,n_imp_main_unc.shape[1]-1] / n_imp_main_profile[i,n_imp_main_profile.shape[1]-1] ) * n_imp_main_profile_extrap[i,j]
    
    
    exp_data = {
            "device": device,
            "equilibrium": equilibrium_file,
            "kinetic_profiles": {
                "rhop_ne": np.array(ne_dict['rho']),
                "ne":  np.array((np.median(ne_dict['ne_median'],axis=1))).T*1e-6,
                "ne_unc":  np.array((np.median(ne_dict['ne_median_uncertainty'],axis=1)))*1e-6,
                "rhop_Te": np.array(Te_dict['rho']),
                "Te":  np.array(Te_dict['Te_median']).T, 
                "Te_unc":  np.array(Te_dict['Te_median_uncertainty']),
                "rhop_Ti": np.array(Ti_dict['rho']),
                "Ti":  np.array(Ti_dict['Ti_median']),
                "rhop_n0": np.array(n0_dict['rho']),
                "n0": np.array(n0_dict['n0']),
            },
            "kinetic_profiles_ELM": {
                "time_ELM": ne_dict['time_window'],
                "rhop": ne_dict['rho'],
                "ne": ne_dict['ne_median']*1e-6 ,
                "ne_unc": ne_dict['ne_median_uncertainty']*1e-6 ,
            },
            "electron_density_data": {
                "index_core_TS": ne_dict['index_core_TS'],
                "rho_core_TS": ne_dict['rho_core_TS'],
                "ne_core_TS": ne_dict['ne_core_TS']*1e-6,
                "index_edge_TS": ne_dict['index_edge_TS'],
                "rho_edge_TS": ne_dict['rho_edge_TS'],
                "ne_edge_TS": ne_dict['ne_edge_TS']*1e-6,
                "rho_LIB": ne_dict['rho_LIB'],
                "ne_LIB": ne_dict['ne_LIB']*1e-6,
            },
            "electron_temperature_data": {
                "index_core_TS": Te_dict['index_core_TS'],
                "rho_core_TS": Te_dict['rho_core_TS'],
                "Te_core_TS": Te_dict['Te_core_TS'],
                "index_edge_TS": Te_dict['index_edge_TS'],
                "rho_edge_TS": Te_dict['rho_edge_TS'],
                "Te_edge_TS": Te_dict['Te_edge_TS'],
                "index_ECE": Te_dict['index_ECE'],
                "rho_ECE": Te_dict['rho_ECE'],
                "Te_ECE": Te_dict['Te_ECE'],
            },
            "ion_temperature_data": {
                "rho_core": Ti_dict['rho_core'],
                "Ti_core": Ti_dict['Ti_core'],
                "rho_edge": Ti_dict['rho_edge'],
                "Ti_edge": Ti_dict['Ti_edge'],
            },
        }
    
    exp_data.update({
        "D_density_plasma": {
            "rhop": exp_data["kinetic_profiles"]["rhop_ne"],
            "n_D": exp_data["kinetic_profiles"]["ne"] * main_species_concentration ,
            "n_D_unc": exp_data["kinetic_profiles"]["ne_unc"] ,
            },
        "D_density_plasma_ELM": {
            "time_ELM": exp_data["kinetic_profiles_ELM"]["time_ELM"],
            "rhop": exp_data["kinetic_profiles_ELM"]["rhop"],
            "n_D": exp_data["kinetic_profiles_ELM"]["ne"] * main_species_concentration ,
            "n_D_unc": exp_data["kinetic_profiles_ELM"]["ne_unc"] ,
            },
    })
    
    exp_data.update({
        f"{imp_main}_density_plasma": {
            "time": nimp_core_dict['time_imp_main'],
            "rhop": nimp_core_dict['rho_imp_main'],
            f"n_{imp_main}": nimp_core_dict['n_imp_main']*1e-6 ,
            f"n_{imp_main}_unc": nimp_core_dict['n_imp_main_err']*1e-6 ,
            "rhop_extrap": rho_imp_main_extrap,
            f"n_{imp_main}_extrap": n_imp_main_profile_extrap*1e-6 ,
            f"n_{imp_main}_unc_extrap": n_imp_main_unc_extrap*1e-6 ,
            },
    })
    
    for i in range(0,len(imp_background)):
        exp_data.update({
            f"{imp_background[i]}_density_plasma": {
                "rhop": nimp_core_dict['rho_imp_background'][i],
                f"n_{imp_background[i]}": nimp_core_dict['n_imp_background'][i]*1e-6 ,
                f"n_{imp_background[i]}_unc": nimp_core_dict['n_imp_background_err'][i]*1e-6 ,
                },
        })
        
    
    exp_data.update({
        f"{imp_main}_partial_pressures": {
            "time": nimp_gas_dict['time_imp_gas'],
            "n_div": nimp_gas_dict['n_imp_gas_div']*1e-6 ,
            "n_pump": nimp_gas_dict['n_imp_gas_pump']*1e-6 ,
            },
    })
    
    
    return exp_data
