import os
import aug_sfutils as sf
import numpy as np
import h5py
import pandas as pd
import scipy
import scipy.io
from scipy.signal import savgol_filter
from omfit_classes import omfit_eqdsk, omfit_gapy


def equilibrium(shot,shotfile='EQI',time=0.,exp='augd'):
    
    equilibrium = omfit_eqdsk.OMFITgeqdsk(filename="dummy_filename")
    equilibrium.from_aug_sfutils(shot=shot, time=time, eq_shotfile=shotfile)
    os.remove(equilibrium.filename)
    
    return equilibrium


def ne(shot,time_inter_ELM_start,time_inter_ELM_end,radial_shift_kin,ELM_frequency,time_resolution,sim_time,exp_time_shift):
    
    # Extract data from .nc file
    
    file = os.path.dirname(os.path.realpath(__file__)) + os.sep + f'aug_{shot}_ELM_cycle.nc'
    
    with h5py.File(file, 'r') as f:
        time_IDA = f['IDA/time'][:]
        rho_IDA = f['IDA/rho'][:]
        ne_IDA = f['IDA/ne'][:]

    # Find time indices
    
    time_start_index_IDA = np.argmin(np.abs(time_inter_ELM_start - time_IDA))
    time_end_index_IDA = np.argmin(np.abs(time_inter_ELM_end - time_IDA))

    # Extract relevant slices and compute medians
    
    rho_inter_ELM_data = rho_IDA[time_start_index_IDA:time_end_index_IDA, :]
    rho_inter_ELM_median = np.mean(rho_inter_ELM_data, axis=0)
    
    ne_inter_ELM_data = ne_IDA[time_start_index_IDA:time_end_index_IDA, :]
    ne_inter_ELM_median = np.mean(ne_inter_ELM_data, axis=0)
    
    # Compute the entire ELM cycle
    
    time_bins = np.linspace(time_resolution, 1/ELM_frequency, int(1/ELM_frequency/time_resolution))
    
    left_edges = np.concatenate(([0], time_bins[:-1]))
    
    ne_cycle = np.array([
        np.mean(ne_IDA[(time_IDA+exp_time_shift >= t0) & (time_IDA+exp_time_shift < t1)], axis=0)
        for t0, t1 in zip(left_edges, time_bins)
    ])
    
    cycles = int(sim_time/(1/ELM_frequency))
    
    full_time_bins = np.linspace(time_resolution, sim_time, num=int(sim_time/time_resolution))
    
    full_ne_cycle = np.tile(ne_cycle, (cycles, 1))
    
    ne_dict = {}
    
    ne_dict['rho_inter_ELM'] = rho_inter_ELM_median + radial_shift_kin
    ne_dict['ne_inter_ELM'] = ne_inter_ELM_median
    
    ne_dict['time_ELM_cycle'] = full_time_bins
    ne_dict['rho_ELM_cycle'] = rho_inter_ELM_median + radial_shift_kin
    ne_dict['ne_ELM_cycle'] = full_ne_cycle
    
    return ne_dict

    
def Te(shot,time_inter_ELM_start,time_inter_ELM_end,radial_shift_kin,ELM_frequency,time_resolution,sim_time,exp_time_shift):
    
    # Extract data from .nc file
    
    file = os.path.dirname(os.path.realpath(__file__)) + os.sep + f'aug_{shot}_ELM_cycle.nc' 
    
    with h5py.File(file, 'r') as f:
        time_IDA = f['IDA/time'][:]
        rho_IDA = f['IDA/rho'][:]
        Te_IDA = f['IDA/Te'][:]

    # Find time indices
    
    time_start_index_IDA = np.argmin(np.abs(time_inter_ELM_start - time_IDA))
    time_end_index_IDA = np.argmin(np.abs(time_inter_ELM_end - time_IDA))

    # Extract relevant slices and compute medians
    
    rho_inter_ELM_data = rho_IDA[time_start_index_IDA:time_end_index_IDA, :]
    rho_inter_ELM_median = np.mean(rho_inter_ELM_data, axis=0)
    
    Te_inter_ELM_data = Te_IDA[time_start_index_IDA:time_end_index_IDA, :]
    Te_inter_ELM_median = np.mean(Te_inter_ELM_data, axis=0)
    
    # Compute the entire ELM cycle
    
    time_bins = np.linspace(time_resolution, 1/ELM_frequency, int(1/ELM_frequency/time_resolution))
    
    left_edges = np.concatenate(([0], time_bins[:-1]))
    
    Te_cycle = np.array([
        np.mean(Te_IDA[(time_IDA+exp_time_shift >= t0) & (time_IDA+exp_time_shift < t1)], axis=0)
        for t0, t1 in zip(left_edges, time_bins)
    ])
    
    cycles = int(sim_time/(1/ELM_frequency))
    
    full_time_bins = np.linspace(time_resolution, sim_time, num=int(sim_time/time_resolution))
    
    full_Te_cycle = np.tile(Te_cycle, (cycles, 1))
    
    Te_dict = {}
    
    Te_dict['rho_inter_ELM'] = rho_inter_ELM_median + radial_shift_kin
    Te_dict['Te_inter_ELM'] = Te_inter_ELM_median
    
    Te_dict['time_ELM_cycle'] = full_time_bins
    Te_dict['rho_ELM_cycle'] = rho_inter_ELM_median + radial_shift_kin
    Te_dict['Te_ELM_cycle'] = full_Te_cycle
    
    return Te_dict


def Ti(shot,time_window_data,shotfile_core='CEZ',shotfile_edge='CMZ',exp='augd'):
    
    # Extract data
    
    EQU = sf.EQU(shot,diag='EQH')
    
    CEZ = sf.SFREAD(shotfile_core, shot, exp=exp)
    CMZ = sf.SFREAD(shotfile_edge, shot, exp=exp)
    
    time_CEZ = CEZ.getobject('time') 
    time_CMZ = CMZ.getobject('time') 
    
    data_CEZ = CEZ.getobject('Ti_c') 
    data_uncertainty_CEZ = CEZ.getobject('err_Ti_c')
    
    time_start_data = time_window_data[0]
    time_end_data = time_window_data[1]
    
    time_CEZ_interval = [time_start_data]
    k = 0
    data_CEZ_interval= []
    while time_CEZ_interval[k] < time_end_data - 0.00000001:
        time_CEZ_interval.append(time_CEZ_interval[k] + 0.02)
        time_interval_start_index_CEZ = np.argmin(np.abs(time_CEZ_interval[k] - time_CEZ))
        time_interval_end_index_CEZ = np.argmin(np.abs(time_CEZ_interval[k+1] - time_CEZ))
        data_CEZ_delta_interval_k = []
        for j in range(time_interval_end_index_CEZ - time_interval_start_index_CEZ + 1):
            data_CEZ_delta_interval_k.append(data_CEZ[j + time_interval_start_index_CEZ - 1, :])
        data_CEZ_interval_k = np.median(np.array(data_CEZ_delta_interval_k), axis=0)
        data_CEZ_interval.append(data_CEZ_interval_k.tolist()) 
        k += 1
    time_CEZ_interval = time_CEZ_interval[1:]
    data_CEZ_interval = np.array(data_CEZ_interval)
    
    data_CMZ = CMZ.getobject('Ti_c') 
    data_uncertainty_CMZ = CMZ.getobject('err_Ti_c')
    
    time_CMZ_interval = [time_start_data]
    k = 0
    data_CMZ_interval= []
    while time_CMZ_interval[k] < time_end_data - 0.00000001:
        time_CMZ_interval.append(time_CMZ_interval[k] + 0.02)
        time_interval_start_index_CMZ = np.argmin(np.abs(time_CMZ_interval[k] - time_CMZ))
        time_interval_end_index_CMZ = np.argmin(np.abs(time_CMZ_interval[k+1] - time_CMZ))
        data_CMZ_delta_interval_k = []
        for j in range(time_interval_end_index_CMZ - time_interval_start_index_CMZ + 1):
            data_CMZ_delta_interval_k.append(data_CMZ[j + time_interval_start_index_CMZ - 1, :])
        data_CMZ_interval_k = np.median(np.array(data_CMZ_delta_interval_k), axis=0)
        data_CMZ_interval.append(data_CMZ_interval_k.tolist()) 
        k += 1
    time_CMZ_interval = time_CMZ_interval[1:]
    data_CMZ_interval = np.array(data_CMZ_interval)
    
    # Calculate coordinates
    
    R_CEZ = CEZ.getobject('R_time') 
    z_CEZ = CEZ.getobject('z_time') 
    
    R_CMZ = CMZ.getobject('R') 
    z_CMZ = CMZ.getobject('z') 
    
    temp = np.where(data_CEZ_interval[0] == 0)[0]

    if len(temp) > 0:
        temp = temp[0]
    
        data_CEZ_interval = data_CEZ_interval[:, :temp]
    
        R_CEZ = R_CEZ[:temp, :]
        z_CEZ = z_CEZ[:temp, :]
        
    x_CEZ = []
    for i in range(len(time_CEZ_interval) - 1):
        x_CEZ.append(sf.rz2rho(EQU, R_CEZ[:,i], z_CEZ[:,i], t_in=time_CEZ_interval[i], coord_out='rho_pol', extrapolate=False).T)
    x_CEZ = np.array(x_CEZ)[:,:,0]
    x_CEZ = np.column_stack([x_CEZ.T, x_CEZ.T[:,-1]])
        
    temp = np.where(data_CMZ_interval[0] == 0)[0]

    if len(temp) > 0:
        temp = temp[0]
    
        data_CMZ_interval = data_CMZ_interval[:, :temp]
    
        R_CMZ = R_CMZ[:temp, :]
        z_CMZ = z_CMZ[:temp, :]
        
    x_CMZ = []
    for i in range(len(time_CMZ_interval) - 1):
        x_CMZ.append(sf.rz2rho(EQU, R_CMZ.T, z_CMZ.T, t_in=time_CMZ_interval[i], coord_out='rho_pol', extrapolate=False).T)
    x_CMZ = np.array(x_CMZ)[:,:,0]
    x_CMZ = np.column_stack([x_CMZ.T, x_CMZ.T[:,-1]])
    
    # Spline fit
    
    x_CEZ_fit = np.zeros(data_CEZ_interval.shape[1])
    data_CEZ_fit = np.zeros(data_CEZ_interval.shape[1])
    
    for i in range(data_CEZ_interval.shape[1]):
        x_CEZ_fit[i] = np.mean(x_CEZ[i, :])
        data_CEZ_fit[i] = np.mean(data_CEZ_interval[:, i])
    
    x_CMZ_fit = np.zeros(data_CMZ_interval.shape[1])
    data_CMZ_fit = np.zeros(data_CMZ_interval.shape[1])
    
    for i in range(data_CMZ_interval.shape[1]):
        x_CMZ_fit[i] = np.mean(x_CMZ[i, :])
        data_CMZ_fit[i] = np.mean(data_CMZ_interval[:, i])
    
    x_final = np.concatenate((x_CEZ_fit, x_CMZ_fit))
    
    data_final = np.concatenate((data_CEZ_fit, data_CMZ_fit))
    
    p = np.polyfit(x_final, data_final, 4)
    
    x_plot = np.linspace(0, 1.2, 200)
    
    data_plot = np.polyval(p, x_plot)
    
    data_plot = np.maximum(data_plot, 1)
    
    # Return dictionary
    
    rho = x_plot
    Ti_median = data_plot
    
    rho_core = x_CEZ
    Ti_core = data_CEZ_interval
    
    rho_edge = x_CMZ
    Ti_edge = data_CMZ_interval
    
    Ti_dict = {}
    
    Ti_dict['rho'] = rho
    Ti_dict['Ti_median'] = Ti_median
    
    Ti_dict['rho_core'] = rho_core
    Ti_dict['Ti_core'] = Ti_core
    
    Ti_dict['rho_edge'] = rho_edge
    Ti_dict['Ti_edge'] = Ti_edge

    return Ti_dict


def n0(shot):
    
    n0_pickle_file = os.path.dirname(os.path.realpath(__file__)) + os.sep + f'aug_n0.pkl'    
    n0_pickle = pd.read_pickle(n0_pickle_file)
        
    rho_n0 = n0_pickle["rhop"]
    n0 = n0_pickle["n0"]
    
    # Return dictionary
    
    n0_dict = {}
    
    n0_dict['rho'] = rho_n0
    n0_dict['n0'] = n0
    
    return n0_dict

    
def nimp_core(shot_imp,imp,shotfile_imp,time_window,exp='augd',scaling_factor_core=1.0):
    
    # Open shotfiles
    
    EQU = sf.EQU(shot_imp,diag='EQH') 
    SF = sf.SFREAD(shotfile_imp, shot_imp, exp=exp)    
    
    time_start = time_window[0]
    time_end = time_window[1]
    
    # Timebase

    time_imp = SF.getobject('time') 
    
    # Areabase
        
    R_imp = SF.getobject('R').T 
    z_imp = SF.getobject('z').T
    
    # Density
    
    if imp == 'He':
        n_imp = SF.getobject('nimp_plc') 
    else:
        n_imp = SF.getobject('nimp_plc')   
        
    # Remove empty signals
    
    for j in reversed(range(len(R_imp[0,:]))):
        if R_imp[0,j] == 0. or n_imp[0,j] == 0.:
            R_imp = np.delete(R_imp, j, axis=1)
            z_imp = np.delete(z_imp, j, axis=1)
            n_imp = np.delete(n_imp, j, axis=1)
        
    # Calculate flux surfaces
    
    R_imp_flux_surfaces = np.mean(R_imp, axis=0)
    z_imp_flux_surfaces = np.mean(z_imp, axis=0)
    
    rho_imp = sf.rz2rho(EQU, R_imp_flux_surfaces, z_imp_flux_surfaces,
                             t_in=time_imp[0], coord_out='rho_pol', extrapolate=False)
    index_first_surface = np.argmin(rho_imp)
    n_imp = n_imp[:,index_first_surface:]
    rho_imp = rho_imp[:,index_first_surface:]
        
    # Remove edge signals
    
    try:
    
        for j in reversed(range(len(rho_imp[0,:]))):
            if rho_imp[0,j] > 0.9:
                rho_imp = np.delete(rho_imp, j, axis=1)
                n_imp = np.delete(n_imp, j, axis=1)
                    
    except:
        
        pass
        
    # Mean values of background impurity densities
        
    index_time_start = np.argmin(np.abs(time_start - time_imp))
    index_time_end = np.argmin(np.abs(time_end - time_imp))
    n_imp_median = np.median(n_imp[index_time_start:index_time_end], axis=0)
    
    # Return dictionary
    
    nimp_core = {}

    nimp_core['rho_imp'] = rho_imp
    nimp_core['n_imp'] = n_imp_median*scaling_factor_core
    
    return nimp_core
        

def nimp_edge(shot,time_inter_ELM_start,time_inter_ELM_end,scaling_factor_edge,ELM_frequency,time_resolution,sim_time,exp_time_shift):
    
    # Extract data from .nc file
    
    file = os.path.dirname(os.path.realpath(__file__)) + os.sep + f'aug_{shot}_ELM_cycle.nc' 
    
    with h5py.File(file, 'r') as f:
        time = f['CPZ/time'][:]
        rho = f['CPZ/rho'][:]
        inte = f['CPZ/inte'][:]

    inte[:,0] = inte[:,0]*0.7

    # Find time indices
    
    time_start_index = np.argmin(np.abs(time_inter_ELM_start - time))
    time_end_index = np.argmin(np.abs(time_inter_ELM_end - time))

    # Extract relevant slices and compute medians
    
    rho_inter_ELM_data = rho[time_start_index:time_end_index, :]
    rho_inter_ELM_median = np.mean(rho_inter_ELM_data, axis=0)
    
    inte_inter_ELM_data = inte[time_start_index:time_end_index, :]
    inte_inter_ELM_median = np.mean(inte_inter_ELM_data, axis=0)
    
    # Compute the entire ELM cycle
    
    time_bins = np.linspace(time_resolution, 1/ELM_frequency, int(1/ELM_frequency/time_resolution))
    
    left_edges = np.concatenate(([0], time_bins[:-1]))
    
    rho_cycle = np.array([
        np.mean(rho[(time+exp_time_shift >= t0) & (time+exp_time_shift < t1)], axis=0)
        for t0, t1 in zip(left_edges, time_bins)
    ])
    
    nimp_edge_cycle = np.array([
        np.mean(inte[(time+exp_time_shift >= t0) & (time+exp_time_shift < t1)], axis=0)
        for t0, t1 in zip(left_edges, time_bins)
    ])
    
    cycles = int(sim_time/(1/ELM_frequency))
    
    full_time_bins = np.linspace(time_resolution, sim_time, num=int(sim_time/time_resolution))
    
    full_rho_cycle = np.tile(rho_cycle, (cycles, 1))
    
    full_nimp_edge_cycle = np.tile(nimp_edge_cycle, (cycles, 1))
    
    nimp_edge = {}
    
    nimp_edge['rho_inter_ELM_data'] = rho_inter_ELM_data
    nimp_edge['nimp_inter_ELM_data'] = inte_inter_ELM_data*scaling_factor_edge
    
    nimp_edge['rho_inter_ELM_median'] = rho_inter_ELM_median
    nimp_edge['nimp_inter_ELM_median'] = inte_inter_ELM_median*scaling_factor_edge
    
    nimp_edge['time_ELM_cycle'] = full_time_bins
    nimp_edge['rho_ELM_cycle'] = full_rho_cycle
    nimp_edge['nimp_ELM_cycle'] = full_nimp_edge_cycle*scaling_factor_edge
    
    return nimp_edge
