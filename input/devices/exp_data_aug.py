import os
import aug_sfutils as sf
import numpy as np
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


def ne(shot,time_window,time_window_data,interval_elm,time_points_elm,shotfile='IDA',exp='augd',shotfile_elm='ELM',exp_elm='augd'):
    
    # Extract data from integrated data analysis
    
    IDA = sf.SFREAD(shotfile, shot, exp=exp)
    IDA_data = sf.SFREAD('IDA', shot, exp='augd')
    
    time = IDA.getobject('time') 
    
    data = IDA.getobject('ne') 
    data_uncertainty = IDA.getobject('ne_unc') 
    
    time_start = time_window[0]
    time_end = time_window[1]
    
    time_start_data = time_window_data[0]
    time_end_data = time_window_data[1]
    
    index_start = np.argmin(np.abs(time - time_start))
    index_end = np.argmin(np.abs(time - time_end))
    
    # Calculate the ELM onset times
    
    ELM = sf.SFREAD(shotfile_elm, shot, exp=exp_elm)
    
    time_ELM = ELM.getobject('t_begELM') 
    
    t_IDA = time[index_start:index_end + 1]
    
    index_ELM_start = np.argmin(np.abs(time_ELM - time_start))
    index_ELM_end = np.argmin(np.abs(time_ELM - time_end))    
    
    zeros = np.zeros(index_ELM_end - index_ELM_start + 1)
    t_begELM = time_ELM[index_ELM_start:index_ELM_end + 1]
    
    for i in range(len(t_begELM) - 1, -1, -1):
        a_i = np.min(np.abs(t_IDA - t_begELM[i]))
        if a_i > 3e-4:
            t_begELM = np.delete(t_begELM, i)
    
    # Extract time windows relative to ELM onset times from IDA
    
    time_windows = [[0 for _ in range(time_points_elm)] for _ in range(len(t_begELM))]
    data_windows = [[[0 for _ in range(time_points_elm)] for _ in range(len(t_begELM))] for _ in range(data.shape[0])]
    data_windows_uncertainty = [[[0 for _ in range(time_points_elm)] for _ in range(len(t_begELM))] for _ in range(data.shape[0])]
    
    data_for_median = [[[0 for _ in range(len(t_begELM))] for _ in range(time_points_elm)] for _ in range(data.shape[0])]
    data_for_median_uncertainty = [[[0 for _ in range(len(t_begELM))] for _ in range(time_points_elm)] for _ in range(data.shape[0])]
    
    med = [[0 for _ in range(time_points_elm)] for _ in range(data.shape[0])]
    med_uncertainty = [[0 for _ in range(time_points_elm)] for _ in range(data.shape[0])]
    
    rho = IDA.getobject('rhop')[:,0]  
    
    ne_median = np.zeros((len(rho), time_points_elm))
    ne_median_uncertainty = np.zeros((len(rho), time_points_elm))
    
    for rho_index in range(data.shape[0]):
        
        for i in range(len(t_begELM)):
            t_start = t_begELM[i] - interval_elm
            index_IDA_start = np.argmin(np.abs(time - t_start))
            index_IDA_end = index_IDA_start + time_points_elm
    
            for j in range(index_IDA_end - index_IDA_start):
                try:
                    time_windows[i][j] = (t_IDA[index_IDA_start + j] - t_IDA[index_IDA_start])
                    data_windows[rho_index][i][j] = (data[rho_index, index_IDA_start + j])
                    data_windows_uncertainty[rho_index][i][j] = (data_uncertainty[rho_index, index_IDA_start + j])
                except IndexError:
                    pass
    
    # Calculate data and median for integrated data analysis
        
        for i in range(len(data_windows[rho_index]) - 3):
            for j in range(len(data_windows[rho_index][i])):
                data_for_median[rho_index][j][i] = data_windows[rho_index][i][j]
                data_for_median_uncertainty[rho_index][j][i] = data_windows_uncertainty[rho_index][i][j]
                
        for i in range(len(data_for_median[rho_index])):
            med[rho_index][i] = np.median(data_for_median[rho_index][i], axis=0, overwrite_input=True)
            med_uncertainty[rho_index][i] = np.median(data_for_median_uncertainty[rho_index][i], axis=0, overwrite_input=True)
        
        for i in range(len(rho)):
            ne_median[i,:] = np.convolve(med[i][:], np.ones(4)/4, mode='same')
            ne_median_uncertainty[i,:] = np.convolve(med_uncertainty[i][:], np.ones(4)/4, mode='same')
    
    ne_median[:,0] = ne_median[:,2]
    ne_median[:,1] = ne_median[:,2]
    ne_median[:,-1] = ne_median[:,-3]
    ne_median[:,-2] = ne_median[:,-3]
    
    ne_median_uncertainty[:,1] = ne_median_uncertainty[:,3]
    ne_median_uncertainty[:,2] = ne_median_uncertainty[:,3]
    ne_median_uncertainty[:,-1] = ne_median_uncertainty[:,-3]
    ne_median_uncertainty[:,-2] = ne_median_uncertainty[:,-3]
    
    time_window = np.array(time_windows[0])-interval_elm
    
    # Smooth data
    
    for j in range(len(t_begELM)):
        ne_median[:,j] = savgol_filter(ne_median[:,j], 11, 1)
        ne_median_uncertainty[:,j] = savgol_filter(ne_median_uncertainty[:,j], 11, 1)
    
    # rho = IDA.getobject('rhop')[:,0]
    
    # ne = data[:, index_start:index_end + 1]
    # ne_unc = data_uncertainty[:, index_start:index_end + 1]
    
    # ne_median = np.zeros(len(rho))
    # ne_median_uncertainty = np.zeros(len(rho))
    
    # for i in range(len(rho)):
    #     ne_median[i] = np.median(ne[i, :])
    #     ne_median_uncertainty[i] = np.median(ne_unc[i, :])
        
    # ne_median = ne_median.reshape(-1, 1)
    # ne_median_uncertainty = ne_median_uncertainty.reshape(-1, 1)
    
    # time_window = np.linspace(-interval_elm, -interval_elm + (time_points_elm - 1) * 0.0001, time_points_elm)
    
    # for i in range(1, time_points_elm):
    #     ne_median = np.hstack((ne_median, ne_median[:, :1]))
    #     ne_median_uncertainty = np.hstack((ne_median_uncertainty, ne_median_uncertainty[:, :1]))
    
    # Extract data from diagnostics

    LIN = sf.SFREAD('LIN', shot, exp='augd')
    
    time_LIB = LIN.getobject('time') 
    
    index_core_TS = IDA_data.getobject('Ntscprof')
    index_edge_TS = IDA_data.getobject('Ntseprof')
    data_core_TS = IDA_data.getobject('tsdatcne') 
    data_edge_TS = IDA_data.getobject('tsdatene') 
    
    data_LIB = LIN.getobject('ne') 

    time_data_start_index = np.argmin(np.abs(time_start_data - time))
    time_data_end_index = np.argmin(np.abs(time_end_data - time))
    
    index_core_TS_interval = index_core_TS[time_data_start_index:time_data_end_index + 1]-1
    index_edge_TS_interval = index_edge_TS[time_data_start_index:time_data_end_index + 1]-1
    data_core_TS_interval = data_core_TS[:, :, time_data_start_index:time_data_end_index + 1]
    data_edge_TS_interval = data_edge_TS[:, :, time_data_start_index:time_data_end_index + 1]
    
    time_data_start_index_LIB = np.argmin(np.abs(time_start_data - time_LIB))
    time_data_end_index_LIB = np.argmin(np.abs(time_end_data - time_LIB))
    
    data_LIB_interval = data_LIB[:, time_data_start_index_LIB:time_data_end_index_LIB + 1]
    
    # Calculate coordinates from diagnostics

    x_core_TS = IDA_data.getobject('x_ts_co')
    x_edge_TS = IDA_data.getobject('x_ts_ed')
    x_LIB = LIN.getobject('rhop')
    
    x_core_TS_interval = np.zeros((x_core_TS.shape[0], time_data_end_index - time_data_start_index + 1))
    x_edge_TS_interval = np.zeros((x_edge_TS.shape[0], time_data_end_index - time_data_start_index + 1))
    x_LIB_interval = np.zeros((x_LIB.shape[0], time_data_end_index_LIB - time_data_start_index_LIB + 1))

    for j in range(time_data_end_index - time_data_start_index + 1):
        x_core_TS_interval[:, j] = x_core_TS[:, j + time_data_start_index - 1]
        x_edge_TS_interval[:, j] = x_edge_TS[:, j + time_data_start_index - 1]

    for j in range(time_data_end_index_LIB - time_data_start_index_LIB + 1):
        x_LIB_interval[:, j] = x_LIB[:, j + time_data_start_index_LIB - 1]
        
    # Return dictionary
    
    index_core_TS = index_core_TS_interval.copy()
    rho_core_TS = x_core_TS_interval.copy()
    ne_core_TS = data_core_TS_interval.copy()
    
    index_edge_TS = index_edge_TS_interval.copy()
    rho_edge_TS = x_edge_TS_interval.copy()
    ne_edge_TS = data_edge_TS_interval.copy()
    
    rho_LIB = x_LIB_interval.copy()
    ne_LIB = data_LIB_interval.copy()
    
    ne_dict = {}
    
    ne_dict['time_window'] = time_window
    ne_dict['rho'] = rho
    ne_dict['ne_median'] = ne_median
    ne_dict['ne_median_uncertainty'] = ne_median_uncertainty
    
    ne_dict['rho_LIB'] = rho_LIB
    ne_dict['ne_LIB'] = ne_LIB
    
    ne_dict['index_core_TS'] = index_core_TS
    ne_dict['rho_core_TS'] = rho_core_TS
    ne_dict['ne_core_TS'] = ne_core_TS
    
    ne_dict['index_edge_TS'] = index_edge_TS
    ne_dict['rho_edge_TS'] = rho_edge_TS
    ne_dict['ne_edge_TS'] = ne_edge_TS

    return ne_dict


def Te(shot,time_window,time_window_data,shotfile='IDA',exp='augd'):
    
    # Extract data from integrated data analysis
    
    IDA = sf.SFREAD(shotfile, shot, exp=exp)
    IDA_data = sf.SFREAD('IDA', shot, exp='augd')
    
    time = IDA.getobject('time') 
    
    data = IDA.getobject('Te') 
    data_uncertainty = IDA.getobject('Te_unc') 
    
    time_start = time_window[0]
    time_end = time_window[1]
    
    time_start_data = time_window_data[0]
    time_end_data = time_window_data[1]
    
    index_start = np.argmin(np.abs(time - time_start))
    index_end = np.argmin(np.abs(time - time_end))
    
    # # Calculate the ELM onset times
    
    # ELM = sf.SFREAD(shotfile_elm, shot, exp=exp_elm)
    
    # time_ELM = ELM.getobject('t_begELM') 
    
    # t_IDA = time[index_start:index_end + 1]
    
    # index_ELM_start = np.argmin(np.abs(time_ELM - time_start))
    # index_ELM_end = np.argmin(np.abs(time_ELM - time_end))    
    
    # zeros = np.zeros(index_ELM_end - index_ELM_start + 1)
    # t_begELM = time_ELM[index_ELM_start:index_ELM_end + 1]
    
    # for i in range(len(t_begELM) - 1, -1, -1):
    #     a_i = np.min(np.abs(t_IDA - t_begELM[i]))
    #     if a_i > 3e-4:
    #         t_begELM = np.delete(t_begELM, i)
    
    # # Extract time windows relative to ELM onset times from IDA
    
    # time_windows = [[0 for _ in range(time_points_elm)] for _ in range(len(t_begELM))]
    # data_windows = [[[0 for _ in range(time_points_elm)] for _ in range(len(t_begELM))] for _ in range(data.shape[0])]
    # data_windows_uncertainty = [[[0 for _ in range(time_points_elm)] for _ in range(len(t_begELM))] for _ in range(data.shape[0])]
    
    # data_for_median = [[[0 for _ in range(len(t_begELM))] for _ in range(time_points_elm)] for _ in range(data.shape[0])]
    # data_for_median_uncertainty = [[[0 for _ in range(len(t_begELM))] for _ in range(time_points_elm)] for _ in range(data.shape[0])]
    
    # med = [[0 for _ in range(time_points_elm)] for _ in range(data.shape[0])]
    # med_uncertainty = [[0 for _ in range(time_points_elm)] for _ in range(data.shape[0])]
    
    # rho = IDA.getobject('rhop')[:,0]  
    
    # Te_median = np.zeros((len(rho), time_points_elm))
    # Te_median_uncertainty = np.zeros((len(rho), time_points_elm))
    
    # for rho_index in range(data.shape[0]):
        
    #     for i in range(len(t_begELM)):
    #         t_start = t_begELM[i] - interval_elm
    #         index_IDA_start = np.argmin(np.abs(time - t_start))
    #         index_IDA_end = index_IDA_start + time_points_elm
    
    #         for j in range(index_IDA_end - index_IDA_start):
    #             try:
    #                 time_windows[i][j] = (t_IDA[index_IDA_start + j] - t_IDA[index_IDA_start])
    #                 data_windows[rho_index][i][j] = (data[rho_index, index_IDA_start + j])
    #                 data_windows_uncertainty[rho_index][i][j] = (data_uncertainty[rho_index, index_IDA_start + j])
    #             except IndexError:
    #                 pass
    
    # Calculate data and median for integrated data analysis
    
    #     for i in range(len(data_windows[rho_index]) - 3):
    #         for j in range(len(data_windows[rho_index][i])):
    #             data_for_median[rho_index][j][i] = data_windows[rho_index][i][j]
    #             data_for_median_uncertainty[rho_index][j][i] = data_windows_uncertainty[rho_index][i][j]
                
    #     for i in range(len(data_for_median[rho_index])):
    #         med[rho_index][i] = np.median(data_for_median[rho_index][i], axis=0, overwrite_input=True)
    #         med_uncertainty[rho_index][i] = np.median(data_for_median_uncertainty[rho_index][i], axis=0, overwrite_input=True)
        
    #     for i in range(len(rho)):
    #         Te_median[i,:] = np.convolve(med[i][:], np.ones(4)/4, mode='same')
    #         Te_median_uncertainty[i,:] = np.convolve(med_uncertainty[i][:], np.ones(4)/4, mode='same')
    
    # Te_median[:,0] = Te_median[:,2]
    # Te_median[:,1] = Te_median[:,2]
    # Te_median[:,-1] = Te_median[:,-3]
    # Te_median[:,-2] = Te_median[:,-3]
    
    # Te_median_uncertainty[:,1] = Te_median_uncertainty[:,3]
    # Te_median_uncertainty[:,2] = Te_median_uncertainty[:,3]
    # Te_median_uncertainty[:,-1] = Te_median_uncertainty[:,-3]
    # Te_median_uncertainty[:,-2] = Te_median_uncertainty[:,-3]
    
    # time_window = np.array(time_windows[0])-interval_elm
    
    # # Smooth data
    
    # for j in range(len(t_begELM)):
    #     Te_median[:,j] = savgol_filter(Te_median[:,j], 11, 1)
    #     Te_median_uncertainty[:,j] = savgol_filter(Te_median_uncertainty[:,j], 11, 1)
    
    rho = IDA.getobject('rhop')[:,0]

    index_start = np.argmin(np.abs(time - time_start))
    index_end = np.argmin(np.abs(time - time_end))
    
    Te = data[:, index_start:index_end + 1]
    Te_unc = data_uncertainty[:, index_start:index_end + 1]
    
    Te_median = np.zeros(len(rho))
    Te_median_uncertainty = np.zeros(len(rho))
    
    for i in range(len(rho)):
        Te_median[i] = np.median(Te[i, :])
        Te_median_uncertainty[i] = np.median(Te_unc[i, :])
    
    Te_median = Te_median.reshape(-1, 1)
    Te_median_uncertainty = Te_median_uncertainty.reshape(-1, 1)
    
    # Extract data from diagnostics
    
    index_core_TS = IDA_data.getobject('Ntscprof')
    index_edge_TS = IDA_data.getobject('Ntseprof')
    data_core_TS = IDA_data.getobject('tsdatcte') 
    data_edge_TS = IDA_data.getobject('tsdatete') 
    
    time_data_start_index = np.argmin(np.abs(time_start_data - time))
    time_data_end_index = np.argmin(np.abs(time_end_data - time))
    
    index_core_TS_interval = index_core_TS[time_data_start_index:time_data_end_index + 1]-1
    index_edge_TS_interval = index_edge_TS[time_data_start_index:time_data_end_index + 1]-1
    data_core_TS_interval = data_core_TS[:, :, time_data_start_index:time_data_end_index + 1]
    data_edge_TS_interval = data_edge_TS[:, :, time_data_start_index:time_data_end_index + 1]
    
    index_ECE = IDA_data.getobject('Neceprof')
    data_ECE = IDA_data.getobject('ece_dat') 
    
    index_ECE_interval = index_ECE[time_data_start_index:time_data_end_index + 1]-1
    data_ECE_interval = data_ECE[:, :, time_data_start_index:time_data_end_index + 1]
    
    # Calculate coordinates from diagnostics

    x_core_TS = IDA_data.getobject('x_ts_co')
    x_edge_TS = IDA_data.getobject('x_ts_ed')
    
    x_core_TS_interval = np.zeros((x_core_TS.shape[0], time_data_end_index - time_data_start_index + 1))
    x_edge_TS_interval = np.zeros((x_edge_TS.shape[0], time_data_end_index - time_data_start_index + 1))

    for j in range(time_data_end_index - time_data_start_index + 1):
        x_core_TS_interval[:, j] = x_core_TS[:, j + time_data_start_index - 1]
        x_edge_TS_interval[:, j] = x_edge_TS[:, j + time_data_start_index - 1]
        
    x_ECE = IDA_data.getobject('ece_rhop')
    
    x_ECE_interval = np.zeros((x_ECE.shape[0], time_data_end_index - time_data_start_index + 1))

    for j in range(time_data_end_index - time_data_start_index + 1):
        x_ECE_interval[:, j] = x_ECE[:, j + time_data_start_index - 1]
        
    # Return dictionary
    
    index_core_TS = index_core_TS_interval.copy()
    rho_core_TS = x_core_TS_interval.copy()
    Te_core_TS = data_core_TS_interval.copy()
    
    index_edge_TS = index_edge_TS_interval.copy()
    rho_edge_TS = x_edge_TS_interval.copy()
    Te_edge_TS = data_edge_TS_interval.copy()
    
    index_ECE = index_ECE_interval.copy()
    rho_ECE = x_ECE_interval.copy()
    Te_ECE = data_ECE_interval.copy()
    
    Te_dict = {}
    
    Te_dict['time_window'] = time_window
    Te_dict['rho'] = rho
    Te_dict['Te_median'] = Te_median
    Te_dict['Te_median_uncertainty'] = Te_median_uncertainty

    Te_dict['index_core_TS'] = index_core_TS
    Te_dict['rho_core_TS'] = rho_core_TS
    Te_dict['Te_core_TS'] = Te_core_TS
    
    Te_dict['index_edge_TS'] = index_edge_TS
    Te_dict['rho_edge_TS'] = rho_edge_TS
    Te_dict['Te_edge_TS'] = Te_edge_TS
    
    Te_dict['index_ECE'] = index_ECE
    Te_dict['rho_ECE'] = rho_ECE
    Te_dict['Te_ECE'] = Te_ECE

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

    
def nimp_core(shot_imp_main,imp_main,shotfile_imp_main,
              shots_imp_background,imp_background,shotfiles_imp_background,
              time_windows_background,exp='augd'):
    
    # Open shotfiles
    
    EQU_MAIN = sf.EQU(shot_imp_main,diag='EQH')
    EQU_BACKGROUND = []
    for i in range(0,len(imp_background)):
        EQU_BACKGROUND.append(sf.EQU(shots_imp_background[i],diag='EQH'))
    
    SF_MAIN = sf.SFREAD(shotfile_imp_main, shot_imp_main, exp=exp)
    SF_BACKGROUND = []
    for i in range(0,len(imp_background)):
        SF_BACKGROUND.append(sf.SFREAD(shotfiles_imp_background[i], shots_imp_background[i], exp=exp))
        
    times_start_background = time_windows_background[0]
    times_end_background = time_windows_background[1]
        
    # Timebases
    
    time_imp_main = SF_MAIN.getobject('time') 
    times_imp_background = []
    for i in range(0,len(imp_background)):
        times_imp_background.append(SF_BACKGROUND[i].getobject('time'))
        
    # Areabases
        
    R_imp_main = SF_MAIN.getobject('R').T 
    z_imp_main = SF_MAIN.getobject('z').T
    R_imp_background = []
    z_imp_background = []
    for i in range(0,len(imp_background)):
        R_imp_background.append(SF_BACKGROUND[i].getobject('R').T)
        z_imp_background.append(SF_BACKGROUND[i].getobject('z').T)
        
    # Densities
    
    if imp_main == 'He':
        n_imp_main = SF_MAIN.getobject('nimp_plc') 
    else:
        n_imp_main = SF_MAIN.getobject('nimp') 
    n_imp_main_err = SF_MAIN.getobject('err_nimp')    
    n_imp_background = []
    n_imp_background_err = []
    for i in range(0,len(imp_background)):
        if imp_background[i] == 'He':
            n_imp_background.append(SF_BACKGROUND[i].getobject('nimp_plc'))
        else:
            n_imp_background.append(SF_BACKGROUND[i].getobject('nimp'))
        n_imp_background_err.append(SF_BACKGROUND[i].getobject('err_nimp'))
        
    # Remove empty signals
    
    for j in reversed(range(len(R_imp_main[0,:]))):
        if R_imp_main[0,j] == 0. or n_imp_main[0,j] == 0.:
            R_imp_main = np.delete(R_imp_main, j, axis=1)
            z_imp_main = np.delete(z_imp_main, j, axis=1)
            n_imp_main = np.delete(n_imp_main, j, axis=1)
            n_imp_main_err = np.delete(n_imp_main_err, j, axis=1)
    for i in range(0,len(imp_background)):
        for j in reversed(range(len(R_imp_background[i][0,:]))):
            if R_imp_background[i][0,j] == 0. or n_imp_background[i][0,j] == 0.:
                R_imp_background[i] = np.delete(R_imp_background[i], j, axis=1)
                z_imp_background[i] = np.delete(z_imp_background[i], j, axis=1)
                n_imp_background[i] = np.delete(n_imp_background[i], j, axis=1)
                n_imp_background_err[i] = np.delete(n_imp_background_err[i], j, axis=1)
    
    # Calculate flux surfaces
    
    R_imp_main_flux_surfaces = np.mean(R_imp_main, axis=0)
    z_imp_main_flux_surfaces = np.mean(z_imp_main, axis=0)
    
    rho_imp_main = sf.rz2rho(EQU_MAIN, R_imp_main_flux_surfaces, z_imp_main_flux_surfaces,
                             t_in=time_imp_main[0], coord_out='rho_pol', extrapolate=False)
    index_first_surface_main = np.argmin(rho_imp_main)
    n_imp_main = n_imp_main[:,index_first_surface_main:]
    n_imp_main_err = n_imp_main_err[:,index_first_surface_main:]
    rho_imp_main = rho_imp_main[:,index_first_surface_main:]
    
    R_imp_background_flux_surfaces = []
    z_imp_background_flux_surfaces = []
    rho_imp_background = []
    index_first_surface_background = []
    
    for i in range(0,len(imp_background)):
        
        R_imp_background_flux_surfaces.append(np.mean(R_imp_background[i], axis=0))
        z_imp_background_flux_surfaces.append(np.mean(z_imp_background[i], axis=0))
        
        rho_imp_background.append(sf.rz2rho(EQU_BACKGROUND[i], R_imp_background_flux_surfaces[i], z_imp_background_flux_surfaces[i],
                                 t_in=times_imp_background[i][0], coord_out='rho_pol', extrapolate=False))
        index_first_surface_background.append(np.argmin(rho_imp_background[i]))
        n_imp_background[i] = n_imp_background[i][:,index_first_surface_main:]
        n_imp_background_err[i] = n_imp_background_err[i][:,index_first_surface_main:]
        rho_imp_background[i] = rho_imp_background[i][:,index_first_surface_background[i]:]
        
    # Remove edge signals
    
    try:
    
        for j in reversed(range(len(rho_imp_main[0,:]))):
            if rho_imp_main[0,j] > 0.9:
                rho_imp_main = np.delete(rho_imp_main, j, axis=1)
                n_imp_main = np.delete(n_imp_main, j, axis=1)
                n_imp_main_err = np.delete(n_imp_main_err, j, axis=1)
        
        for i in range(0,len(imp_background)):
            for j in reversed(range(len(rho_imp_background[i][0,:]))):
                if rho_imp_background[i][0,j] > 0.9:
                    rho_imp_background[i] = np.delete(rho_imp_background[i], j, axis=1)
                    n_imp_background[i] = np.delete(n_imp_background[i], j, axis=1)
                    n_imp_background_err[i] = np.delete(n_imp_background_err[i], j, axis=1)
                    
    except:
        
        pass
                
    # Mean values of background impurity densities
        
    index_time_start = []
    index_time_end = []
    for i in range(0,len(imp_background)):
        index_time_start.append(np.argmin(np.abs(times_start_background[i] - times_imp_background[i])))
        index_time_end.append(np.argmin(np.abs(times_end_background[i] - times_imp_background[i])))
        n_imp_background[i] = np.median(n_imp_background[i][index_time_start[i]:index_time_end[i]], axis=0)
        n_imp_background_err[i] = np.median(n_imp_background_err[i][index_time_start[i]:index_time_end[i]], axis=0)
        
    # Return dictionary
    
    nimp_core = {}
    
    nimp_core['time_imp_main'] = time_imp_main
    nimp_core['rho_imp_main'] = rho_imp_main
    nimp_core['n_imp_main'] = n_imp_main
    nimp_core['n_imp_main_err'] = n_imp_main_err
    
    nimp_core['rho_imp_background'] = rho_imp_background
    nimp_core['n_imp_background'] = n_imp_background
    nimp_core['n_imp_background_err'] = n_imp_background_err
    
    return nimp_core
        

def nimp_gas(shot,imp,shotfile_lines,shotfile_currents,
             shotfile_total_pressures,gauge_div,gauge_pump,impurity_pressure_drop,main_species_pressure_drop,
             time_window_pre_puff,time_window_correction,exp='augd'):
    
    SF_LINES = sf.SFREAD(shotfile_lines, shot, exp=exp)
    SF_CURRENTS = sf.SFREAD(shotfile_currents, shot, exp=exp)
    SF_PRESSURES = sf.SFREAD(shotfile_total_pressures, shot, exp=exp) 
    
    time_lines = SF_LINES.getobject('TIME')
    time_currents = SF_CURRENTS.getobject('Time')
    time_pressures = SF_PRESSURES.getobject('TIME')
    
    location_index = 2
    
    current = SF_CURRENTS.getobject('I_mon_2')
    
    if SF_PRESSURES.getobject('F' + str(gauge_pump).zfill(2)) is not None:
        flux_density = SF_PRESSURES.getobject('F' + str(gauge_pump).zfill(2))
    else:
        flux_density = SF_PRESSURES.getobject('F' + str(gauge_div).zfill(2)) / main_species_pressure_drop
    
    L_main = SF_LINES.getobject('D_0_6561')[:,location_index-1]
    if imp == 'He':
        L_imp = SF_LINES.getobject('He0_6678')[:,location_index-1]
        
    time = np.linspace(0, 10, 1001)
    current = np.interp(time, time_currents, current)
    L_main = np.interp(time, time_lines, L_main)
    L_imp = np.interp(time, time_lines, L_imp)
    flux_density = np.interp(time, time_pressures, flux_density)
    
    current_mean = np.mean(current[int((time_window_pre_puff[0]*100+1)):int((time_window_pre_puff[1]*100+1))])
    L_main_mean = np.mean(L_main[int((time_window_pre_puff[0]*100+1)):int((time_window_pre_puff[1]*100+1))])
    flux_density_mean = np.nanmean(flux_density[int((time_window_pre_puff[0]*100+1)):int((time_window_pre_puff[1]*100+1))])
    neutral_density_mean = (flux_density_mean*4)/1245
    
    j = int((time_window_correction[0]*100+1))
    ratio = np.zeros(int(time_window_correction[1])*100)
    while j < time_window_correction[1]*100:
        ratio[j] = current[j]/current_mean
        L_main[j] = L_main[j]/ratio[j]
        L_imp[j] = L_imp[j]/ratio[j]
        j += 1
        
    f_imp = L_imp/L_main_mean
    
    temp = neutral_density_mean/L_main_mean
    n_main = temp*L_main
    n_imp = neutral_density_mean*f_imp
    
    # Return dictionary
    
    nimp_gas = {}
    
    nimp_gas['time_imp_gas'] = time
    
    nimp_gas['n_imp_gas_div'] = impurity_pressure_drop*n_imp
    nimp_gas['n_imp_gas_pump'] = n_imp
    
    return nimp_gas
    