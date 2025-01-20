import numpy as np
import matplotlib.pyplot as plt


def electron_density(exp_data):
    
    kinetic_profiles = exp_data["kinetic_profiles"]
    electron_density_data = exp_data["electron_density_data"]
    
    rhop_ne = kinetic_profiles["rhop_ne"]
    ne = kinetic_profiles["ne"]
    ne_unc = kinetic_profiles["ne_unc"]
    
    index_core_TS = electron_density_data["index_core_TS"]-1
    rho_core_TS = electron_density_data["rho_core_TS"]
    ne_core_TS = electron_density_data["ne_core_TS"]
    index_edge_TS = electron_density_data["index_edge_TS"]-1
    rho_edge_TS = electron_density_data["rho_edge_TS"]
    ne_edge_TS = electron_density_data["ne_edge_TS"]
    rho_LIB = electron_density_data["rho_LIB"]
    ne_LIB = electron_density_data["ne_LIB"]

    fig, ax1 = plt.subplots(figsize=(7,5))
    fig.suptitle('Experimental electron density data')
    ax1.set_xlabel(r'$\rho_p$')
    ax1.set_ylabel('$n_e$ [$\mathrm{{cm}}$$^{{-3}}$]')
    
    ax1.scatter(rho_LIB[:,0],ne_LIB[:,0],label='LIB',color='green',s=0.5)
    if ne_LIB.shape[1] > 1:
        for j in range(1,ne_LIB.shape[1]):
            ax1.scatter(rho_LIB[:,j],ne_LIB[:,j],color='green',s=0.5)
     
    ax1.scatter(rho_core_TS[:,0],ne_core_TS[:,index_core_TS[0],0],label='Core TS',color='blue',s=0.5)
    if ne_core_TS.shape[2] > 1:
        for j in range(1,ne_core_TS.shape[2]):
            ax1.scatter(rho_core_TS[:,j],ne_core_TS[:,index_core_TS[j],j],color='blue',s=0.5)
            
    ax1.scatter(rho_edge_TS[:,0],ne_edge_TS[:,index_edge_TS[0],0],label='Edge TS',color='red',s=0.5)
    if ne_edge_TS.shape[2] > 1:
        for j in range(1,ne_edge_TS.shape[2]):
            ax1.scatter(rho_edge_TS[:,j],ne_edge_TS[:,index_edge_TS[j],j],color='red',s=0.5)
    
    ax1.plot(rhop_ne,ne,linewidth = 3,color='black',label='IDA fit')

    ax1.axvline(1, c="r", ls=":", lw=0.5)
    ax1.set_xlim(0,np.max(rhop_ne))
    ax1.set_ylim(0,np.max(ne)*1.15)
    ax1.legend(loc="best").set_draggable(True)
    
    print('Experimental electron density profile plot prepared.')
    
    
def electron_temperature(exp_data):
    
    kinetic_profiles = exp_data["kinetic_profiles"]
    electron_temperature_data = exp_data["electron_temperature_data"]
    
    rhop_Te = kinetic_profiles["rhop_Te"]
    Te = kinetic_profiles["Te"]
    Te_unc = kinetic_profiles["Te_unc"]
    
    index_core_TS = electron_temperature_data["index_core_TS"]-1
    rho_core_TS = electron_temperature_data["rho_core_TS"]
    Te_core_TS = electron_temperature_data["Te_core_TS"]
    index_edge_TS = electron_temperature_data["index_edge_TS"]-1
    rho_edge_TS = electron_temperature_data["rho_edge_TS"]
    Te_edge_TS = electron_temperature_data["Te_edge_TS"]
    index_ECE = electron_temperature_data["index_ECE"]-1
    rho_ECE = electron_temperature_data["rho_ECE"]
    Te_ECE = electron_temperature_data["Te_ECE"]

    fig, ax1 = plt.subplots(figsize=(7,5))
    fig.suptitle('Experimental electron temperature data')
    ax1.set_xlabel(r'$\rho_p$')
    ax1.set_ylabel('$T_e$ [$\mathrm{{eV}}$]')
    
    ax1.scatter(rho_ECE[:,0],Te_ECE[:,index_ECE[0],0],label='ECE',color='green',s=0.5)
    if Te_ECE.shape[1] > 1:
        for j in range(1,Te_ECE.shape[1]):
            ax1.scatter(rho_ECE[:,j],Te_ECE[:,index_ECE[j],j],color='green',s=0.5)
     
    ax1.scatter(rho_core_TS[:,0],Te_core_TS[:,index_core_TS[0],0],label='Core TS',color='blue',s=0.5)
    if Te_core_TS.shape[2] > 1:
        for j in range(1,Te_core_TS.shape[2]):
            ax1.scatter(rho_core_TS[:,j],Te_core_TS[:,index_core_TS[j],j],color='blue',s=0.5)
            
    ax1.scatter(rho_edge_TS[:,0],Te_edge_TS[:,index_edge_TS[0],0],label='Edge TS',color='red',s=0.5)
    if Te_edge_TS.shape[2] > 1:
        for j in range(1,Te_edge_TS.shape[2]):
            ax1.scatter(rho_edge_TS[:,j],Te_edge_TS[:,index_edge_TS[j],j],color='red',s=0.5)
    
    ax1.plot(rhop_Te,Te[0,:],linewidth = 3,color='black',label='IDA fit')

    ax1.axvline(1, c="r", ls=":", lw=0.5)
    ax1.set_xlim(0,np.max(rhop_Te))
    ax1.set_ylim(0,np.max(Te)*1.15)
    ax1.legend(loc="best").set_draggable(True)
    
    print('Experimental electron temperature profile plot prepared.')
    

def ion_temperature(exp_data):
    
    kinetic_profiles = exp_data["kinetic_profiles"]
    ion_temperature_data = exp_data["ion_temperature_data"]
    
    rhop_Ti = kinetic_profiles["rhop_Ti"]
    Ti = kinetic_profiles["Ti"]
    
    rho_core = ion_temperature_data["rho_core"]
    Ti_core = ion_temperature_data["Ti_core"].transpose()
    rho_edge = ion_temperature_data["rho_edge"]
    Ti_edge = ion_temperature_data["Ti_edge"].transpose()

    fig, ax1 = plt.subplots(figsize=(7,5))
    fig.suptitle('Experimental ion temperature data')
    ax1.set_xlabel(r'$\rho_p$')
    ax1.set_ylabel('$T_i$ [$\mathrm{{eV}}$]')
     
    ax1.scatter(rho_core[:,0],Ti_core[:,0],label='Core CXRS',color='blue',s=0.5)
    for j in range(1,Ti_core.shape[1]):
        ax1.scatter(rho_core[:,j],Ti_core[:,j],color='blue',s=0.5)
            
    ax1.scatter(rho_edge[:,0],Ti_edge[:,0],label='Edge CXRS',color='red',s=0.5)
    for j in range(1,Ti_edge.shape[1]):
        ax1.scatter(rho_edge[:,j],Ti_edge[:,j],color='red',s=0.5)
    
    ax1.plot(rhop_Ti,Ti,linewidth = 3,color='black',label='Spline fit')

    ax1.axvline(1, c="r", ls=":", lw=0.5)
    ax1.set_xlim(0,np.max(rhop_Ti))
    ax1.set_ylim(0,np.max(Ti)*1.15)
    ax1.legend(loc="best").set_draggable(True)
    
    print('Experimental ion temperature profile plot prepared.')
    