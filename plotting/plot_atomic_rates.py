import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
from aurora import transport_utils
from aurora import plot_tools

def plot(namelist,exp_data,asim,out,time_plot,time_exp_shift=0.0,xmin=None,xmax=None):
    
    rhop_exp = exp_data[f'{namelist["imp"]}_density_plasma']['rhop'].transpose()[:,0]
    n_imp_exp = exp_data[f'{namelist["imp"]}_density_plasma'][f'n_{namelist["imp"]}'].transpose()
    n_imp_unc_exp = exp_data[f'{namelist["imp"]}_density_plasma'][f'n_{namelist["imp"]}_unc'].transpose()
    
    time_exp = exp_data[f'{namelist["imp"]}_density_plasma']['time'].transpose() + time_exp_shift
    
    time = asim.time_out
    
    idx = np.argmin(np.abs(np.asarray(time) - time_plot))
    
    idx_exp = np.argmin(np.abs(np.asarray(time_exp) - time_plot))
    
    n_imp_exp = n_imp_exp[:,idx_exp]
    n_imp_unc_exp = n_imp_unc_exp[:,idx_exp]
    
    nz = out['nz']
    nimp = nz[:,:,idx]
    
    fig = plt.figure()
    fig.set_size_inches(20, 8, forward=True)
    ax1 = plt.subplot2grid((44, 54), (0, 0), rowspan=18, colspan=11, fig=fig)
    ax2 = plt.subplot2grid((44, 54), (0, 14), rowspan=18, colspan=11, fig=fig)
    ax3 = plt.subplot2grid((44, 54), (0, 28), rowspan=18, colspan=11, fig=fig)
    ax4 = plt.subplot2grid((44, 54), (0, 42), rowspan=18, colspan=11, fig=fig)
    ax5 = plt.subplot2grid((44, 54), (21, 0), rowspan=18, colspan=11, fig=fig)
    ax6 = plt.subplot2grid((44, 54), (21, 14), rowspan=18, colspan=11, fig=fig)
    ax7 = plt.subplot2grid((44, 54), (21, 28), rowspan=18, colspan=11, fig=fig)
    ax8 = plt.subplot2grid((44, 54), (21, 42), rowspan=18, colspan=11, fig=fig)
    
    colors = plot_tools.load_color_codes_reservoirs()
    blue,light_blue,green,light_green,grey,light_grey,red,light_red = colors
    
    color_indices = np.zeros(len(out))
    colors = ()
    cmap = mpl.colormaps['viridis']
    
    for i in range(0,nimp.shape[1]):
        color_indices[i] = (1/(nimp.shape[1]+1)) * (i+1)
        colors = colors + (cmap(color_indices[i]),) 
    
    ax1.set_title('Simulation', loc='right', fontsize = 12)
    ax1.errorbar(rhop_exp,n_imp_exp,yerr=n_imp_unc_exp,ecolor=light_blue,linewidth=1,fmt=' ',zorder=0)
    ax1.scatter(rhop_exp,n_imp_exp,color=blue,zorder=1,label='Experiment')
    for i in range(0,nimp.shape[1]):
        ax1.plot(asim.rhop_grid,nimp[:,i],linewidth=2,color=colors[i],label=f'Simulation, z = {i}')
    ax1.set_ylim(0,None)
    ax1.tick_params(axis='x',labelbottom=False)
    ax1.set_ylabel(f'$n_{{{namelist["imp"]}}}$ [$\mathrm{{cm}}$$^{{-3}}$]')
    ax1.legend(loc="best").set_draggable(True)
    
    ax2.set_title('Ionization rates', loc='right', fontsize = 12)
    for i in range(0,nimp.shape[1]-1):
        ax2.plot(asim.rhop_grid,asim.Sne_rates[:,i,idx]/asim.ne[0,:],linewidth=2,color=colors[i],label=fr'z = {i} ${{\rightarrow}}$ z = {i+1}')
    ax2.set_ylim(1e-16,1e-6)
    ax2.tick_params(axis='x',labelbottom=False)
    ax2.set_ylabel(f'$S_{{{namelist["imp"]}}}^{{ion}}$ [$\mathrm{{cm}}$$^{{3}}$$\mathrm{{s}}$$^{{-1}}$]')
    ax2.set_yscale('log')
    ax2.legend(loc="best").set_draggable(True)
    
    ax3.set_title('Radiative + dieletric recomb. rates', loc='right', fontsize = 12)
    for i in range(1,nimp.shape[1]):
        ax3.plot(asim.rhop_grid,asim.Rne_RDR_rates[:,i-1,idx]/asim.ne[0,:],linewidth=2,color=colors[i],label=fr'z = {i} ${{\rightarrow}}$ z = {i-1}')
    ax3.set_ylim(1e-16,1e-6)
    ax3.tick_params(axis='x',labelbottom=False)
    ax3.set_ylabel(fr'$\alpha_{{{namelist["imp"]}}}^{{rec,RDR}}$ [$\mathrm{{cm}}$$^{{3}}$$\mathrm{{s}}$$^{{-1}}$]')
    ax3.set_yscale('log')
    ax3.legend(loc="best").set_draggable(True)
    
    ax4.set_title('CX-driven recomb. rates', loc='right', fontsize = 12)
    for i in range(1,nimp.shape[1]):
        ax4.plot(asim.rhop_grid,asim.Rne_CX_rates[:,i-1,idx]/asim.ne[0,:],linewidth=2,color=colors[i],label=fr'z = {i} ${{\rightarrow}}$ z = {i-1}')
    ax4.set_ylim(1e-16,1e-6)
    ax4.tick_params(axis='x',labelbottom=False)
    ax4.set_ylabel(fr'$\alpha_{{{namelist["imp"]}}}^{{rec,CX}}$ [$\mathrm{{cm}}$$^{{3}}$$\mathrm{{s}}$$^{{-1}}$]')
    ax4.set_yscale('log')
    ax4.legend(loc="best").set_draggable(True)
    
    rhop_ne, ne, rho_LIB, ne_LIB, rho_core_TS, ne_core_TS, index_core_TS, rho_edge_TS, ne_edge_TS, index_edge_TS = electron_density(exp_data)
    
    ax5.set_title('Electron density profile', loc='right', fontsize = 12)
    ax5.scatter(rho_LIB[:,0],ne_LIB[:,0],label='LIB',color='green',s=0.2)
    if ne_LIB.shape[1] > 1:
        for j in range(1,ne_LIB.shape[1]):
            ax5.scatter(rho_LIB[:,j],ne_LIB[:,j],color='green',s=0.2)
    ax5.scatter(rho_core_TS[:,0],ne_core_TS[:,index_core_TS[0],0],label='Core TS',color='blue',s=0.2)
    if ne_core_TS.shape[2] > 1:
        for j in range(1,ne_core_TS.shape[2]):
            ax5.scatter(rho_core_TS[:,j],ne_core_TS[:,index_core_TS[j],j],color='blue',s=0.2)  
    ax5.scatter(rho_edge_TS[:,0],ne_edge_TS[:,index_edge_TS[0],0],label='Edge TS',color='red',s=0.2)
    if ne_edge_TS.shape[2] > 1:
        for j in range(1,ne_edge_TS.shape[2]):
            ax5.scatter(rho_edge_TS[:,j],ne_edge_TS[:,index_edge_TS[j],j],color='red',s=0.2)
    ax5.plot(asim.rhop_grid,asim.ne[0,:],linewidth = 3,color='black',label='IDA fit')
    ax5.set_ylim(0,np.max(ne)*1.15)
    ax5.set_ylabel('$n_e$ [$\mathrm{{cm}}$$^{{-3}}$]')
    ax5.legend(loc="best").set_draggable(True)
    
    rhop_Te, Te, index_ECE, rho_ECE, Te_ECE, rho_core_TS, Te_core_TS, index_core_TS, rho_edge_TS, Te_edge_TS, index_edge_TS = electron_temperature(exp_data)
    
    ax6.set_title('Electron temperature profile', loc='right', fontsize = 12)
    ax6.scatter(rho_ECE[:,0],Te_ECE[:,index_ECE[0],0],label='ECE',color='green',s=0.2)
    if Te_ECE.shape[1] > 1:
        for j in range(1,Te_ECE.shape[1]):
            ax6.scatter(rho_ECE[:,j],Te_ECE[:,index_ECE[j],j],color='green',s=0.2)     
    ax6.scatter(rho_core_TS[:,0],Te_core_TS[:,index_core_TS[0],0],label='Core TS',color='blue',s=0.2)
    if Te_core_TS.shape[2] > 1:
        for j in range(1,Te_core_TS.shape[2]):
            ax6.scatter(rho_core_TS[:,j],Te_core_TS[:,index_core_TS[j],j],color='blue',s=0.2)         
    ax6.scatter(rho_edge_TS[:,0],Te_edge_TS[:,index_edge_TS[0],0],label='Edge TS',color='red',s=0.2)
    if Te_edge_TS.shape[2] > 1:
        for j in range(1,Te_edge_TS.shape[2]):
            ax6.scatter(rho_edge_TS[:,j],Te_edge_TS[:,index_edge_TS[j],j],color='red',s=0.2)
    ax6.plot(asim.rhop_grid,asim.Te[0,:],linewidth = 3,color='black',label='IDA fit')
    ax6.set_ylim(0,np.max(Te)*1.15)
    ax6.set_ylabel('$T_e$ [$\mathrm{{eV}}$]')
    ax6.legend(loc="best").set_draggable(True)
    
    rhop_Ti, Ti, rho_core, Ti_core, rho_edge, Ti_edge = ion_temperature(exp_data)
    
    ax7.set_title('Ion temperature profile', loc='right', fontsize = 12)
    ax7.scatter(rho_core[:,0],Ti_core[:,0],label='Core CXRS',color='blue',s=0.2)
    for j in range(1,Ti_core.shape[1]):
        ax7.scatter(rho_core[:,j],Ti_core[:,j],color='blue',s=0.2)            
    ax7.scatter(rho_edge[:,0],Ti_edge[:,0],label='Edge CXRS',color='red',s=0.2)
    for j in range(1,Ti_edge.shape[1]):
        ax7.scatter(rho_edge[:,j],Ti_edge[:,j],color='red',s=0.2)
    ax7.plot(asim.rhop_grid,asim.Ti[0,:],linewidth = 3,color='black',label='Spline fit')
    ax7.set_ylim(0,np.max(Ti)*1.15)
    ax7.set_ylabel('$T_i$ [$\mathrm{{eV}}$]')
    ax7.legend(loc="best").set_draggable(True)
     
    rhop_n0, n0 = neutral_density(exp_data)
    
    ax8.set_title(f'{asim.main_element} neutral density profile', loc='right', fontsize = 12)
    ax8.plot(asim.rhop_grid,asim.n0[0,:],linewidth = 3,color='black',label='SOLPS')
    ax8.set_ylim(0,np.max(n0)*1.15)
    ax8.set_ylabel(f'$n_{{{asim.main_element}^0}}$ [$\mathrm{{cm}}$$^{{-3}}$]')
    ax8.legend(loc="best").set_draggable(True)
    
    ax3.sharey(ax2)
    ax4.sharey(ax2)
    
    ax2.sharex(ax1)
    ax3.sharex(ax1)
    ax4.sharex(ax1)
    ax5.sharex(ax1)
    ax6.sharex(ax1)
    ax7.sharex(ax1)
    ax8.sharex(ax1)
    
    if xmin is not None:
        ax8.set_xlim(xmin,xmax)
    else:
        ax8.set_xlim(0,np.max(asim.rhop_grid))
    
    ax1.axvline(1, c="r", ls=":", lw=0.5)
    ax2.axvline(1, c="r", ls=":", lw=0.5)
    ax3.axvline(1, c="r", ls=":", lw=0.5)
    ax4.axvline(1, c="r", ls=":", lw=0.5)
    ax5.axvline(1, c="r", ls=":", lw=0.5)
    ax6.axvline(1, c="r", ls=":", lw=0.5)
    ax7.axvline(1, c="r", ls=":", lw=0.5)
    ax8.axvline(1, c="r", ls=":", lw=0.5)
    
    ax5.set_xlabel(r'$\rho_p$')
    ax6.set_xlabel(r'$\rho_p$')
    ax7.set_xlabel(r'$\rho_p$')
    ax8.set_xlabel(r'$\rho_p$')
    
    plt.tight_layout()
    
    print('Atomic rates plots prepared.')
    
    
def plot_sensitivity_analysis(namelists,exp_data,asim,out,time_plot,labels,time_exp_shift=0.0,xmin=None,xmax=None):
    
    rhop_exp = exp_data[f'{namelists[0]["imp"]}_density_plasma']['rhop'].transpose()[:,0]
    n_imp_exp = exp_data[f'{namelists[0]["imp"]}_density_plasma'][f'n_{namelists[0]["imp"]}'].transpose()
    n_imp_unc_exp = exp_data[f'{namelists[0]["imp"]}_density_plasma'][f'n_{namelists[0]["imp"]}_unc'].transpose()
    
    time_exp = exp_data[f'{namelists[0]["imp"]}_density_plasma']['time'].transpose() + time_exp_shift
    
    time = asim[0].time_out
    
    idx = np.argmin(np.abs(np.asarray(time) - time_plot))
    
    idx_exp = np.argmin(np.abs(np.asarray(time_exp) - time_plot))
    
    n_imp_exp = n_imp_exp[:,idx_exp]
    n_imp_unc_exp = n_imp_unc_exp[:,idx_exp]

    nz = ()
    nimp = ()
    for i in range(0,len(out)):
        nz = nz + (out[i]['nz'],)
        nimp = nimp + (nz[i][:,:,idx],)
    
    fig = plt.figure()
    fig.set_size_inches(20, 8, forward=True)
    ax1 = plt.subplot2grid((44, 54), (0, 0), rowspan=18, colspan=11, fig=fig)
    ax2 = plt.subplot2grid((44, 54), (0, 14), rowspan=18, colspan=11, fig=fig)
    ax3 = plt.subplot2grid((44, 54), (0, 28), rowspan=18, colspan=11, fig=fig)
    ax4 = plt.subplot2grid((44, 54), (0, 42), rowspan=18, colspan=11, fig=fig)
    ax5 = plt.subplot2grid((44, 54), (21, 0), rowspan=18, colspan=11, fig=fig)
    ax6 = plt.subplot2grid((44, 54), (21, 14), rowspan=18, colspan=11, fig=fig)
    ax7 = plt.subplot2grid((44, 54), (21, 28), rowspan=18, colspan=11, fig=fig)
    ax8 = plt.subplot2grid((44, 54), (21, 42), rowspan=18, colspan=11, fig=fig)
    
    colors = plot_tools.load_color_codes_reservoirs()
    blue,light_blue,green,light_green,grey,light_grey,red,light_red = colors
    
    color_indices_sim = np.zeros(len(out))
    colors_sim = ()
    cmap = mpl.colormaps['viridis']
    for i in range(0,len(out)):
        color_indices_sim[i] = (1/(len(out)+1)) * (i+1)
        colors_sim = colors_sim + (cmap(color_indices_sim[i]),) 
        
    color_indices = np.zeros(nimp[0].shape[1])
    colors = ()
    cmap = mpl.colormaps['viridis']
    for i in range(0,nimp[0].shape[1]):
        color_indices[i] = (1/(nimp[0].shape[1]+1)) * (i+1)
        colors = colors + (cmap(color_indices[i]),) 
    
    ax1.set_title('Simulation', loc='right', fontsize = 12)
    ax1.errorbar(rhop_exp,n_imp_exp,yerr=n_imp_unc_exp,ecolor=light_blue,linewidth=1,fmt=' ',zorder=0)
    ax1.scatter(rhop_exp,n_imp_exp,color=blue,zorder=1,label='Experiment')
    ax1.plot(asim[0].rhop_grid,nimp[0][:,-1],linewidth=2,color=colors_sim[0],label=labels[0])
    ax1.plot(asim[1].rhop_grid,nimp[1][:,-1],linewidth=2,color=colors_sim[1],label=labels[1])
    ax1.set_ylim(0,None)
    ax1.tick_params(axis='x',labelbottom=False)
    ax1.set_ylabel(f'$n_{{{namelists[0]["imp"]}}}$ [$\mathrm{{cm}}$$^{{-3}}$]')
    ax1.legend(loc="best").set_draggable(True)
    
    ax2.set_title('Ionization rates', loc='right', fontsize = 12)
    for i in range(0,nimp[0].shape[1]-1):
        ax2.plot(asim[0].rhop_grid,asim[0].Sne_rates[:,i,idx],linewidth=2,color=colors[i],label=fr'z = {i} ${{\rightarrow}}$ z = {i+1}')
    for i in range(0,nimp[1].shape[1]-1):
        ax2.plot(asim[1].rhop_grid,asim[1].Sne_rates[:,i,idx],linewidth=2,linestyle='--',color=colors[i]) 
    ax2.set_ylim(1e-2,1e8)
    ax2.tick_params(axis='x',labelbottom=False)
    ax2.set_ylabel(f'$S_{{{namelists[0]["imp"]}}}^{{ion}}$ [$\mathrm{{cm}}$$^{{-3}}$$\mathrm{{s}}$$^{{-1}}$]')
    ax2.set_yscale('log')
    ax2.legend(loc="best").set_draggable(True)
    
    ax3.set_title('Radiative + dieletric recomb. rates', loc='right', fontsize = 12)
    for i in range(1,nimp[0].shape[1]):
        ax3.plot(asim[0].rhop_grid,asim[0].Rne_RDR_rates[:,i-1,idx],linewidth=2,color=colors[i],label=fr'z = {i} ${{\rightarrow}}$ z = {i-1}')
    for i in range(1,nimp[1].shape[1]):
        ax3.plot(asim[1].rhop_grid,asim[1].Rne_RDR_rates[:,i-1,idx],linewidth=2,linestyle='--',color=colors[i])
    ax3.set_ylim(1e-2,1e8)
    ax3.tick_params(axis='x',labelbottom=False)
    ax3.set_ylabel(fr'$\alpha_{{{namelists[0]["imp"]}}}^{{rec,RDR}}$ [$\mathrm{{cm}}$$^{{-3}}$$\mathrm{{s}}$$^{{-1}}$]')
    ax3.set_yscale('log')
    ax3.legend(loc="best").set_draggable(True)
    
    ax4.set_title('CX-driven recomb. rates', loc='right', fontsize = 12)
    for i in range(1,nimp[0].shape[1]):
        ax4.plot(asim[0].rhop_grid,asim[0].Rne_CX_rates[:,i-1,idx],linewidth=2,color=colors[i],label=fr'z = {i} ${{\rightarrow}}$ z = {i-1}')
    for i in range(1,nimp[1].shape[1]):
        ax4.plot(asim[1].rhop_grid,asim[1].Rne_CX_rates[:,i-1,idx],linewidth=2,linestyle='--',color=colors[i])
    ax4.set_ylim(1e-2,1e8)
    ax4.tick_params(axis='x',labelbottom=False)
    ax4.set_ylabel(fr'$\alpha_{{{namelists[0]["imp"]}}}^{{rec,CX}}$ [$\mathrm{{cm}}$$^{{-3}}$$\mathrm{{s}}$$^{{-1}}$]')
    ax4.set_yscale('log')
    ax4.legend(loc="best").set_draggable(True)
    
    rhop_ne, ne, rho_LIB, ne_LIB, rho_core_TS, ne_core_TS, index_core_TS, rho_edge_TS, ne_edge_TS, index_edge_TS = electron_density(exp_data)
    
    ax5.set_title('Electron density profile', loc='right', fontsize = 12)
    ax5.scatter(rho_LIB[:,0],ne_LIB[:,0],label='LIB',color='green',s=0.2)
    if ne_LIB.shape[1] > 1:
        for j in range(1,ne_LIB.shape[1]):
            ax5.scatter(rho_LIB[:,j],ne_LIB[:,j],color='green',s=0.2)
    ax5.scatter(rho_core_TS[:,0],ne_core_TS[:,index_core_TS[0],0],label='Core TS',color='blue',s=0.2)
    if ne_core_TS.shape[2] > 1:
        for j in range(1,ne_core_TS.shape[2]):
            ax5.scatter(rho_core_TS[:,j],ne_core_TS[:,index_core_TS[j],j],color='blue',s=0.2)  
    ax5.scatter(rho_edge_TS[:,0],ne_edge_TS[:,index_edge_TS[0],0],label='Edge TS',color='red',s=0.2)
    if ne_edge_TS.shape[2] > 1:
        for j in range(1,ne_edge_TS.shape[2]):
            ax5.scatter(rho_edge_TS[:,j],ne_edge_TS[:,index_edge_TS[j],j],color='red',s=0.2)
    ax5.plot(asim[0].rhop_grid,asim[0].ne[0,:],linewidth = 3,color='black',label='IDA fit')
    ax5.set_ylim(0,np.max(ne)*1.15)
    ax5.set_ylabel('$n_e$ [$\mathrm{{cm}}$$^{{-3}}$]')
    ax5.legend(loc="best").set_draggable(True)
    
    rhop_Te, Te, index_ECE, rho_ECE, Te_ECE, rho_core_TS, Te_core_TS, index_core_TS, rho_edge_TS, Te_edge_TS, index_edge_TS = electron_temperature(exp_data)
    
    ax6.set_title('Electron temperature profile', loc='right', fontsize = 12)
    ax6.scatter(rho_ECE[:,0],Te_ECE[:,index_ECE[0],0],label='ECE',color='green',s=0.2)
    if Te_ECE.shape[1] > 1:
        for j in range(1,Te_ECE.shape[1]):
            ax6.scatter(rho_ECE[:,j],Te_ECE[:,index_ECE[j],j],color='green',s=0.2)     
    ax6.scatter(rho_core_TS[:,0],Te_core_TS[:,index_core_TS[0],0],label='Core TS',color='blue',s=0.2)
    if Te_core_TS.shape[2] > 1:
        for j in range(1,Te_core_TS.shape[2]):
            ax6.scatter(rho_core_TS[:,j],Te_core_TS[:,index_core_TS[j],j],color='blue',s=0.2)         
    ax6.scatter(rho_edge_TS[:,0],Te_edge_TS[:,index_edge_TS[0],0],label='Edge TS',color='red',s=0.2)
    if Te_edge_TS.shape[2] > 1:
        for j in range(1,Te_edge_TS.shape[2]):
            ax6.scatter(rho_edge_TS[:,j],Te_edge_TS[:,index_edge_TS[j],j],color='red',s=0.2)
    ax6.plot(asim[0].rhop_grid,asim[0].Te[0,:],linewidth = 3,color='black',label='IDA fit')
    ax6.set_ylim(0,np.max(Te)*1.15)
    ax6.set_ylabel('$T_e$ [$\mathrm{{eV}}$]')
    ax6.legend(loc="best").set_draggable(True)
    
    rhop_Ti, Ti, rho_core, Ti_core, rho_edge, Ti_edge = ion_temperature(exp_data)
    
    ax7.set_title('Ion temperature profile', loc='right', fontsize = 12)
    ax7.scatter(rho_core[:,0],Ti_core[:,0],label='Core CXRS',color='blue',s=0.2)
    for j in range(1,Ti_core.shape[1]):
        ax7.scatter(rho_core[:,j],Ti_core[:,j],color='blue',s=0.2)            
    ax7.scatter(rho_edge[:,0],Ti_edge[:,0],label='Edge CXRS',color='red',s=0.2)
    for j in range(1,Ti_edge.shape[1]):
        ax7.scatter(rho_edge[:,j],Ti_edge[:,j],color='red',s=0.2)
    ax7.plot(asim[0].rhop_grid,asim[0].Ti[0,:],linewidth = 3,color='black',label='Spline fit')
    ax7.set_ylim(0,np.max(Ti)*1.15)
    ax7.set_ylabel('$T_i$ [$\mathrm{{eV}}$]')
    ax7.legend(loc="best").set_draggable(True)
     
    rhop_n0, n0 = neutral_density(exp_data)
    
    ax8.set_title(f'{asim[0].main_element} neutral density profile', loc='right', fontsize = 12)
    ax8.plot(asim[0].rhop_grid,asim[0].n0[0,:],linewidth = 3,color='black',label='SOLPS')
    ax8.set_ylim(0,np.max(n0)*1.15)
    ax8.set_ylabel(f'$n_{{{asim[0].main_element}^0}}$ [$\mathrm{{cm}}$$^{{-3}}$]')
    ax8.legend(loc="best").set_draggable(True)
    
    ax3.sharey(ax2)
    ax4.sharey(ax2)
    
    ax2.sharex(ax1)
    ax3.sharex(ax1)
    ax4.sharex(ax1)
    ax5.sharex(ax1)
    ax6.sharex(ax1)
    ax7.sharex(ax1)
    ax8.sharex(ax1)
    
    if xmin is not None:
        ax8.set_xlim(xmin,xmax)
    else:
        ax8.set_xlim(0,np.max(asim[0].rhop_grid))
    
    ax1.axvline(1, c="r", ls=":", lw=0.5)
    ax2.axvline(1, c="r", ls=":", lw=0.5)
    ax3.axvline(1, c="r", ls=":", lw=0.5)
    ax4.axvline(1, c="r", ls=":", lw=0.5)
    ax5.axvline(1, c="r", ls=":", lw=0.5)
    ax6.axvline(1, c="r", ls=":", lw=0.5)
    ax7.axvline(1, c="r", ls=":", lw=0.5)
    ax8.axvline(1, c="r", ls=":", lw=0.5)
    
    ax5.set_xlabel(r'$\rho_p$')
    ax6.set_xlabel(r'$\rho_p$')
    ax7.set_xlabel(r'$\rho_p$')
    ax8.set_xlabel(r'$\rho_p$')
    
    plt.tight_layout()
    
    print('Atomic rates plots prepared.')
    
    
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
    
    return rhop_ne, ne, rho_LIB, ne_LIB, rho_core_TS, ne_core_TS, index_core_TS, rho_edge_TS, ne_edge_TS, index_edge_TS
    
    
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
    
    return rhop_Te, Te, index_ECE, rho_ECE, Te_ECE, rho_core_TS, Te_core_TS, index_core_TS, rho_edge_TS, Te_edge_TS, index_edge_TS
    

def ion_temperature(exp_data):
    
    kinetic_profiles = exp_data["kinetic_profiles"]
    ion_temperature_data = exp_data["ion_temperature_data"]
    
    rhop_Ti = kinetic_profiles["rhop_Ti"]
    Ti = kinetic_profiles["Ti"]
    
    rho_core = ion_temperature_data["rho_core"]
    Ti_core = ion_temperature_data["Ti_core"].transpose()
    rho_edge = ion_temperature_data["rho_edge"]
    Ti_edge = ion_temperature_data["Ti_edge"].transpose()
    
    return rhop_Ti, Ti, rho_core, Ti_core, rho_edge, Ti_edge


def neutral_density(exp_data):
    
    kinetic_profiles = exp_data["kinetic_profiles"]
    
    rhop_n0 = kinetic_profiles["rhop_n0"]
    n0 = kinetic_profiles["n0"]
    
    return rhop_n0, n0
