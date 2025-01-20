import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from aurora import plot_tools

def plot(namelist,exp_data,asim,out,time_inter_ELM,time_intra_ELM,time_ELM_peak,time_exp_shift=0.0,xmin=None,xmax=None):
    
    rhop_exp = exp_data[f'{namelist["imp"]}_density_plasma']['rhop'].transpose()[:,0]
    n_imp_exp = exp_data[f'{namelist["imp"]}_density_plasma'][f'n_{namelist["imp"]}'].transpose()
    n_imp_unc_exp = exp_data[f'{namelist["imp"]}_density_plasma'][f'n_{namelist["imp"]}_unc'].transpose()
    
    time_exp = exp_data[f'{namelist["imp"]}_density_plasma']['time'].transpose() + time_exp_shift
    
    time = asim.time_out
    
    idx_inter_ELM = np.argmin(np.abs(np.asarray(time) - time_inter_ELM))
    idx_intra_ELM = np.argmin(np.abs(np.asarray(time) - time_intra_ELM))
    idx_ELM_peak = np.argmin(np.abs(np.asarray(time) - time_ELM_peak))
    
    idx_exp = np.argmin(np.abs(np.asarray(time_exp) - time_inter_ELM))
    
    n_imp_exp = n_imp_exp[:,idx_exp]
    n_imp_unc_exp = n_imp_unc_exp[:,idx_exp]
    
    nz = out['nz']
    nimp_inter_ELM = nz[:,:,idx_inter_ELM]
    nimp_intra_ELM = nz[:,:,idx_intra_ELM]
    
    Sne_rate_inter_ELM = asim.Sne_rates[:,0,idx_inter_ELM]
    Sne_rate_intra_ELM = asim.Sne_rates[:,0,idx_ELM_peak]
    
    rcl_rad_prof_inter_ELM = asim.rcl_rad_prof[:,idx_inter_ELM]
    rcl_rad_prof_intra_ELM = asim.rcl_rad_prof[:,idx_ELM_peak]
    
    rfl_rad_prof_inter_ELM = asim.rfl_rad_prof[:,idx_inter_ELM]
    rfl_rad_prof_intra_ELM = asim.rfl_rad_prof[:,idx_ELM_peak]
    
    spt_rad_prof_inter_ELM = np.zeros((asim.spt_rad_prof.shape[0],asim.spt_rad_prof.shape[1]))
    spt_rad_prof_intra_ELM = np.zeros((asim.spt_rad_prof.shape[0],asim.spt_rad_prof.shape[1]))
    for i in range(0,asim.spt_rad_prof.shape[1]):
        spt_rad_prof_inter_ELM[:,i] = asim.spt_rad_prof[:,i,idx_inter_ELM]
        spt_rad_prof_intra_ELM[:,i] = asim.spt_rad_prof[:,i,idx_ELM_peak]
    
    recycling_profiles_inter_ELM = [rcl_rad_prof_inter_ELM,rfl_rad_prof_inter_ELM]
    recycling_profiles_intra_ELM = [rcl_rad_prof_intra_ELM,rfl_rad_prof_intra_ELM]
    for i in range(0,asim.spt_rad_prof.shape[1]):
        recycling_profiles_inter_ELM.append(spt_rad_prof_inter_ELM[:,i])
        recycling_profiles_intra_ELM.append(spt_rad_prof_intra_ELM[:,i])
    
    maxs_inter_ELM = np.zeros(2+asim.spt_rad_prof.shape[1])
    maxs_intra_ELM = np.zeros(2+asim.spt_rad_prof.shape[1])
    
    maxs_inter_ELM[0] = np.max(rcl_rad_prof_inter_ELM)
    maxs_inter_ELM[1] = np.max(rfl_rad_prof_inter_ELM)
    for i in range(2,asim.spt_rad_prof.shape[1]+2):
        maxs_inter_ELM[i] = np.max(spt_rad_prof_inter_ELM[:,i-2])
    maxs_intra_ELM[0] = np.max(rcl_rad_prof_intra_ELM)
    maxs_intra_ELM[1] = np.max(rfl_rad_prof_intra_ELM)
    for i in range(2,asim.spt_rad_prof.shape[1]+2):
        maxs_intra_ELM[i] = np.max(spt_rad_prof_intra_ELM[:,i-2])
    
    index_max_inter_ELM = np.argmax(maxs_inter_ELM)
    index_max_intra_ELM = np.argmax(maxs_intra_ELM)
    
    for i in list(range(2+asim.spt_rad_prof.shape[1])):
        recycling_profiles_inter_ELM[i] = recycling_profiles_inter_ELM[i] * (np.max(recycling_profiles_inter_ELM[index_max_inter_ELM])/np.max(recycling_profiles_inter_ELM[i]))
    for i in list(range(2+asim.spt_rad_prof.shape[1])):
        recycling_profiles_intra_ELM[i] = recycling_profiles_intra_ELM[i] * (np.max(recycling_profiles_intra_ELM[index_max_intra_ELM])/np.max(recycling_profiles_intra_ELM[i]))
        
    fig = plt.figure()
    fig.set_size_inches(12, 7.5, forward=True)
    ax1 = plt.subplot2grid((50, 24), (0, 0), rowspan=18, colspan=10, fig=fig)
    ax2 = plt.subplot2grid((50, 24), (21, 0), rowspan=18, colspan=10, fig=fig)
    #ax3 = plt.subplot2grid((60, 24), (42, 0), rowspan=18, colspan=10, fig=fig)
    ax4 = plt.subplot2grid((50, 24), (0, 12), rowspan=18, colspan=10, fig=fig)
    ax5 = plt.subplot2grid((50, 24), (21, 12), rowspan=18, colspan=10, fig=fig)
    #ax6 = plt.subplot2grid((60, 24), (42, 12), rowspan=18, colspan=10, fig=fig)
    
    colors = plot_tools.load_color_codes_reservoirs()
    blue,light_blue,green,light_green,grey,light_grey,red,light_red = colors
    
    colors_PWI = plot_tools.load_color_codes_PWI()
    reds, blues, light_blues, greens = colors_PWI
    
    color_indices = np.zeros(nz.shape[1])
    colors = ()
    cmap = mpl.colormaps['viridis']
    for i in range(0,nz.shape[1]):
        color_indices[i] = (1/(nz.shape[1]+1)) * (i+1)
        colors = colors + (cmap(color_indices[i]),) 
    
    ax1.set_title('Inter-ELM', loc='right', fontsize = 12)
    ax1.errorbar(rhop_exp,n_imp_exp,yerr=n_imp_unc_exp,ecolor=light_blue,linewidth=1,fmt=' ',zorder=0)
    ax1.scatter(rhop_exp,n_imp_exp,color=blue,zorder=1,label='Experiment')
    for i in range(0,nimp_inter_ELM.shape[1]):
        ax1.plot(asim.rhop_grid,nimp_inter_ELM[:,i],linewidth=2,color=colors[i],label=f'Simulation (z={i})')
    ax1.set_ylim(0,None)
    ax1.tick_params(axis='x',labelbottom=False)
    ax1.set_ylabel(f'$n_{{{namelist["imp"]}}}$ [$\mathrm{{cm}}$$^{{-3}}$]')
    ax1.legend(loc="best").set_draggable(True)
    
    ax2.set_title('Recycled neutrals profiles (normalized)', loc='right', fontsize = 12)
    ax2.plot(asim.rhop_grid,Sne_rate_inter_ELM,linewidth=1,linestyle='--',color='black')
    ax2.set_ylim(0,np.max(Sne_rate_inter_ELM)*1.2)
    ax2.set_ylabel(f'$S_{{{namelist["imp"]},0}}^{{ion}}$ [$\mathrm{{cm}}$$^{{-3}}$$\mathrm{{s}}$$^{{-1}}$]')
    ax2right = ax2.twinx()
    ax2right.plot(asim.rhop_grid,recycling_profiles_inter_ELM[0],linewidth=2,label=f'Promptly recycled (en. = {asim.imp_recycling_energy_eV} eV)',color='green')
    ax2right.plot(asim.rhop_grid,recycling_profiles_inter_ELM[1],linewidth=2,label=f'Reflected (en. = {round(asim.E_refl_main_wall[idx_inter_ELM])} eV)',color='red')
    ax2right.plot(asim.rhop_grid,recycling_profiles_inter_ELM[2],linewidth=2,label=f'Sputtered from {asim.imp} (en. = {round(asim.E_sput_main_wall[0,idx_inter_ELM])} eV)',color=blues[0])
    for i in range(1,asim.spt_rad_prof.shape[1]):
        ax2right.plot(asim.rhop_grid,recycling_profiles_inter_ELM[i+2],linewidth=2,label=f'Sputtered from {asim.background_species[i-1]} (en. = {round(asim.E_sput_main_wall[i,idx_inter_ELM])} eV)',color=blues[i])
    ax2right.tick_params(axis='y',labelright=False)
    ax2right.set_ylim(0,None)
    ax2right.set_ylabel(f'$n_{{{namelist["imp"]}^{{0}}}}$ [a.u.]')
    ax2right.legend(loc="best",bbox_to_anchor=(1, -0.25)).set_draggable(True)
    
    ax4.set_title(f'{round((time_intra_ELM-time_inter_ELM)*1000,1)} ms after ELM crash', loc='right', fontsize = 12)
    ax4.errorbar(rhop_exp,n_imp_exp,yerr=n_imp_unc_exp,ecolor=light_blue,linewidth=1,fmt=' ',zorder=0)
    ax4.scatter(rhop_exp,n_imp_exp,color=blue,zorder=1,label='Experiment')
    for i in range(0,nimp_inter_ELM.shape[1]):
        ax4.plot(asim.rhop_grid,nimp_intra_ELM[:,i],linewidth=2,color=colors[i],label=f'Simulation (z={i})')
    ax4.set_ylim(0,None)
    ax4.tick_params(axis='x',labelbottom=False)
    ax4.set_ylabel(f'$n_{{{namelist["imp"]}}}$ [$\mathrm{{cm}}$$^{{-3}}$]')
    ax4.legend(loc="best").set_draggable(True)
    
    ax5.set_title('Recycled neutrals profiles (normalized)', loc='right', fontsize = 12)
    ax5.plot(asim.rhop_grid,Sne_rate_intra_ELM,linewidth=1,linestyle='--',color='black')
    ax5.set_ylim(0,np.max(Sne_rate_intra_ELM)*1.2)
    ax5.set_ylabel(f'$S_{{{namelist["imp"]},0}}^{{ion}}$ [$\mathrm{{cm}}$$^{{-3}}$$\mathrm{{s}}$$^{{-1}}$]')
    ax5right = ax5.twinx()
    ax5right.plot(asim.rhop_grid,recycling_profiles_intra_ELM[0],linewidth=2,label=f'Promptly recycled (en. = {asim.imp_recycling_energy_eV} eV)',color='green')
    ax5right.plot(asim.rhop_grid,recycling_profiles_intra_ELM[1],linewidth=2,label=f'Reflected (en. = {round(asim.E_refl_main_wall[idx_ELM_peak])} eV)',color='red')
    ax5right.plot(asim.rhop_grid,recycling_profiles_intra_ELM[2],linewidth=2,label=f'Sputtered from {asim.imp} (en. = {round(asim.E_sput_main_wall[0,idx_ELM_peak])} eV)',color=blues[0])
    for i in range(1,asim.spt_rad_prof.shape[1]):
        ax5right.plot(asim.rhop_grid,recycling_profiles_intra_ELM[i+2],linewidth=2,label=f'Sputtered from {asim.background_species[i-1]} (en. = {round(asim.E_sput_main_wall[i,idx_ELM_peak])} eV)',color=blues[i])
    ax5right.tick_params(axis='y',labelright=False)
    ax5right.set_ylim(0,None)
    ax5right.set_ylabel(f'$n_{{{namelist["imp"]}^{{0}}}}$ [a.u.]')
    ax5right.legend(loc="best",bbox_to_anchor=(1, -0.25)).set_draggable(True)
    
    ax2.sharex(ax1)
    ax4.sharex(ax1)
    ax5.sharex(ax1)
    
    if xmin is not None:
        ax5.set_xlim(xmin,xmax)
    else:
        ax5.set_xlim(0,np.max(asim.rhop_grid))
        
    ax1.axvline(1, c="r", ls=":", lw=0.5)
    ax2.axvline(1, c="r", ls=":", lw=0.5)
    ax4.axvline(1, c="r", ls=":", lw=0.5)
    ax5.axvline(1, c="r", ls=":", lw=0.5)
    
    ax2.set_xlabel(r'$\rho_p$')
    ax5.set_xlabel(r'$\rho_p$')
    
    plt.tight_layout()
    
    print('Neutrals recycling profile plots prepared.')
    
    
def plot_sensitivity_analysis(namelists,exp_data,asim,out,time_inter_ELM,time_intra_ELM,time_ELM_peak,labels,time_exp_shift=0.0,xmin=None,xmax=None):
    
    rhop_exp = exp_data[f'{namelists[0]["imp"]}_density_plasma']['rhop'].transpose()[:,0]
    n_imp_exp = exp_data[f'{namelists[0]["imp"]}_density_plasma'][f'n_{namelists[0]["imp"]}'].transpose()
    n_imp_unc_exp = exp_data[f'{namelists[0]["imp"]}_density_plasma'][f'n_{namelists[0]["imp"]}_unc'].transpose()
    
    time_exp = exp_data[f'{namelists[0]["imp"]}_density_plasma']['time'].transpose() + time_exp_shift
    
    time = asim[0].time_out
    
    idx_inter_ELM = np.argmin(np.abs(np.asarray(time) - time_inter_ELM))
    idx_intra_ELM = np.argmin(np.abs(np.asarray(time) - time_intra_ELM))
    idx_ELM_peak = np.argmin(np.abs(np.asarray(time) - time_ELM_peak))
    
    idx_exp = np.argmin(np.abs(np.asarray(time_exp) - time_inter_ELM))
    
    n_imp_exp = n_imp_exp[:,idx_exp]
    n_imp_unc_exp = n_imp_unc_exp[:,idx_exp]
    
    nz = ()
    nimp_inter_ELM = ()
    nimp_intra_ELM = ()
    for i in range(0,len(out)):
        nz = nz + (out[i]['nz'],)
        nimp_inter_ELM = nimp_inter_ELM + (nz[i][:,:,idx_inter_ELM],)
        nimp_intra_ELM = nimp_intra_ELM + (nz[i][:,:,idx_intra_ELM],)
      
    Sne_rate_inter_ELM = ()
    Sne_rate_intra_ELM = ()
    
    for i in range(0,len(out)):
        Sne_rate_inter_ELM = Sne_rate_inter_ELM + (asim[i].Sne_rates[:,0,idx_inter_ELM],)
        Sne_rate_intra_ELM = Sne_rate_intra_ELM + (asim[i].Sne_rates[:,0,idx_ELM_peak],)
    
    rcl_rad_prof_inter_ELM = ()
    rcl_rad_prof_intra_ELM = ()
    for i in range(0,len(out)):
        rcl_rad_prof_inter_ELM = rcl_rad_prof_inter_ELM + (asim[i].rcl_rad_prof[:,idx_inter_ELM],)
        rcl_rad_prof_intra_ELM = rcl_rad_prof_intra_ELM + (asim[i].rcl_rad_prof[:,idx_ELM_peak],)
    
    rfl_rad_prof_inter_ELM = ()
    rfl_rad_prof_intra_ELM = ()
    for i in range(0,len(out)):
        rfl_rad_prof_inter_ELM = rfl_rad_prof_inter_ELM + (asim[i].rfl_rad_prof[:,idx_inter_ELM],)
        rfl_rad_prof_intra_ELM = rfl_rad_prof_intra_ELM + (asim[i].rfl_rad_prof[:,idx_ELM_peak],)
    
    spt_rad_prof_inter_ELM = ()
    spt_rad_prof_intra_ELM = ()
    for i in range(0,len(out)):
        spt_rad_prof_inter_ELM = spt_rad_prof_inter_ELM + (np.zeros((asim[i].spt_rad_prof.shape[0],asim[i].spt_rad_prof.shape[1])),)
        spt_rad_prof_intra_ELM = spt_rad_prof_intra_ELM + (np.zeros((asim[i].spt_rad_prof.shape[0],asim[i].spt_rad_prof.shape[1])),)
        for j in range(0,asim[i].spt_rad_prof.shape[1]):
            spt_rad_prof_inter_ELM[i][:,j] = asim[i].spt_rad_prof[:,j,idx_inter_ELM]
            spt_rad_prof_intra_ELM[i][:,j] = asim[i].spt_rad_prof[:,j,idx_ELM_peak]
    
    recycling_profiles_inter_ELM = ()
    recycling_profiles_intra_ELM = ()
    for i in range(0,len(out)):
        recycling_profiles_inter_ELM = recycling_profiles_inter_ELM + ([rcl_rad_prof_inter_ELM[i],rfl_rad_prof_inter_ELM[i]],)
        recycling_profiles_intra_ELM = recycling_profiles_intra_ELM + ([rcl_rad_prof_intra_ELM[i],rfl_rad_prof_intra_ELM[i]],)
        for j in range(0,asim[i].spt_rad_prof.shape[1]):
            recycling_profiles_inter_ELM[i].append(spt_rad_prof_inter_ELM[i][:,j])
            recycling_profiles_intra_ELM[i].append(spt_rad_prof_intra_ELM[i][:,j])
    
    maxs_inter_ELM = ()
    maxs_intra_ELM = ()
    for i in range(0,len(out)):
        maxs_inter_ELM = maxs_inter_ELM + (np.zeros(2+asim[i].spt_rad_prof.shape[1]),)
        maxs_intra_ELM = maxs_intra_ELM + (np.zeros(2+asim[i].spt_rad_prof.shape[1]),)
    
    for i in range(0,len(out)):
        maxs_inter_ELM[i][0] = np.max(rcl_rad_prof_inter_ELM[i])
        maxs_inter_ELM[i][1] = np.max(rfl_rad_prof_inter_ELM[i])
        for j in range(2,asim[i].spt_rad_prof.shape[1]+2):
            maxs_inter_ELM[i][j] = np.max(spt_rad_prof_inter_ELM[i][:,j-2])
        maxs_intra_ELM[i][0] = np.max(rcl_rad_prof_intra_ELM[i])
        maxs_intra_ELM[i][1] = np.max(rfl_rad_prof_intra_ELM[i])
        for j in range(2,asim[i].spt_rad_prof.shape[1]+2):
            maxs_intra_ELM[i][j] = np.max(spt_rad_prof_intra_ELM[i][:,j-2])
    
    index_max_inter_ELM = ()
    index_max_intra_ELM = ()
    for i in range(0,len(out)):
        index_max_inter_ELM = index_max_inter_ELM + (np.argmax(maxs_inter_ELM[i]),)
        index_max_intra_ELM = index_max_intra_ELM + (np.argmax(maxs_intra_ELM[i]),)
    
    for i in range(0,len(out)):
        for j in list(range(2+asim[i].spt_rad_prof.shape[1])):
            recycling_profiles_inter_ELM[i][j] = recycling_profiles_inter_ELM[i][j] * (np.max(recycling_profiles_inter_ELM[i][index_max_inter_ELM[i]])/np.max(recycling_profiles_inter_ELM[i][j]))
        for j in list(range(2+asim[i].spt_rad_prof.shape[1])):
            recycling_profiles_intra_ELM[i][j] = recycling_profiles_intra_ELM[i][j] * (np.max(recycling_profiles_intra_ELM[i][index_max_intra_ELM[i]])/np.max(recycling_profiles_intra_ELM[i][j]))
        
    fig = plt.figure()
    fig.set_size_inches(13, 12, forward=True)
    ax1 = plt.subplot2grid((60, 26), (0, 0), rowspan=18, colspan=10, fig=fig)
    ax2 = plt.subplot2grid((60, 26), (21, 0), rowspan=18, colspan=10, fig=fig)
    ax3 = plt.subplot2grid((60, 26), (40, 0), rowspan=18, colspan=10, fig=fig)
    ax4 = plt.subplot2grid((60, 26), (0, 12), rowspan=18, colspan=10, fig=fig)
    ax5 = plt.subplot2grid((60, 26), (21, 12), rowspan=18, colspan=10, fig=fig)
    ax6 = plt.subplot2grid((60, 26), (40, 12), rowspan=18, colspan=10, fig=fig)
    
    colors = plot_tools.load_color_codes_reservoirs()
    blue,light_blue,green,light_green,grey,light_grey,red,light_red = colors
    
    colors_PWI = plot_tools.load_color_codes_PWI()
    reds, blues, light_blues, greens = colors_PWI
    
    color_indices = np.zeros(len(out))
    colors = ()
    cmap = mpl.colormaps['viridis']
    for i in range(0,len(out)):
        color_indices[i] = (1/(len(out)+1)) * (i+1)
        colors = colors + (cmap(color_indices[i]),) 
    
    ax1.set_title('Inter-ELM', loc='right', fontsize = 12)
    ax1.errorbar(rhop_exp,n_imp_exp,yerr=n_imp_unc_exp,ecolor=light_blue,linewidth=1,fmt=' ',zorder=0)
    ax1.scatter(rhop_exp,n_imp_exp,color=blue,zorder=1,label='Experiment')
    ax1.plot(asim[0].rhop_grid,nimp_inter_ELM[0][:,-1],linewidth=2,color=colors[0],label=labels[0])
    ax1.plot(asim[1].rhop_grid,nimp_inter_ELM[1][:,-1],linewidth=2,color=colors[1],label=labels[1])
    ax1.set_ylim(0,None)
    ax1.tick_params(axis='x',labelbottom=False)
    ax1.set_ylabel(f'$n_{{{namelists[0]["imp"]}^{{+}}}}$ [$\mathrm{{cm}}$$^{{-3}}$]')
    ax1.legend(loc="best").set_draggable(True)
    
    ax2.set_title('Recycled neutrals profiles (normalized)', loc='right', fontsize = 12)
    ax2.plot(asim[0].rhop_grid,Sne_rate_inter_ELM[0],linewidth=1,linestyle='--',color='black')
    ax2.set_ylim(0,np.max(Sne_rate_inter_ELM[0])*1.2)
    ax2.set_ylabel(f'$S_{{{namelists[0]["imp"]},0}}^{{ion}}$ [$\mathrm{{cm}}$$^{{-3}}$$\mathrm{{s}}$$^{{-1}}$]')
    ax2right = ax2.twinx()
    ax2right.plot(asim[0].rhop_grid,recycling_profiles_inter_ELM[0][0],linewidth=2,label=f'Promptly recycled (en. = {asim[0].imp_recycling_energy_eV} eV)',color='green')
    ax2right.plot(asim[0].rhop_grid,recycling_profiles_inter_ELM[0][1],linewidth=2,label=f'Reflected (en. = {round(asim[0].E_refl_main_wall[idx_inter_ELM])} eV)',color='red')
    ax2right.plot(asim[0].rhop_grid,recycling_profiles_inter_ELM[0][2],linewidth=2,label=f'Sputtered from {asim[0].imp} (en. = {round(asim[0].E_sput_main_wall[0,idx_inter_ELM])} eV)',color=blues[0])
    for i in range(1,asim[0].spt_rad_prof.shape[1]):
        ax2right.plot(asim[0].rhop_grid,recycling_profiles_inter_ELM[0][i+2],linewidth=2,label=f'Sputtered from {asim[0].background_species[i-1]} (en. = {round(asim[0].E_sput_main_wall[i,idx_inter_ELM])} eV)',color=blues[i])
    ax2right.tick_params(axis='y',labelright=False)
    ax2right.set_ylim(0,None)
    ax2.tick_params(axis='x',labelbottom=False)
    ax2right.set_ylabel(f'$n_{{{namelists[0]["imp"]}^{{0}}}}$ [a.u.]')
    ax2right.legend(loc="best").set_draggable(True)
    
    ax3.plot(asim[1].rhop_grid,Sne_rate_inter_ELM[1],linewidth=1,linestyle='--',color='black')
    ax3.set_ylim(0,np.max(Sne_rate_inter_ELM[1])*1.2)
    ax3.set_ylabel(f'$S_{{{namelists[1]["imp"]},0}}^{{ion}}$ [$\mathrm{{cm}}$$^{{-3}}$$\mathrm{{s}}$$^{{-1}}$]')
    ax3right = ax3.twinx()
    ax3right.plot(asim[1].rhop_grid,recycling_profiles_inter_ELM[1][0],linewidth=2,label=f'All recycled neutr. (en. = {asim[1].imp_recycling_energy_eV} eV)',color='black')
    ax3right.plot(asim[1].rhop_grid,recycling_profiles_inter_ELM[1][1],linewidth=2,color='black')
    ax3right.plot(asim[1].rhop_grid,recycling_profiles_inter_ELM[1][2],linewidth=2,color='black')
    for i in range(1,asim[1].spt_rad_prof.shape[1]):
        ax3right.plot(asim[1].rhop_grid,recycling_profiles_inter_ELM[1][i+2],linewidth=2,color='black')
    ax3right.tick_params(axis='y',labelright=False)
    ax3right.set_ylim(0,None)
    ax3right.set_ylabel(f'$n_{{{namelists[0]["imp"]}^{{0}}}}$ [a.u.]')
    ax3right.legend(loc="best").set_draggable(True)
    
    ax4.set_title(f'{round((time_intra_ELM-time_inter_ELM)*1000,1)} ms after ELM crash', loc='right', fontsize = 12)
    ax4.errorbar(rhop_exp,n_imp_exp,yerr=n_imp_unc_exp,ecolor=light_blue,linewidth=1,fmt=' ',zorder=0)
    ax4.scatter(rhop_exp,n_imp_exp,color=blue,zorder=1,label='Experiment')
    ax4.plot(asim[0].rhop_grid,nimp_intra_ELM[0][:,-1],linewidth=2,color=colors[0],label=labels[0])
    ax4.plot(asim[1].rhop_grid,nimp_intra_ELM[1][:,-1],linewidth=2,color=colors[1],label=labels[1])
    ax4.set_ylim(0,None)
    ax4.tick_params(axis='x',labelbottom=False)
    ax4.set_ylabel(f'$n_{{{namelists[0]["imp"]}^{{+}}}}$ [$\mathrm{{cm}}$$^{{-3}}$]')
    ax4.legend(loc="best").set_draggable(True)
    
    ax5.set_title('Recycled neutrals profiles (normalized)', loc='right', fontsize = 12)
    ax5.plot(asim[0].rhop_grid,Sne_rate_intra_ELM[0],linewidth=1,linestyle='--',color='black')
    ax5.set_ylim(0,np.max(Sne_rate_intra_ELM[0])*1.2)
    ax5.set_ylabel(f'$S_{{{namelists[0]["imp"]},0}}^{{ion}}$ [$\mathrm{{cm}}$$^{{-3}}$$\mathrm{{s}}$$^{{-1}}$]')
    ax5right = ax5.twinx()
    ax5right.plot(asim[0].rhop_grid,recycling_profiles_intra_ELM[0][0],linewidth=2,label=f'Promptly recycled (en. = {asim[0].imp_recycling_energy_eV} eV)',color='green')
    ax5right.plot(asim[0].rhop_grid,recycling_profiles_intra_ELM[0][1],linewidth=2,label=f'Reflected (en. = {round(asim[0].E_refl_main_wall[idx_ELM_peak])} eV)',color='red')
    ax5right.plot(asim[0].rhop_grid,recycling_profiles_intra_ELM[0][2],linewidth=2,label=f'Sputtered from {asim[0].imp} (en. = {round(asim[0].E_sput_main_wall[0,idx_ELM_peak])} eV)',color=blues[0])
    for i in range(1,asim[0].spt_rad_prof.shape[1]):
        ax5right.plot(asim[0].rhop_grid,recycling_profiles_intra_ELM[0][i+2],linewidth=2,label=f'Sputtered from {asim[0].background_species[i-1]} (en. = {round(asim[0].E_sput_main_wall[i,idx_ELM_peak])} eV)',color=blues[i])
    ax5right.tick_params(axis='y',labelright=False)
    ax5right.set_ylim(0,None)
    ax5.tick_params(axis='x',labelbottom=False)
    ax5right.set_ylabel(f'$n_{{{namelists[0]["imp"]}^{{0}}}}$ [a.u.]')
    ax5right.legend(loc="best").set_draggable(True)
    
    ax6.plot(asim[1].rhop_grid,Sne_rate_intra_ELM[1],linewidth=1,linestyle='--',color='black')
    ax6.set_ylim(0,np.max(Sne_rate_intra_ELM[1])*1.2)
    ax6.set_ylabel(f'$S_{{{namelists[1]["imp"]},0}}^{{ion}}$ [$\mathrm{{cm}}$$^{{-3}}$$\mathrm{{s}}$$^{{-1}}$]')
    ax6right = ax6.twinx()
    ax6right.plot(asim[1].rhop_grid,recycling_profiles_intra_ELM[1][0],linewidth=2,label=f'All recycled neutr. (en. = {asim[1].imp_recycling_energy_eV} eV)',color='black')
    ax6right.plot(asim[1].rhop_grid,recycling_profiles_intra_ELM[1][1],linewidth=2,color='black')
    ax6right.plot(asim[1].rhop_grid,recycling_profiles_intra_ELM[1][2],linewidth=2,color='black')
    for i in range(1,asim[1].spt_rad_prof.shape[1]):
        ax6right.plot(asim[1].rhop_grid,recycling_profiles_intra_ELM[1][i+2],linewidth=2,color='black')
    ax6right.tick_params(axis='y',labelright=False)
    ax6right.set_ylim(0,None)
    ax6right.set_ylabel(f'$n_{{{namelists[0]["imp"]}^{{0}}}}$ [a.u.]')
    ax6right.legend(loc="best").set_draggable(True)
    
    ax2.sharex(ax1)
    ax3.sharex(ax1)
    ax4.sharex(ax1)
    ax5.sharex(ax1)
    ax6.sharex(ax1)
    
    if xmin is not None:
        ax6.set_xlim(xmin,xmax)
    else:
        ax6.set_xlim(0,np.max(asim[0].rhop_grid))
        
    ax1.axvline(1, c="r", ls=":", lw=0.5)
    ax2.axvline(1, c="r", ls=":", lw=0.5)
    ax3.axvline(1, c="r", ls=":", lw=0.5)
    ax4.axvline(1, c="r", ls=":", lw=0.5)
    ax5.axvline(1, c="r", ls=":", lw=0.5)
    ax6.axvline(1, c="r", ls=":", lw=0.5)
    
    ax3.set_xlabel(r'$\rho_p$')
    ax6.set_xlabel(r'$\rho_p$')
    
    plt.tight_layout()
    
    print('Neutrals recycling profile plots prepared.')