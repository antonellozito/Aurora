import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
from aurora import transport_utils
from aurora import plot_tools

def plot(namelist,exp_data,asim,out,rhop,D,D_ELM,v,v_ELM,rhop_transp,transp_coeffs,time_inter_ELM,time_intra_ELM,time_exp_shift=0.0,xmin=None,xmax=None,Dmin=None,Dmax=None,vmin=None,vmax=None):
    
    rhop_exp = exp_data[f'{namelist["imp"]}_density_plasma']['rhop'].transpose()[:,0]
    n_imp_exp = exp_data[f'{namelist["imp"]}_density_plasma'][f'n_{namelist["imp"]}'].transpose()
    n_imp_unc_exp = exp_data[f'{namelist["imp"]}_density_plasma'][f'n_{namelist["imp"]}_unc'].transpose()
    
    time_exp = exp_data[f'{namelist["imp"]}_density_plasma']['time'].transpose() + time_exp_shift
    
    time = asim.time_out
    
    idx_inter_ELM = np.argmin(np.abs(np.asarray(time) - time_inter_ELM))
    idx_intra_ELM = np.argmin(np.abs(np.asarray(time) - time_intra_ELM))
    
    idx_exp = np.argmin(np.abs(np.asarray(time_exp) - time_inter_ELM))
    
    n_imp_exp = n_imp_exp[:,idx_exp]
    n_imp_unc_exp = n_imp_unc_exp[:,idx_exp]
    
    nz = out['nz']
    nimp_inter_ELM = nz[:,-1,idx_inter_ELM]
    nimp_intra_ELM = nz[:,-1,idx_intra_ELM]
    
    D_inter_ELM = scipy.signal.savgol_filter(transport_utils.interp_transp(rhop,np.array(D)/1e4,asim.rhop_grid,"Pchip_spline"), 25, 9)
    D_intra_ELM = scipy.signal.savgol_filter(transport_utils.interp_transp(rhop,np.array(D_ELM)/1e4,asim.rhop_grid,"Pchip_spline"), 25, 9)
    v_inter_ELM = scipy.signal.savgol_filter(transport_utils.interp_transp(rhop,np.array(v)/1e2,asim.rhop_grid,"Pchip_spline"), 25, 9)
    v_intra_ELM = scipy.signal.savgol_filter(transport_utils.interp_transp(rhop,np.array(v_ELM)/1e2,asim.rhop_grid,"Pchip_spline"), 25, 9)
    
    D_tot = transp_coeffs.Dz
    D_BP = transp_coeffs.Dz_BP
    D_PS = transp_coeffs.Dz_PS
    D_CL = transp_coeffs.Dz_CL
    
    v_BP = transp_coeffs.Vconv_BP
    v_BP = scipy.signal.savgol_filter(v_BP, 100, 3)
    v_PS = transp_coeffs.Vconv_PS
    v_PS = scipy.signal.savgol_filter(v_PS, 100, 3)
    v_CL = transp_coeffs.Vconv_CL
    v_CL = scipy.signal.savgol_filter(v_CL, 100, 3)
    v_tot = v_BP + v_PS + v_CL

    fig = plt.figure()
    fig.set_size_inches(12, 12, forward=True)
    ax1 = plt.subplot2grid((60, 24), (0, 0), rowspan=18, colspan=10, fig=fig)
    ax2 = plt.subplot2grid((60, 24), (21, 0), rowspan=18, colspan=10, fig=fig)
    ax3 = plt.subplot2grid((60, 24), (40, 0), rowspan=18, colspan=10, fig=fig)
    ax4 = plt.subplot2grid((60, 24), (0, 12), rowspan=18, colspan=10, fig=fig)
    ax5 = plt.subplot2grid((60, 24), (21, 12), rowspan=18, colspan=10, fig=fig)
    ax6 = plt.subplot2grid((60, 24), (40, 12), rowspan=18, colspan=10, fig=fig)
    
    colors = plot_tools.load_color_codes_reservoirs()
    blue,light_blue,green,light_green,grey,light_grey,red,light_red = colors
    
    ax1.set_title('Inter-ELM', loc='right', fontsize = 12)
    ax1.errorbar(rhop_exp,n_imp_exp,yerr=n_imp_unc_exp,ecolor=light_blue,linewidth=1,fmt=' ',zorder=0)
    ax1.scatter(rhop_exp,n_imp_exp,color=blue,zorder=1,label='Experiment')
    ax1.plot(asim.rhop_grid,nimp_inter_ELM,linewidth=2,color='black',label='Simulation')
    ax1.set_ylim(0,None)
    ax1.tick_params(axis='x',labelbottom=False)
    ax1.set_ylabel(f'$n_{{{namelist["imp"]}^{{+}}}}$ [$\mathrm{{cm}}$$^{{-3}}$]')
    ax1.legend(loc="best")
    
    ax2.set_title('Inter-ELM transport profiles', loc='right', fontsize = 12, color='red')
    ax2.plot(rhop_transp,D_BP,linewidth=1,linestyle='--',label='Banana-plateau')
    ax2.plot(rhop_transp,D_PS,linewidth=1,linestyle='--',label='Pfirsch–Schlüter')
    ax2.plot(rhop_transp,D_CL,linewidth=1,linestyle='--',label='Classical')
    ax2.plot(rhop_transp,D_tot,linewidth=2,linestyle='--',label='Tot. neoclassical $D$',color='black')
    ax2.plot(asim.rhop_grid,D_inter_ELM,linewidth=2,color='red')
    ax2.set_ylim(Dmin if Dmin is not None else None, Dmax if Dmax is not None else None)
    ax2.set_yscale('log')
    ax2.tick_params(axis='x',labelbottom=False)
    ax2.set_ylabel('$D$ [$\mathrm{{m}}$$^{{2}}$/$s$]')
    ax2.legend(loc="best")

    ax3.plot(rhop_transp,v_BP,linewidth=1,linestyle='--',label='Banana-plateau')
    ax3.plot(rhop_transp,v_PS,linewidth=1,linestyle='--',label='Pfirsch–Schlüter')
    ax3.plot(rhop_transp,v_CL,linewidth=1,linestyle='--',label='Classical')
    ax3.plot(rhop_transp,v_tot,linewidth=2,linestyle='--',label='Tot. neoclassical $v$',color='black')
    ax3.plot(asim.rhop_grid,v_inter_ELM,linewidth=2,color='red')
    ax3.set_ylim(vmin if vmin is not None else None, vmax if vmax is not None else None)
    ax3.set_ylabel('$v$ [$\mathrm{{m}}$/$s$]')
    ax3.legend(loc="best")
    
    ax4.set_title(f'{round((time_intra_ELM-time_inter_ELM)*1000,1)} ms after ELM crash', loc='right', fontsize = 12)
    ax4.errorbar(rhop_exp,n_imp_exp,yerr=n_imp_unc_exp,ecolor=light_blue,linewidth=1,fmt=' ',zorder=0)
    ax4.scatter(rhop_exp,n_imp_exp,color=blue,zorder=1,label='Experiment')
    ax4.plot(asim.rhop_grid,nimp_intra_ELM,linewidth=2,color='black',label='Simulation')
    ax4.set_ylim(0,None)
    ax4.tick_params(axis='x',labelbottom=False)
    ax4.set_ylabel(f'$n_{{{namelist["imp"]}^{{+}}}}$ [$\mathrm{{cm}}$$^{{-3}}$]')
    ax4.legend(loc="best")
    
    ax5.set_title('Intra-ELM transport profiles', loc='right', fontsize = 12, color='red')
    ax5.plot(asim.rhop_grid,D_intra_ELM,linewidth=2,color='red')
    ax5.set_ylim(Dmin if Dmin is not None else None, Dmax if Dmax is not None else None)
    ax5.set_yscale('log')
    ax5.tick_params(axis='x',labelbottom=False)
    ax5.set_ylabel('$D$ [$\mathrm{{m}}$$^{{2}}$/$s$]')
    
    ax6.plot(asim.rhop_grid,v_intra_ELM,linewidth=2,color='red')
    ax6.set_ylim(vmin if vmin is not None else None, vmax if vmax is not None else None)
    ax6.set_ylabel('$v$ [$\mathrm{{m}}$/$s$]')
        
    ax2.sharex(ax1)
    ax3.sharex(ax1)
    ax4.sharex(ax1)
    ax5.sharex(ax1)
    ax6.sharex(ax1)
    
    if xmin is not None:
        ax6.set_xlim(xmin,xmax)
    else:
        ax6.set_xlim(0,np.max(asim.rhop_grid))
    
    ax1.axvline(1, c="r", ls=":", lw=0.5)
    ax2.axvline(1, c="r", ls=":", lw=0.5)
    ax3.axvline(1, c="r", ls=":", lw=0.5)
    ax4.axvline(1, c="r", ls=":", lw=0.5)
    ax5.axvline(1, c="r", ls=":", lw=0.5)
    ax6.axvline(1, c="r", ls=":", lw=0.5)
    
    ax3.set_xlabel(r'$\rho_p$')
    ax6.set_xlabel(r'$\rho_p$')
    
    plt.tight_layout()
    
    print('Transport coefficients plots prepared.')
    

def plot_sensitivity_analysis(namelists,exp_data,asim,out,rhop,D,D_ELM,v,v_ELM,rhop_transp,transp_coeffs,time_inter_ELM,time_intra_ELM,labels,time_exp_shift=0.0,xmin=None,xmax=None,Dmin=None,Dmax=None,vmin=None,vmax=None):
    
    rhop_exp = exp_data[f'{namelists[0]["imp"]}_density_plasma']['rhop'].transpose()[:,0]
    n_imp_exp = exp_data[f'{namelists[0]["imp"]}_density_plasma'][f'n_{namelists[0]["imp"]}'].transpose()
    n_imp_unc_exp = exp_data[f'{namelists[0]["imp"]}_density_plasma'][f'n_{namelists[0]["imp"]}_unc'].transpose()
    
    time_exp = exp_data[f'{namelists[0]["imp"]}_density_plasma']['time'].transpose() + time_exp_shift
    
    time = asim[0].time_out
    
    idx_inter_ELM = np.argmin(np.abs(np.asarray(time) - time_inter_ELM))
    idx_intra_ELM = np.argmin(np.abs(np.asarray(time) - time_intra_ELM))
    
    idx_exp = np.argmin(np.abs(np.asarray(time_exp) - time_inter_ELM))
    
    n_imp_exp = n_imp_exp[:,idx_exp]
    n_imp_unc_exp = n_imp_unc_exp[:,idx_exp]

    nz = ()
    nimp_inter_ELM = ()
    nimp_intra_ELM = ()
    for i in range(0,len(out)):
        nz = nz + (out[i]['nz'],)
        nimp_inter_ELM = nimp_inter_ELM + (nz[i][:,-1,idx_inter_ELM],)
        nimp_intra_ELM = nimp_intra_ELM + (nz[i][:,-1,idx_intra_ELM],)
        
    D_inter_ELM = ()
    v_inter_ELM = ()
    for i in range(0,len(out)):
        D_inter_ELM = D_inter_ELM + (scipy.signal.savgol_filter(transport_utils.interp_transp(rhop,np.array(D[i])/1e4,asim[i].rhop_grid,"Pchip_spline"), 25, 9),)
        v_inter_ELM = v_inter_ELM + (scipy.signal.savgol_filter(transport_utils.interp_transp(rhop,np.array(v[i])/1e2,asim[i].rhop_grid,"Pchip_spline"), 25, 9),)
   
    D_intra_ELM = scipy.signal.savgol_filter(transport_utils.interp_transp(rhop,np.array(D_ELM)/1e4,asim[0].rhop_grid,"Pchip_spline"), 25, 9)
    v_intra_ELM = scipy.signal.savgol_filter(transport_utils.interp_transp(rhop,np.array(v_ELM)/1e2,asim[0].rhop_grid,"Pchip_spline"), 25, 9)

    D_tot = transp_coeffs.Dz
    D_BP = transp_coeffs.Dz_BP
    D_PS = transp_coeffs.Dz_PS
    D_CL = transp_coeffs.Dz_CL
    
    v_BP = transp_coeffs.Vconv_BP
    v_BP = scipy.signal.savgol_filter(v_BP, 100, 3)
    v_PS = transp_coeffs.Vconv_PS
    v_PS = scipy.signal.savgol_filter(v_PS, 100, 3)
    v_CL = transp_coeffs.Vconv_CL
    v_CL = scipy.signal.savgol_filter(v_CL, 100, 3)
    v_tot = v_BP + v_PS + v_CL

    fig = plt.figure()
    fig.set_size_inches(13, 12, forward=True)
    ax1 = plt.subplot2grid((60, 26), (0, 0), rowspan=18, colspan=10, fig=fig)
    ax2 = plt.subplot2grid((60, 26), (21, 0), rowspan=18, colspan=10, fig=fig)
    ax3 = plt.subplot2grid((60, 26), (40, 0), rowspan=18, colspan=10, fig=fig)
    ax4 = plt.subplot2grid((60, 26), (0, 12), rowspan=18, colspan=10, fig=fig)
    ax5 = plt.subplot2grid((60, 26), (21, 12), rowspan=18, colspan=10, fig=fig)
    ax6 = plt.subplot2grid((60, 26), (40, 12), rowspan=18, colspan=10, fig=fig)
    ax7 = plt.subplot2grid((60, 26), (0, 23), rowspan=58, colspan=1, fig=fig)
    
    colors = plot_tools.load_color_codes_reservoirs()
    blue,light_blue,green,light_green,grey,light_grey,red,light_red = colors
    
    color_indices = np.zeros(len(out))
    colors = ()
    cmap = mpl.colormaps['viridis']
    for i in range(0,len(out)):
        color_indices[i] = (1/(len(out)+1)) * (i+1)
        colors = colors + (cmap(color_indices[i]),) 
    
    ax1.set_title('Inter-ELM', loc='right', fontsize = 12)
    ax1.errorbar(rhop_exp,n_imp_exp,yerr=n_imp_unc_exp,ecolor=light_blue,linewidth=1,fmt=' ',zorder=0)
    ax1.scatter(rhop_exp,n_imp_exp,color=blue,zorder=1,label='Experiment')
    for i in range(0,len(out)):
        ax1.plot(asim[i].rhop_grid,nimp_inter_ELM[i],linewidth=2,color=colors[i])
    ax1.set_ylim(0,None)
    ax1.tick_params(axis='x',labelbottom=False)
    ax1.set_ylabel(f'$n_{{{namelists[0]["imp"]}^{{+}}}}$ [$\mathrm{{cm}}$$^{{-3}}$]')
    ax1.legend(loc="best").set_draggable(True)
    
    ax2.set_title('Inter-ELM transport profiles', loc='right', fontsize = 12, color=colors[int(len(out)//2)])
    ax2.plot(rhop_transp,D_BP,linewidth=1,linestyle='--',label='Banana-plateau')
    ax2.plot(rhop_transp,D_PS,linewidth=1,linestyle='--',label='Pfirsch–Schlüter')
    ax2.plot(rhop_transp,D_CL,linewidth=1,linestyle='--',label='Classical')
    ax2.plot(rhop_transp,D_tot,linewidth=2,linestyle='--',label='Tot. neoclassical $D$',color='black')
    for i in range(0,len(out)):
        ax2.plot(asim[i].rhop_grid,D_inter_ELM[i],linewidth=2,color=colors[i])
    ax2.set_ylim(Dmin if Dmin is not None else None, Dmax if Dmax is not None else None)
    ax2.set_yscale('log')
    ax2.tick_params(axis='x',labelbottom=False)
    ax2.set_ylabel('$D$ [$\mathrm{{m}}$$^{{2}}$/$s$]')
    ax2.legend(loc="best").set_draggable(True)

    ax3.plot(rhop_transp,v_BP,linewidth=1,linestyle='--',label='Banana-plateau')
    ax3.plot(rhop_transp,v_PS,linewidth=1,linestyle='--',label='Pfirsch–Schlüter')
    ax3.plot(rhop_transp,v_CL,linewidth=1,linestyle='--',label='Classical')
    ax3.plot(rhop_transp,v_tot,linewidth=2,linestyle='--',label='Tot. neoclassical $v$',color='black')
    for i in range(0,len(out)):
        ax3.plot(asim[i].rhop_grid,v_inter_ELM[i],linewidth=2,color=colors[i])
    ax3.set_ylim(vmin if vmin is not None else None, vmax if vmax is not None else None)
    ax3.set_ylabel('$v$ [$\mathrm{{m}}$/$s$]')
    ax3.legend(loc="best").set_draggable(True)
    
    ax4.set_title(f'{round((time_intra_ELM-time_inter_ELM)*1000,1)} ms after ELM crash', loc='right', fontsize = 12)
    ax4.errorbar(rhop_exp,n_imp_exp,yerr=n_imp_unc_exp,ecolor=light_blue,linewidth=1,fmt=' ',zorder=0)
    ax4.scatter(rhop_exp,n_imp_exp,color=blue,zorder=1,label='Experiment')
    for i in range(0,len(out)):
        ax4.plot(asim[i].rhop_grid,nimp_intra_ELM[i],linewidth=2,color=colors[i])
    ax4.set_ylim(0,None)
    ax4.tick_params(axis='x',labelbottom=False)
    ax4.set_ylabel(f'$n_{{{namelists[0]["imp"]}^{{+}}}}$ [$\mathrm{{cm}}$$^{{-3}}$]')
    ax4.legend(loc="best").set_draggable(True)
    
    ax5.set_title('Intra-ELM transport profiles', loc='right', fontsize = 12, color=colors[int(len(out)//2)])
    ax5.plot(asim[0].rhop_grid,D_intra_ELM,linewidth=2,color=colors[int(len(out)//2)])
    ax5.set_ylim(Dmin if Dmin is not None else None, Dmax if Dmax is not None else None)
    ax5.set_yscale('log')
    ax5.tick_params(axis='x',labelbottom=False)
    ax5.set_ylabel('$D$ [$\mathrm{{m}}$$^{{2}}$/$s$]')
    
    ax6.plot(asim[0].rhop_grid,v_intra_ELM,linewidth=2,color=colors[int(len(out)//2)])
    ax6.set_ylim(vmin if vmin is not None else None, vmax if vmax is not None else None)
    ax6.set_ylabel('$v$ [$\mathrm{{m}}$/$s$]')
    
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
    
    cmap = plt.cm.viridis
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', colors, cmap.N)
    bounds = np.linspace(0, 1, len(out)+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ticks = np.zeros(len(out))
    i = 0
    tick = 0
    while i < len(out):
        ticks[i] = tick + (1/(len(out)))/2
        tick = ticks[i] + (1/(len(out)))/2
        i = i+1
    cb = mpl.colorbar.ColorbarBase(ax7, cmap=cmap, norm=norm,
        spacing='proportional', ticks=ticks, boundaries=bounds, format='%1i')
    cb.ax.set_yticklabels(labels)
    
    plt.tight_layout()
    
    print('Transport coefficients plots prepared.')
    