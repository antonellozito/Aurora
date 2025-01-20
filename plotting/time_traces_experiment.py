import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np, copy
import pandas as pd
import matplotlib.gridspec as mplgs
import matplotlib.widgets as mplw
from matplotlib.cm import ScalarMappable
from scipy.interpolate import interp1d
import bottleneck as bn
import itertools
from aurora import grids_utils
from aurora import plot_tools


def nmain(shot,exp_data,namelist,asim,out,delay_ELM_onset,tmin=None,tmax=None):
        
    rhop_exp = exp_data[f'{namelist["main_element"]}_density_plasma_ELM']['rhop']
    time_ELM = exp_data[f'{namelist["main_element"]}_density_plasma_ELM']['time_ELM']
    nD_exp = exp_data[f'{namelist["main_element"]}_density_plasma_ELM'][f'n_{namelist["main_element"]}']
    nD_unc_exp = exp_data[f'{namelist["main_element"]}_density_plasma_ELM'][f'n_{namelist["main_element"]}_unc']
    
    time_ELM = time_ELM + abs(time_ELM[0]) + (time_ELM[1]-time_ELM[0])
    time_ELM = time_ELM.round(decimals = 4)
    
    idx = np.argmin(np.abs(time_ELM - delay_ELM_onset))
    nD_exp = np.roll(nD_exp,-idx,axis = 1)
    nD_unc_exp = np.roll(nD_unc_exp,-idx,axis = 1)
    
    time_end = asim.time_out[-1].round(decimals = 4)
    temp = round(time_end/(time_ELM[1]-time_ELM[0]))
    time_grid_exp = np.linspace(time_ELM[0],time_end,temp)
    
    num_repetitions = round(temp/len(time_ELM))
    nD_exp = np.tile(nD_exp,num_repetitions)
    nD_unc_exp = np.tile(nD_unc_exp,num_repetitions)
    
    nD_exp = np.reshape(nD_exp,(1,nD_exp.shape[0],nD_exp.shape[1]))
    nD_unc_exp = np.reshape(nD_unc_exp,(1,nD_unc_exp.shape[0],nD_unc_exp.shape[1]))
    
    f_nD_exp = interp1d(rhop_exp, nD_exp, axis=1, kind="linear", fill_value="extrapolate", assume_sorted=True)
    f_unc_nD_exp = interp1d(rhop_exp, nD_unc_exp, axis=1, kind="linear", fill_value="extrapolate", assume_sorted=True)
    
    nD_exp = f_nD_exp(asim.rhop_grid)
    nD_unc_exp = f_unc_nD_exp(asim.rhop_grid) 

    slider_plot(
                asim.time_out,
                asim.rhop_grid,
                out['nz'].transpose(1, 0, 2),
                time_grid_exp,
                nD_exp,
                nD_unc_exp,
                puff = False,
                tmin = tmin,
                tmax = tmax,
                xlabel="$\mathrm{time}$ [$\mathrm{s}$]",
                ylabel=r'$\rho_p$',
                zlabel=f'$n_{{{namelist["main_element"]}}}$ [$\mathrm{{cm}}$$^{{-3}}$]',
                plot_title = 'Main ion time traces',
                labels=[str(i) for i in np.arange(0, out['nz'].shape[1])],
                x_line=None,
                namelist=namelist,
            )
    
    print('Simulated vs experimental main ion density time trace slider plot prepared.')
    
    return



def nimp(shot,exp_data,namelist,asim,out,interval,puff=False,time_exp_shift=0.0,tmin=None,tmax=None,errorbars=False):
    
    time_average, sim_average_profiles_full = plot_tools.time_average_profiles(asim.namelist['timing'], asim.time_out, out['nz'], interval = interval)
    
    f_sim_average_profiles = interp1d(asim.rhop_grid, sim_average_profiles_full, axis=0, kind="linear", fill_value="extrapolate", assume_sorted=False)
        
    rhop_exp = exp_data[f'{namelist["imp"]}_density_plasma']['rhop'].transpose()[:,0]
    n_imp_exp = exp_data[f'{namelist["imp"]}_density_plasma'][f'n_{namelist["imp"]}'].transpose()
    n_imp_unc_exp = exp_data[f'{namelist["imp"]}_density_plasma'][f'n_{namelist["imp"]}_unc'].transpose()
    
    sim_average_profiles = f_sim_average_profiles(rhop_exp)
    
    if "time" in exp_data[f'{namelist["imp"]}_density_plasma']:
        
        time_exp = exp_data[f'{namelist["imp"]}_density_plasma']['time'].transpose() + time_exp_shift
        
        f_n_imp_exp = interp1d(time_exp, n_imp_exp, axis=1, kind="linear", fill_value="extrapolate", assume_sorted=False)
        f_unc_n_imp_exp = interp1d(time_exp, n_imp_unc_exp, axis=1, kind="linear", fill_value="extrapolate", assume_sorted=False)
        
        n_imp_exp = f_n_imp_exp(time_average)
        n_imp_unc_exp = f_unc_n_imp_exp(time_average)
        
        idx_exp_max = np.argmin(np.abs(time_average - 8.2))
        
        n_imp_exp = n_imp_exp.reshape(1,n_imp_exp.shape[0],n_imp_exp.shape[1])
        n_imp_unc_exp = n_imp_unc_exp.reshape(1,n_imp_unc_exp.shape[0],n_imp_unc_exp.shape[1])
        
        slider_plot(
                    time_average,
                    rhop_exp,
                    sim_average_profiles.transpose(1, 0, 2),
                    (time_average)[0:idx_exp_max],
                    n_imp_exp[:,:,0:idx_exp_max],
                    n_imp_unc_exp[:,:,0:idx_exp_max],
                    puff = puff,
                    tmin = tmin,
                    tmax = tmax,
                    xlabel="$\mathrm{time}$ [$\mathrm{s}$]",
                    ylabel=r'$\rho_p$',
                    zlabel=f'$n_{{{namelist["imp"]}^{{+}}}}$ [$\mathrm{{cm}}$$^{{-3}}$]',
                    plot_title = f'{namelist["imp"]} time traces',
                    labels=[str(i) for i in np.arange(0, out['nz'].shape[1])],
                    x_line=None,
                    namelist=namelist,
                )
    
    else:
    
        pass
    
    print('Simulated vs experimental impurity ion density time trace slider plot prepared.')
    
    return



def nimp_average(shot,exp_data,namelist,asim,out,interval,puff=False,time_exp_shift=0.0,tmin=None,tmax=None):
        
    rhop_exp = exp_data[f'{namelist["imp"]}_density_plasma']['rhop_extrap'].transpose()
    n_imp_exp = exp_data[f'{namelist["imp"]}_density_plasma'][f'n_{namelist["imp"]}_extrap'].transpose()
    n_imp_unc_exp = exp_data[f'{namelist["imp"]}_density_plasma'][f'n_{namelist["imp"]}_unc_extrap'].transpose()
        
    nz = out['nz'].transpose(2, 1, 0)
        
    n_sim = grids_utils.vol_int(nz[:,-1,:],asim.rvol_grid,asim.pro_grid,asim.Raxis_cm,rvol_max=asim.rvol_lcfs)/asim.core_vol
    
    time_average, n_sim_average = plot_tools.time_average_reservoirs(asim.namelist['timing'], asim.time_out, n_sim, interval = interval)
        
    if "time" in exp_data[f'{namelist["imp"]}_density_plasma']:
        
        time_exp = exp_data[f'{namelist["imp"]}_density_plasma']['time'].transpose() + time_exp_shift
        
        f_n_imp_exp = interp1d(time_exp, n_imp_exp, axis=1, kind="linear", fill_value="extrapolate", assume_sorted=False)
        f_unc_n_imp_exp = interp1d(time_exp, n_imp_unc_exp, axis=1, kind="linear", fill_value="extrapolate", assume_sorted=False)
        
        n_imp_exp = f_n_imp_exp(time_average)
        n_imp_unc_exp = f_unc_n_imp_exp(time_average)
        
        idx_exp_max = np.argmin(np.abs(time_average - 8.2))
        
        n_imp_exp_extrap = np.zeros((len(asim.rhop_grid),n_imp_exp.shape[1]))
        n_imp_unc_exp_extrap = np.zeros((len(asim.rhop_grid),n_imp_unc_exp.shape[1]))
        
        for i in range(0,n_imp_exp.shape[1]):
            f = interp1d(rhop_exp,n_imp_exp[:,i],fill_value="extrapolate")
            n_imp_exp_extrap[:,i] = f(asim.rhop_grid)
            f_unc = interp1d(rhop_exp,n_imp_unc_exp[:,i],fill_value="extrapolate")
            n_imp_unc_exp_extrap[:,i] = f_unc(asim.rhop_grid)
            
        n_exp_average = grids_utils.vol_int(n_imp_exp_extrap.transpose()[:,:],asim.rvol_grid,asim.pro_grid,asim.Raxis_cm,rvol_max=asim.rvol_lcfs)/asim.core_vol
        n_exp_max_average = grids_utils.vol_int(n_imp_exp_extrap.transpose()[:,:]+n_imp_unc_exp_extrap.transpose()[:,:],asim.rvol_grid,asim.pro_grid,asim.Raxis_cm,rvol_max=asim.rvol_lcfs)/asim.core_vol
        n_exp_min_average = grids_utils.vol_int(n_imp_exp_extrap.transpose()[:,:]-n_imp_unc_exp_extrap.transpose()[:,:],asim.rvol_grid,asim.pro_grid,asim.Raxis_cm,rvol_max=asim.rvol_lcfs)/asim.core_vol
            
        colors = plot_tools.load_color_codes_reservoirs()
        blue,light_blue,green,light_green,grey,light_grey,red,light_red = colors
        
        fig, ax1 = plt.subplots(figsize=(7,5))
        fig.suptitle(f'Average {namelist["imp"]} density in the core')
        ax1.set_xlabel('$\mathrm{time}$ [$\mathrm{s}$]')
        ax1.set_ylabel(f'$\overline{{n}}_{{{namelist["imp"]}^{{+}}}}$ [$\mathrm{{cm}}$$^{{-3}}$]')
        if puff:
            ax1.axvspan(namelist["src_step_times"][1],
                    namelist["src_step_times"][2],
                    color="gainsboro", zorder = 0)
        ax1.fill_between((time_average)[0:idx_exp_max],n_exp_min_average[0:idx_exp_max],n_exp_max_average[0:idx_exp_max],color=blue,edgecolor="none",alpha=0.15)
        ax1.scatter((time_average)[0:idx_exp_max], n_exp_average[0:idx_exp_max], label="Experiment", color = blue)
        ax1.plot(time_average, n_sim_average, linewidth = 3.5, label="Simulation", color = 'black')
        if tmin is not None:
            ax1.set_xlim(tmin,tmax)
        else:
            ax1.set_xlim((time_average)[0],(time_average)[-1])
        ax1.set_ylim(0,np.max(n_sim_average)*1.15)
        ax1.legend(loc="best").set_draggable(True)
    
    else:
    
        pass
    
    print('Simulated vs experimental impurity ion density ELM-averaged time trace plot prepared.')
    
    return



def nimp_average_sensitivity_analysis(shot,exp_data,namelists,asim,out,interval,labels,puff=False,time_exp_shift=0.0,tmin=None,tmax=None,plot_experiment=True):
        
    rhop_exp = exp_data[f'{namelists[0]["imp"]}_density_plasma']['rhop_extrap'].transpose()
    n_imp_exp = exp_data[f'{namelists[0]["imp"]}_density_plasma'][f'n_{namelists[0]["imp"]}_extrap'].transpose()
    n_imp_unc_exp = exp_data[f'{namelists[0]["imp"]}_density_plasma'][f'n_{namelists[0]["imp"]}_unc_extrap'].transpose()
        
    nz = ()
    for i in range(0,len(out)):
        nz = nz + (out[i]['nz'].transpose(2, 1, 0),)
        
    n_sim = ()
    for i in range(0,len(out)):
        n_sim = n_sim + (grids_utils.vol_int(nz[i][:,-1,:],asim[i].rvol_grid,asim[i].pro_grid,asim[i].Raxis_cm,rvol_max=asim[i].rvol_lcfs)/asim[i].core_vol,)
    
    time_average = ()
    n_sim_average = ()
    for i in range(0,len(out)):
        time_average = time_average + (plot_tools.time_average_reservoirs(asim[i].namelist['timing'], asim[i].time_out, n_sim[i], interval = interval)[0],)
        n_sim_average = n_sim_average + (plot_tools.time_average_reservoirs(asim[i].namelist['timing'], asim[i].time_out, n_sim[i], interval = interval)[1],)
        
    if "time" in exp_data[f'{namelists[0]["imp"]}_density_plasma']:
        
        time_exp = exp_data[f'{namelists[0]["imp"]}_density_plasma']['time'].transpose() + time_exp_shift
        
        f_n_imp_exp = interp1d(time_exp, n_imp_exp, axis=1, kind="linear", fill_value="extrapolate", assume_sorted=False)
        f_unc_n_imp_exp = interp1d(time_exp, n_imp_unc_exp, axis=1, kind="linear", fill_value="extrapolate", assume_sorted=False)
        
        n_imp_exp = f_n_imp_exp(time_average[0])
        n_imp_unc_exp = f_unc_n_imp_exp(time_average[0])
        
        idx_exp_max = np.argmin(np.abs((time_average[0]) - 8.2))
        
        n_imp_exp_extrap = np.zeros((len(asim[0].rhop_grid),n_imp_exp.shape[1]))
        n_imp_unc_exp_extrap = np.zeros((len(asim[0].rhop_grid),n_imp_unc_exp.shape[1]))
        
        for i in range(0,n_imp_exp.shape[1]):
            f = interp1d(rhop_exp,n_imp_exp[:,i],fill_value="extrapolate")
            n_imp_exp_extrap[:,i] = f(asim[0].rhop_grid)
            f_unc = interp1d(rhop_exp,n_imp_unc_exp[:,i],fill_value="extrapolate")
            n_imp_unc_exp_extrap[:,i] = f_unc(asim[0].rhop_grid)
            
        n_exp_average = grids_utils.vol_int(n_imp_exp_extrap.transpose()[:,:],asim[0].rvol_grid,asim[0].pro_grid,asim[0].Raxis_cm,rvol_max=asim[0].rvol_lcfs)/asim[0].core_vol
        n_exp_max_average = grids_utils.vol_int(n_imp_exp_extrap.transpose()[:,:]+n_imp_unc_exp_extrap.transpose()[:,:],asim[0].rvol_grid,asim[0].pro_grid,asim[0].Raxis_cm,rvol_max=asim[0].rvol_lcfs)/asim[0].core_vol
        n_exp_min_average = grids_utils.vol_int(n_imp_exp_extrap.transpose()[:,:]-n_imp_unc_exp_extrap.transpose()[:,:],asim[0].rvol_grid,asim[0].pro_grid,asim[0].Raxis_cm,rvol_max=asim[0].rvol_lcfs)/asim[0].core_vol
            
        colors = plot_tools.load_color_codes_reservoirs()
        blue,light_blue,green,light_green,grey,light_grey,red,light_red = colors
        
        color_indices = np.zeros(len(out))
        colors = ()
        cmap = mpl.colormaps['viridis']
        for i in range(0,len(out)):
            color_indices[i] = (1/(len(out)+1)) * (i+1)
            colors = colors + (cmap(color_indices[i]),) 
        
        fig, ax1 = plt.subplots(figsize=(7,5))
        fig.suptitle(f'Average {namelists[0]["imp"]} density in the core')
        ax1.set_xlabel('$\mathrm{time}$ [$\mathrm{s}$]')
        ax1.set_ylabel(f'$\overline{{n}}_{{{namelists[0]["imp"]}^{{+}}}}$ [$\mathrm{{cm}}$$^{{-3}}$]')
        if puff:
            ax1.axvspan(namelists[0]["src_step_times"][1],
                    namelists[0]["src_step_times"][2],
                    color="gainsboro", zorder = 0)
        if plot_experiment:
            ax1.fill_between((time_average[0])[0:idx_exp_max],n_exp_min_average[0:idx_exp_max],n_exp_max_average[0:idx_exp_max],color='black',edgecolor="none",alpha=0.15)
            ax1.scatter((time_average[0])[0:idx_exp_max], n_exp_average[0:idx_exp_max], label="Experiment", color = 'black')
        for i in range(0,len(out)):
            ax1.plot(time_average[i], n_sim_average[i], linewidth = 3.5, label=labels[i], color = colors[i])
        if tmin is not None:
            ax1.set_xlim(tmin,tmax)
        else:
            ax1.set_xlim((time_average[0])[0],(time_average[0])[-1])
        ax1.set_ylim(0,np.max(n_sim_average)*1.15)
        ax1.legend(loc="best").set_draggable(True)
    
    else:
    
        pass
    
    print('Simulated vs experimental impurity ion density ELM-averaged time trace plot prepared.')
    
    return



def pressures(shot,exp_data,namelist,asim,reservoirs,interval,puff=False,time_exp_shift=0.0,tmin=None,tmax=None):
    
    time_average, n_div_av_sim = plot_tools.time_average_reservoirs(asim.namelist['timing'], asim.time_out, reservoirs["particle_density_in_divertor"], interval = interval)
    time_average, n_pump_av_sim = plot_tools.time_average_reservoirs(asim.namelist['timing'], asim.time_out, reservoirs["particle_density_in_pump"], interval = interval)
        
    time_exp = exp_data[f'{namelist["imp"]}_partial_pressures']['time'].transpose() + time_exp_shift
    
    n_div_exp = exp_data[f'{namelist["imp"]}_partial_pressures']['n_div'].transpose()
    n_pump_exp = exp_data[f'{namelist["imp"]}_partial_pressures']['n_pump'].transpose()
    
    idx_exp_max = np.argmin(np.abs(time_exp - 8.2))
    
    colors = plot_tools.load_color_codes_reservoirs()
    blue,light_blue,green,light_green,grey,light_grey,red,light_red = colors
    
    fig, ax1 = plt.subplots(figsize=(7,5))
    fig.suptitle(f'{namelist["imp"]} partial pressure (pump reservoir)')
    ax1.set_xlabel('$\mathrm{time}$ [$\mathrm{s}$]')
    ax1.set_ylabel(f'$n_{{{namelist["imp"]}^0}}$ [$\mathrm{{cm}}$$^{{-3}}$]')
    if puff:
        ax1.axvspan(namelist["src_step_times"][1],
                namelist["src_step_times"][2],
                color="gainsboro", zorder = 0)
    ax1.scatter(time_exp[0:idx_exp_max], n_pump_exp[0:idx_exp_max], label="Experiment", color = green)
    ax1.plot(time_average, n_pump_av_sim, label="Simulation", linewidth = 3.5, color = 'black')
    if tmin is not None:
        ax1.set_xlim(tmin,tmax)
    else:
        ax1.set_xlim(time_average[0],time_average[-1])
    ax1.set_ylim(0,np.max(n_pump_av_sim)*1.15)
    ax1.legend(loc="best").set_draggable(True)
    
    print('Simulated vs experimental impurity pressure time trace plot prepared.')
    
    
    
def all_traces(shot,exp_data,namelist,asim,out,reservoirs,interval,puff=False,time_exp_shift=0.0,tmin=None,tmax=None):
        
    # plasma
    
    rhop_exp = exp_data[f'{namelist["imp"]}_density_plasma']['rhop_extrap'].transpose()
    n_imp_exp = exp_data[f'{namelist["imp"]}_density_plasma'][f'n_{namelist["imp"]}_extrap'].transpose()
    n_imp_unc_exp = exp_data[f'{namelist["imp"]}_density_plasma'][f'n_{namelist["imp"]}_unc_extrap'].transpose()
        
    nz = out['nz'].transpose(2, 1, 0)
        
    n_sim = grids_utils.vol_int(nz[:,-1,:],asim.rvol_grid,asim.pro_grid,asim.Raxis_cm,rvol_max=asim.rvol_lcfs)/asim.core_vol
    
    time_average, n_sim_average = plot_tools.time_average_reservoirs(asim.namelist['timing'], asim.time_out, n_sim, interval = interval)
                
    time_exp_plasma = exp_data[f'{namelist["imp"]}_density_plasma']['time'].transpose() + time_exp_shift
    
    f_n_imp_exp = interp1d(time_exp_plasma, n_imp_exp, axis=1, kind="linear", fill_value="extrapolate", assume_sorted=False)
    f_unc_n_imp_exp = interp1d(time_exp_plasma, n_imp_unc_exp, axis=1, kind="linear", fill_value="extrapolate", assume_sorted=False)
    
    n_imp_exp = f_n_imp_exp(time_average)
    n_imp_unc_exp = f_unc_n_imp_exp(time_average)
        
    n_imp_exp_extrap = np.zeros((len(asim.rhop_grid),n_imp_exp.shape[1]))
    n_imp_unc_exp_extrap = np.zeros((len(asim.rhop_grid),n_imp_unc_exp.shape[1]))
    
    for i in range(0,n_imp_exp.shape[1]):
        f = interp1d(rhop_exp,n_imp_exp[:,i],fill_value="extrapolate")
        n_imp_exp_extrap[:,i] = f(asim.rhop_grid)
        f_unc = interp1d(rhop_exp,n_imp_unc_exp[:,i],fill_value="extrapolate")
        n_imp_unc_exp_extrap[:,i] = f_unc(asim.rhop_grid)
        
    n_exp_average = grids_utils.vol_int(n_imp_exp_extrap.transpose()[:,:],asim.rvol_grid,asim.pro_grid,asim.Raxis_cm,rvol_max=asim.rvol_lcfs)/asim.core_vol
    n_exp_max_average = grids_utils.vol_int(n_imp_exp_extrap.transpose()[:,:]+n_imp_unc_exp_extrap.transpose()[:,:],asim.rvol_grid,asim.pro_grid,asim.Raxis_cm,rvol_max=asim.rvol_lcfs)/asim.core_vol
    n_exp_min_average = grids_utils.vol_int(n_imp_exp_extrap.transpose()[:,:]-n_imp_unc_exp_extrap.transpose()[:,:],asim.rvol_grid,asim.pro_grid,asim.Raxis_cm,rvol_max=asim.rvol_lcfs)/asim.core_vol
        
    # wall
    
    _, n_mainwall_av_sim = plot_tools.time_average_reservoirs(asim.namelist['timing'], asim.time_out, reservoirs["particle_density_retained_at_main_wall"], interval = interval)
    _, n_divwall_av_sim = plot_tools.time_average_reservoirs(asim.namelist['timing'], asim.time_out, reservoirs["particle_density_retained_at_div_wall"], interval = interval)
    
    # neutrals
    
    _, n_div_av_sim = plot_tools.time_average_reservoirs(asim.namelist['timing'], asim.time_out, reservoirs["particle_density_in_divertor"], interval = interval)
    _, n_pump_av_sim = plot_tools.time_average_reservoirs(asim.namelist['timing'], asim.time_out, reservoirs["particle_density_in_pump"], interval = interval)
        
    time_exp_neutrals = exp_data[f'{namelist["imp"]}_partial_pressures']['time'].transpose() + time_exp_shift
    
    n_pump_exp = exp_data[f'{namelist["imp"]}_partial_pressures']['n_pump'].transpose()
    
    idx_exp_max_neutrals = np.argmin(np.abs(time_exp_neutrals - 8.2))
    
    # plot
    
    colors = plot_tools.load_color_codes_reservoirs()
    blue,light_blue,green,light_green,grey,light_grey,red,light_red = colors
    
    idx_exp_max = np.argmin(np.abs(time_average - 8.2))
    
    #fig, ax1 = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(6,12))
    
    fig = plt.figure()
    fig.set_size_inches(6.5, 12, forward=True)
    ax1 = plt.subplot2grid((32, 18), (0, 0), rowspan=7, colspan=16, fig=fig)
    ax2 = plt.subplot2grid((32, 18), (8, 0), rowspan=7, colspan=16, fig=fig)
    ax3 = plt.subplot2grid((32, 18), (16, 0), rowspan=7, colspan=16, fig=fig)
    ax4 = plt.subplot2grid((32, 18), (24, 0), rowspan=7, colspan=16, fig=fig)
    
    fig.suptitle('Reservoirs time traces', fontsize = 16)
    
    ax1.set_ylabel(f'$\overline{{n}}_{{{namelist["imp"]}^{{+}}}}$ [$\mathrm{{cm}}$$^{{-3}}$]')
    if puff:
        ax1.axvspan(namelist["src_step_times"][1],
                namelist["src_step_times"][2],
                color="gainsboro", zorder = 0)
    ax1.fill_between(time_average[0:idx_exp_max],n_exp_min_average[0:idx_exp_max],n_exp_max_average[0:idx_exp_max],color='black',edgecolor="none",alpha=0.15)
    ax1.scatter(time_average[0:idx_exp_max], n_exp_average[0:idx_exp_max], color = 'black')
    ax1.plot(time_average, n_sim_average, linewidth = 3.5, color = blue)
    ax1.set_ylim(0,np.max(n_sim_average)*1.2)
    ax1.tick_params(axis='x',labelbottom=False)
    ax1.text(0.98,0.95,'Plasma',ha='right',va='top',transform=ax1.transAxes,fontsize=12)
    
    ax2.set_ylabel(f'$\sigma_{{{namelist["imp"]},impl.}}$ [$\mathrm{{cm}}$$^{{-2}}$]')
    if puff:
        ax2.axvspan(namelist["src_step_times"][1],
                namelist["src_step_times"][2],
                color="gainsboro", zorder = 0)
    ax2.plot(time_average, n_mainwall_av_sim, linewidth = 3.5, color = light_grey)
    ax2.set_ylim(0,np.max(n_mainwall_av_sim)*1.2)
    ax2right = ax2.twinx()
    ax2right.set_ylim(0,np.max(n_mainwall_av_sim)*1.2/(namelist['full_PWI']['n_main_wall_sat']/1e4))
    ax2right.set_ylabel('$\mathrm{saturation}$ $\mathrm{level}$')
    ax2.tick_params(axis='x',labelbottom=False)
    ax2.text(0.98,0.95,'Main wall storage',ha='right',va='top',transform=ax2.transAxes,fontsize=12)
    
    ax3.set_ylabel(f'$\sigma_{{{namelist["imp"]},impl.}}$ [$\mathrm{{cm}}$$^{{-2}}$]')
    if puff:
        ax3.axvspan(namelist["src_step_times"][1],
                namelist["src_step_times"][2],
                color="gainsboro", zorder = 0)
    ax3.plot(time_average, n_divwall_av_sim, linewidth = 3.5, color = grey)
    ax3.set_ylim(0,np.max(n_divwall_av_sim)*1.2)
    ax3right = ax3.twinx()
    ax3right.set_ylim(0,np.max(n_divwall_av_sim)*1.2/(namelist['full_PWI']['n_div_wall_sat']/1e4))
    ax3right.set_ylabel('$\mathrm{saturation}$ $\mathrm{level}$')
    ax3.tick_params(axis='x',labelbottom=False)
    ax3.text(0.98,0.95,'Divertor wall storage',ha='right',va='top',transform=ax3.transAxes,fontsize=12)
    
    ax4.set_ylabel(f'$n_{{{namelist["imp"]}^0}}$ [$\mathrm{{cm}}$$^{{-3}}$]')
    if puff:
        ax4.axvspan(namelist["src_step_times"][1],
                namelist["src_step_times"][2],
                color="gainsboro", zorder = 0)
    ax4.scatter(time_exp_neutrals[0:idx_exp_max_neutrals], n_pump_exp[0:idx_exp_max_neutrals], color = 'black')
    ax4.plot(time_average, n_pump_av_sim, linewidth = 3.5, color = green)
    ax4.set_ylim(0,np.max(n_pump_av_sim)*1.2)
    ax4.text(0.98,0.95,'Exhaust gas (pump chamber)',ha='right',va='top',transform=ax4.transAxes,fontsize=12)
    
    ax2.sharex(ax1)
    ax3.sharex(ax1)
    ax4.sharex(ax1)
    if tmin is not None:
        ax4.set_xlim(tmin,tmax)
    else:
        ax4.set_xlim((time_average)[0],(time_average)[-1])
    ax4.set_xlabel('$\mathrm{time}$ [$\mathrm{s}$]')
    
    plt.tight_layout()
    
    print('Simulated vs experimental reservoirs impurity densities time traces plots prepared.')
   
    
   
def all_traces_sensitivity_analysis(shot,exp_data,namelists,asim,out,reservoirs,interval,labels,puff=False,time_exp_shift=0.0,tmin=None,tmax=None,plot_experiment=True):
        
    # plasma
    
    rhop_exp = exp_data[f'{namelists[0]["imp"]}_density_plasma']['rhop_extrap'].transpose()
    n_imp_exp = exp_data[f'{namelists[0]["imp"]}_density_plasma'][f'n_{namelists[0]["imp"]}_extrap'].transpose()
    n_imp_unc_exp = exp_data[f'{namelists[0]["imp"]}_density_plasma'][f'n_{namelists[0]["imp"]}_unc_extrap'].transpose()
    
    nz = ()
    n_sim = ()
    time_average = ()
    n_sim_average = ()
    
    for i in range(0,len(out)):  
        nz = nz + (out[i]['nz'].transpose(2, 1, 0),)
        n_sim = n_sim + (grids_utils.vol_int(nz[i][:,-1,:],asim[i].rvol_grid,asim[i].pro_grid,asim[i].Raxis_cm,rvol_max=asim[i].rvol_lcfs)/asim[i].core_vol,)
        time_average = time_average + (plot_tools.time_average_reservoirs(asim[i].namelist['timing'], asim[i].time_out, n_sim[i], interval = interval)[0],)
        n_sim_average = n_sim_average + (plot_tools.time_average_reservoirs(asim[i].namelist['timing'], asim[i].time_out, n_sim[i], interval = interval)[1],)
                
    time_exp_plasma = exp_data[f'{namelists[0]["imp"]}_density_plasma']['time'].transpose() + time_exp_shift
    
    f_n_imp_exp = interp1d(time_exp_plasma, n_imp_exp, axis=1, kind="linear", fill_value="extrapolate", assume_sorted=False)
    f_unc_n_imp_exp = interp1d(time_exp_plasma, n_imp_unc_exp, axis=1, kind="linear", fill_value="extrapolate", assume_sorted=False)
    
    n_imp_exp = f_n_imp_exp(time_average[0])
    n_imp_unc_exp = f_unc_n_imp_exp(time_average[0])
    
    n_imp_exp_extrap = np.zeros((len(asim[0].rhop_grid),n_imp_exp.shape[1]))
    n_imp_unc_exp_extrap = np.zeros((len(asim[0].rhop_grid),n_imp_unc_exp.shape[1]))
    
    for i in range(0,n_imp_exp.shape[1]):
        f = interp1d(rhop_exp,n_imp_exp[:,i],fill_value="extrapolate")
        n_imp_exp_extrap[:,i] = f(asim[0].rhop_grid)
        f_unc = interp1d(rhop_exp,n_imp_unc_exp[:,i],fill_value="extrapolate")
        n_imp_unc_exp_extrap[:,i] = f_unc(asim[0].rhop_grid)
        
    n_exp_average = grids_utils.vol_int(n_imp_exp_extrap.transpose()[:,:],asim[0].rvol_grid,asim[0].pro_grid,asim[0].Raxis_cm,rvol_max=asim[0].rvol_lcfs)/asim[0].core_vol
    n_exp_max_average = grids_utils.vol_int(n_imp_exp_extrap.transpose()[:,:]+n_imp_unc_exp_extrap.transpose()[:,:],asim[0].rvol_grid,asim[0].pro_grid,asim[0].Raxis_cm,rvol_max=asim[0].rvol_lcfs)/asim[0].core_vol
    n_exp_min_average = grids_utils.vol_int(n_imp_exp_extrap.transpose()[:,:]-n_imp_unc_exp_extrap.transpose()[:,:],asim[0].rvol_grid,asim[0].pro_grid,asim[0].Raxis_cm,rvol_max=asim[0].rvol_lcfs)/asim[0].core_vol
        
    # wall
    
    n_mainwall_av_sim = ()
    n_divwall_av_sim = ()
    
    for i in range(0,len(out)):
        n_mainwall_av_sim = n_mainwall_av_sim + (plot_tools.time_average_reservoirs(asim[i].namelist['timing'], asim[i].time_out, reservoirs[i]["particle_density_retained_at_main_wall"], interval = interval)[1],)
        n_divwall_av_sim = n_divwall_av_sim + (plot_tools.time_average_reservoirs(asim[i].namelist['timing'], asim[i].time_out, reservoirs[i]["particle_density_retained_at_div_wall"], interval = interval)[1],)
    
    # neutrals
    
    n_div_av_sim = ()
    n_pump_av_sim = ()
    
    for i in range(0,len(out)):
        n_div_av_sim = n_div_av_sim + (plot_tools.time_average_reservoirs(asim[i].namelist['timing'], asim[i].time_out, reservoirs[i]["particle_density_in_divertor"], interval = interval)[1],)
        n_pump_av_sim = n_pump_av_sim + (plot_tools.time_average_reservoirs(asim[i].namelist['timing'], asim[i].time_out, reservoirs[i]["particle_density_in_pump"], interval = interval)[1],)
            
    time_exp_neutrals = exp_data[f'{namelists[0]["imp"]}_partial_pressures']['time'].transpose() + time_exp_shift
    
    n_pump_exp = exp_data[f'{namelists[0]["imp"]}_partial_pressures']['n_pump'].transpose()
    
    idx_exp_max_neutrals = np.argmin(np.abs(time_exp_neutrals - 8.2))
    
    # plot
    
    color_indices = np.zeros(len(out))
    colors = ()
    cmap = mpl.colormaps['viridis']
    for i in range(0,len(out)):
        color_indices[i] = (1/(len(out)+1)) * (i+1)
        colors = colors + (cmap(color_indices[i]),) 
    
    time_average = list(time_average)
    for i in range(0,len(out)):
        time_average[i] = time_average[i]
        
    idx_exp_max = np.argmin(np.abs(time_average[0] - 8.2))
    
    #fig = plt.figure(figsize=(6,12))
    
    #fig, ax1 = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(6,12))
   
    fig = plt.figure()
    fig.set_size_inches(9.5, 12, forward=True)
    ax1 = plt.subplot2grid((32, 18), (0, 0), rowspan=7, colspan=11, fig=fig)
    ax2 = plt.subplot2grid((32, 18), (8, 0), rowspan=7, colspan=11, fig=fig)
    ax3 = plt.subplot2grid((32, 18), (16, 0), rowspan=7, colspan=11, fig=fig)
    ax4 = plt.subplot2grid((32, 18), (24, 0), rowspan=7, colspan=11, fig=fig)
    ax5 = plt.subplot2grid((32, 18), (0, 13), rowspan=31, colspan=1, fig=fig)
    
    fig.suptitle('Reservoirs time traces', fontsize = 16)
    
    ax1.set_ylabel(f'$\overline{{n}}_{{{namelists[0]["imp"]}^{{+}}}}$ [$\mathrm{{cm}}$$^{{-3}}$]')
    if puff:
        ax1.axvspan(namelists[0]["src_step_times"][1],
                namelists[0]["src_step_times"][2],
                color="gainsboro", zorder = 0)
    if plot_experiment:
        ax1.fill_between((time_average[0])[0:idx_exp_max],n_exp_min_average[0:idx_exp_max],n_exp_max_average[0:idx_exp_max],color='black',edgecolor="none",alpha=0.15)
        ax1.scatter((time_average[0])[0:idx_exp_max], n_exp_average[0:idx_exp_max], color = 'black')
    for i in range(0,len(out)):
        ax1.plot(time_average[i], n_sim_average[i], linewidth = 3.5, color = colors[i])
    ax1.set_ylim(0,np.max(n_sim_average)*1.2)
    ax1.tick_params(axis='x',labelbottom=False)
    ax1.text(0.98,0.95,'Plasma',ha='right',va='top',transform=ax1.transAxes,fontsize=12)
    
    ax2.set_ylabel(f'$\sigma_{{{namelists[0]["imp"]},impl.}}$ [$\mathrm{{cm}}$$^{{-2}}$]')
    if puff:
        ax2.axvspan(namelists[0]["src_step_times"][1],
                namelists[0]["src_step_times"][2],
                color="gainsboro", zorder = 0)
    for i in range(0,len(out)):
        ax2.plot(time_average[i], n_mainwall_av_sim[i], linewidth = 3.5, color = colors[i])
    ax2.set_ylim(0,np.max(n_mainwall_av_sim)*1.2)
    ax2right = ax2.twinx()
    ax2right.set_ylim(0,np.max(n_mainwall_av_sim)*1.2/(namelists[0]['full_PWI']['n_main_wall_sat']/1e4))
    ax2right.set_ylabel('$\mathrm{saturation}$ $\mathrm{level}$')
    ax2.tick_params(axis='x',labelbottom=False)
    ax2.text(0.98,0.95,'Main wall storage',ha='right',va='top',transform=ax2.transAxes,fontsize=12)
    
    ax3.set_ylabel(f'$\sigma_{{{namelists[0]["imp"]},impl.}}$ [$\mathrm{{cm}}$$^{{-2}}$]')
    if puff:
        ax3.axvspan(namelists[0]["src_step_times"][1],
                namelists[0]["src_step_times"][2],
                color="gainsboro", zorder = 0)
    for i in range(0,len(out)):
        ax3.plot(time_average[i], n_divwall_av_sim[i], linewidth = 3.5, color = colors[i])
    ax3.set_ylim(0,np.max(n_divwall_av_sim)*1.2)
    ax3right = ax3.twinx()
    ax3right.set_ylim(0,np.max(n_divwall_av_sim)*1.2/(namelists[0]['full_PWI']['n_div_wall_sat']/1e4))
    ax3right.set_ylabel('$\mathrm{saturation}$ $\mathrm{level}$')
    ax3.tick_params(axis='x',labelbottom=False)
    ax3.text(0.98,0.95,'Divertor wall storage',ha='right',va='top',transform=ax3.transAxes,fontsize=12)
    
    ax4.set_ylabel(f'$n_{{{namelists[0]["imp"]}^0}}$ [$\mathrm{{cm}}$$^{{-3}}$]')
    if puff:
        ax4.axvspan(namelists[0]["src_step_times"][1],
                namelists[0]["src_step_times"][2],
                color="gainsboro", zorder = 0)
    if plot_experiment:
        ax4.scatter(time_exp_neutrals[0:idx_exp_max_neutrals], n_pump_exp[0:idx_exp_max_neutrals], color = 'black')
    for i in range(0,len(out)):
        #ax4.plot(time_average[i], n_div_av_sim[i], label="Divertor reservoir", linewidth = 3.5, color = green)
        ax4.plot(time_average[i], n_pump_av_sim[i], linewidth = 3.5, color = colors[i])
    ax4.set_ylim(0,np.max(n_pump_av_sim)*1.2)
    ax4.text(0.98,0.95,'Exhaust gas (pump chamber)',ha='right',va='top',transform=ax4.transAxes,fontsize=12)
    
    ax2.sharex(ax1)
    ax3.sharex(ax1)
    ax4.sharex(ax1)
    
    if tmin is not None:
        ax4.set_xlim(tmin,tmax)
    else:
        ax4.set_xlim((time_average[0])[0],(time_average[0])[-1])
    ax4.set_xlabel('$\mathrm{time}$ [$\mathrm{s}$]')
    
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
    cb = mpl.colorbar.ColorbarBase(ax5, cmap=cmap, norm=norm,
        spacing='proportional', ticks=ticks, boundaries=bounds, format='%1i')
    cb.ax.set_yticklabels(labels)
    
    plt.tight_layout()
    
    print('Simulated vs experimental reservoirs impurity densities time traces plots prepared.')
   


def slider_plot(
    x,
    y,
    z,
    x_exp,
    z_exp,
    z_exp_unc,
    tmin,
    tmax,
    puff,
    xlabel,
    ylabel,
    zlabel,
    plot_title,
    labels,
    x_line,
    namelist,
):

    colors = plot_tools.load_color_codes_reservoirs()
    blue,light_blue,green,light_green,grey,light_grey,red,light_red = colors
    
    if labels is None:
        labels = ["" for v in z]

    # make sure not to modify the z array in place
    zz = copy.deepcopy(z)
    zz_exp = copy.deepcopy(z_exp)
    zz_exp_unc = copy.deepcopy(z_exp_unc)

    fig = plt.figure()
    fig.set_size_inches(7, 5, forward=True)

    # separate plot into 3 subgrids
    a_plot = plt.subplot2grid((10, 10), (0, 0), rowspan=8, colspan=8, fig=fig)
    a_legend = plt.subplot2grid((10, 10), (0, 8), rowspan=8, colspan=2, fig=fig)
    a_slider = plt.subplot2grid((10, 10), (9, 0), rowspan=1, colspan=8, fig=fig)

    a_plot.set_xlabel(xlabel)
    a_plot.set_ylabel(zlabel)
    if x_line is not None:
        a_plot.axvline(x_line, c="r", ls=":", lw=0.5)
    
    lim_min = 0
    lim_max = np.max(zz.sum(axis=0))*1.15

    if puff:
        
        puff_area = a_plot.axvspan(
                    namelist["src_step_times"][1],
                    namelist["src_step_times"][2],
                    color="gainsboro",
                    zorder = 0)

    (l_sim,) = a_plot.plot(
            x,
            zz[:, 0, :].sum(axis=0),
            c="k",
            lw=3.5,
        )
    
    _ = a_legend.plot(
            [],
            [],
            c="k",
            lw=3.5,
            label="Simulation",
        )

    l_exp = a_plot.scatter(
            x_exp,
            zz_exp[:, 0, :].sum(axis=0),
            color=blue,
        )
    
    l_exp_unc = a_plot.fill_between(
            x_exp,
            (zz_exp[:, 0, :].sum(axis=0)-zz_exp_unc[:, 0, :].sum(axis=0)),
            (zz_exp[:, 0, :].sum(axis=0)+zz_exp_unc[:, 0, :].sum(axis=0)),
            color=blue,
            edgecolor="none",
            alpha=0.15,
        )
    
    _ = a_legend.scatter(
            [],
            [],
            color=blue,
            label="Experiment",
        )
    
    # if zlim:
    #     lim_min = np.min(zz.sum(axis=0))
    #     lim_max = np.max(zz.sum(axis=0))*1.15

    leg = a_legend.legend(loc="best", fontsize=12).set_draggable(True)
    title = fig.suptitle("")
    a_legend.axis("off")
    a_slider.axis("off")

    def update(dum):

        i = int(slider.val)

        l_sim.set_ydata(zz[:, i, :].sum(axis=0))
        l_exp.set_offsets(np.c_[x_exp,zz_exp[:, i, :].sum(axis=0)])
        
        path = l_exp_unc.get_paths()[0]
        
        v_x = np.hstack([x_exp[0],x_exp,x_exp[-1],x_exp[::-1],x_exp[0]])
        v_y = np.hstack([(zz_exp[:, i, :].sum(axis=0)+zz_exp_unc[:, i, :].sum(axis=0))[0],(zz_exp[:, i, :].sum(axis=0)-zz_exp_unc[:, i, :].sum(axis=0)),(zz_exp[:, i, :].sum(axis=0)-zz_exp_unc[:, i, :].sum(axis=0))[-1],(zz_exp[:, i, :].sum(axis=0)+zz_exp_unc[:, i, :].sum(axis=0))[::-1],(zz_exp[:, i, :].sum(axis=0)+zz_exp_unc[:, i, :].sum(axis=0))[0]])
        vertices = np.vstack([v_x,v_y]).T
        codes = np.array([1]+(2*len(x_exp)+1)*[2]+[79]).astype('uint8')
        
        path.vertices = vertices
        path.codes = codes

        a_plot.relim()
        a_plot.autoscale()
        
        tmin,tmax = get_tlims()
        if tmin is not None:
            a_plot.set_xlim(tmin,tmax)
        else:
            a_plot.set_xlim(x[0],x[-1])
        
        a_plot.set_ylim(lim_min,lim_max)

        if plot_title is not None:
            title.set_text(f"{plot_title}, %s = %.5f" % (ylabel, y[i]) if ylabel else f"{plot_title}, %.5f" % (y[i],))
        else:
            title.set_text("%s = %.5f" % (ylabel, y[i]) if ylabel else "%.5f" % (y[i],))

        fig.canvas.draw()
        
    def get_tlims():
        
        return tmin,tmax

    def arrow_respond(slider, event):
        if event.key == "right":
            slider.set_val(min(slider.val + 1, slider.valmax))
        elif event.key == "left":
            slider.set_val(max(slider.val - 1, slider.valmin))

    slider = mplw.Slider(a_slider, ylabel, 0, len(y) - 1, valinit=0, valfmt="%d")
    slider.on_changed(update)
    update(0)
    fig.canvas.mpl_connect("key_press_event", lambda evt: arrow_respond(slider, evt))
