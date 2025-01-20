import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np, copy
import pandas as pd
import matplotlib.gridspec as mplgs
import matplotlib.widgets as mplw
from matplotlib.cm import ScalarMappable
from matplotlib import animation
from scipy.interpolate import interp1d
import itertools
from aurora import plot_tools

def nmain(shot,exp_data,namelist,asim,out,delay_ELM_onset,tmin=None,tmax=None,xmin=None,xmax=None,errorbars=False,print_animation=False,fps=30,filename=None):
        
    rhop_exp = exp_data[f'{namelist["main_element"]}_density_plasma_ELM']['rhop']
    time_ELM = exp_data[f'{namelist["main_element"]}_density_plasma_ELM']['time_ELM']
    n_D_exp = exp_data[f'{namelist["main_element"]}_density_plasma_ELM'][f'n_{namelist["main_element"]}']
    n_D_unc_exp = exp_data[f'{namelist["main_element"]}_density_plasma_ELM'][f'n_{namelist["main_element"]}_unc']
    
    time_ELM = time_ELM + abs(time_ELM[0]) + (time_ELM[1]-time_ELM[0])
    time_ELM = time_ELM.round(decimals = 4)
    
    idx = np.argmin(np.abs(time_ELM - delay_ELM_onset))
    n_D_exp = np.roll(n_D_exp,-idx,axis = 1)
    n_D_unc_exp = np.roll(n_D_unc_exp,-idx,axis = 1)
    
    time_end = asim.time_out[-1].round(decimals = 4)
    temp = round(time_end/(time_ELM[1]-time_ELM[0]))
    time_grid_exp = np.linspace(time_ELM[0],time_end,temp)
    
    num_repetitions = round(temp/len(time_ELM))
    n_D_exp = np.tile(n_D_exp,num_repetitions)
    n_D_unc_exp = np.tile(n_D_unc_exp,num_repetitions)
    
    n_D_exp = np.reshape(n_D_exp,(1,n_D_exp.shape[0],n_D_exp.shape[1]))
    n_D_unc_exp = np.reshape(n_D_unc_exp,(1,n_D_unc_exp.shape[0],n_D_unc_exp.shape[1]))
    
    f_n_D_exp = interp1d(time_grid_exp, n_D_exp, axis=2, kind="linear", fill_value="extrapolate", assume_sorted=True)
    f_unc_n_D_exp = interp1d(time_grid_exp, n_D_unc_exp, axis=2, kind="linear", fill_value="extrapolate", assume_sorted=True)
    
    n_D_exp = f_n_D_exp(asim.time_out)
    n_D_unc_exp = f_unc_n_D_exp(asim.time_out)
    
    if tmin is not None:
        idx_min = np.argmin(np.abs(np.asarray(asim.time_out) - tmin))
        idx_max = np.argmin(np.abs(np.asarray(asim.time_out) - tmax))
    else:
        idx_min = 0
        idx_max = -1

    slider_plot(
                asim.rhop_grid,
                (asim.time_out)[idx_min:idx_max],
                out['nz'].transpose(1, 0, 2)[:,:,idx_min:idx_max],
                rhop_exp,
                n_D_exp[:,:,idx_min:idx_max],
                n_D_unc_exp[:,:,idx_min:idx_max],
                xmin = xmin,
                xmax = xmax,
                errorbars = errorbars,
                print_animation = print_animation,
                fps = fps,
                filename = filename,
                xlabel=r'$\rho_p$',
                ylabel="time [s]",
                zlabel=f'$n_{{{namelist["main_element"]}}}$ [$\mathrm{{cm}}$$^{{-3}}$]',
                plot_title = 'Main ion density profiles',
                labels=[str(i) for i in np.arange(0, out['nz'].shape[1])],
                x_line=1,
                namelist = namelist,
            )
    
    print('Simulated vs experimental main ion density profile slider plot prepared.')
    
    return



def nimp(shot,exp_data,namelist,asim,out,interval,time_exp_shift=0.0,tmin=None,tmax=None,xmin=None,xmax=None,errorbars=False,print_animation=False,fps=30,filename=None):
    
    time_average, sim_average_profiles = plot_tools.time_average_profiles(asim.namelist['timing'], asim.time_out, out['nz'], interval = interval)
        
    rhop_exp = exp_data[f'{namelist["imp"]}_density_plasma']['rhop'].transpose()[:,0]
    n_imp_exp = exp_data[f'{namelist["imp"]}_density_plasma'][f'n_{namelist["imp"]}'].transpose()
    n_imp_unc_exp = exp_data[f'{namelist["imp"]}_density_plasma'][f'n_{namelist["imp"]}_unc'].transpose()
    
    if "time" in exp_data[f'{namelist["imp"]}_density_plasma']:
        
        time_exp = exp_data[f'{namelist["imp"]}_density_plasma']['time'].transpose() + time_exp_shift
        
        f_n_imp_exp = interp1d(time_exp, n_imp_exp, axis=1, kind="linear", fill_value="extrapolate", assume_sorted=False)
        f_unc_n_imp_exp = interp1d(time_exp, n_imp_unc_exp, axis=1, kind="linear", fill_value="extrapolate", assume_sorted=False)
        
        n_imp_exp = f_n_imp_exp(time_average)
        n_imp_unc_exp = f_unc_n_imp_exp(time_average)
    
    else:
    
        n_imp_exp = np.repeat(n_imp_exp,len(time_average),axis=1)
        n_imp_unc_exp = np.repeat(n_imp_unc_exp,len(time_average),axis=1)
    
    n_imp_exp = n_imp_exp.reshape(1,n_imp_exp.shape[0],n_imp_exp.shape[1])
    n_imp_unc_exp = n_imp_unc_exp.reshape(1,n_imp_unc_exp.shape[0],n_imp_unc_exp.shape[1])

    if tmin is not None:
        idx_min = np.argmin(np.abs(np.asarray(time_average) - tmin))
        idx_max = np.argmin(np.abs(np.asarray(time_average) - tmax))
    else:
        idx_min = 0
        idx_max = -1

    slider_plot(
                asim.rhop_grid,
                (time_average)[idx_min:idx_max],
                sim_average_profiles.transpose(1, 0, 2)[:,:,idx_min:idx_max],
                rhop_exp,
                n_imp_exp[:,:,idx_min:idx_max],
                n_imp_unc_exp[:,:,idx_min:idx_max],
                xmin = xmin,
                xmax = xmax,
                errorbars = errorbars,
                print_animation = print_animation,
                fps = fps,
                filename = filename,
                xlabel=r'$\rho_p$',
                ylabel="time [s]",
                zlabel=f'$n_{{{namelist["imp"]}^{{+}}}}$ [$\mathrm{{cm}}$$^{{-3}}$]',
                plot_title = f'{namelist["imp"]} density profiles',
                labels=[str(i) for i in np.arange(0, out['nz'].shape[1])],
                x_line=1,
                namelist = namelist,
            )
    
    print('Simulated vs experimental impurity ion density profile slider plot prepared.')
    
    return



def nimp_sensitivity_analysis(shot,exp_data,namelists,asim,out,interval,labels,time_exp_shift=0.0,tmin=None,tmax=None,xmin=None,xmax=None,errorbars=False):
    
    time_average = ()
    sim_average_profiles = ()
    for i in range(0,len(out)):
        time_average = time_average + (plot_tools.time_average_profiles(asim[i].namelist['timing'], asim[i].time_out, out[i]['nz'], interval = interval)[0],)
        sim_average_profiles = sim_average_profiles + (plot_tools.time_average_profiles(asim[i].namelist['timing'], asim[i].time_out, out[i]['nz'], interval = interval)[1],)
        
    rhop_exp = exp_data[f'{namelists[0]["imp"]}_density_plasma']['rhop'].transpose()[:,0]
    n_imp_exp = exp_data[f'{namelists[0]["imp"]}_density_plasma'][f'n_{namelists[0]["imp"]}'].transpose()
    n_imp_unc_exp = exp_data[f'{namelists[0]["imp"]}_density_plasma'][f'n_{namelists[0]["imp"]}_unc'].transpose()
    
    if "time" in exp_data[f'{namelists[0]["imp"]}_density_plasma']:
        
        time_exp = exp_data[f'{namelists[0]["imp"]}_density_plasma']['time'].transpose()[:,0] + time_exp_shift
        
        f_n_imp_exp = interp1d(time_exp, n_imp_exp, axis=1, kind="linear", fill_value="extrapolate", assume_sorted=False)
        f_unc_n_imp_exp = interp1d(time_exp, n_imp_unc_exp, axis=1, kind="linear", fill_value="extrapolate", assume_sorted=False)
        
        n_imp_exp = f_n_imp_exp(time_average[0])
        n_imp_unc_exp = f_unc_n_imp_exp(time_average[0])
    
    else:
    
        n_imp_exp = np.repeat(n_imp_exp,len(time_average[0]),axis=1)
        n_imp_unc_exp = np.repeat(n_imp_unc_exp,len(time_average[0]),axis=1)
    
    n_imp_exp = n_imp_exp.reshape(1,n_imp_exp.shape[0],n_imp_exp.shape[1])
    n_imp_unc_exp = n_imp_unc_exp.reshape(1,n_imp_unc_exp.shape[0],n_imp_unc_exp.shape[1])

    if tmin is not None:
        idx_min = np.argmin(np.abs(np.asarray(time_average[0]) - tmin))
        idx_max = np.argmin(np.abs(np.asarray(time_average[0]) - tmax))
    else:
        idx_min = 0
        idx_max = -1
        
    sim_profiles = ()
    for i in range(0,len(out)):
        sim_profiles = sim_profiles + (sim_average_profiles[i].transpose(1, 0, 2)[:,:,idx_min:idx_max],)

    slider_plot_sensitivity_analysis(
                asim[0].rhop_grid,
                (time_average[0])[idx_min:idx_max],
                sim_profiles,
                rhop_exp,
                n_imp_exp[:,:,idx_min:idx_max],
                n_imp_unc_exp[:,:,idx_min:idx_max],
                simlabels = labels,
                xmin = xmin,
                xmax = xmax,
                errorbars = errorbars,
                xlabel=r'$\rho_p$',
                ylabel="time [s]",
                zlabel=f'$n_{{{namelists[0]["imp"]}^{{+}}}}$ [$\mathrm{{cm}}$$^{{-3}}$]',
                plot_title = f'{namelists[0]["imp"]} density profiles',
                labels=[str(i) for i in np.arange(0, out[0]['nz'].shape[1])],
                x_line=1,
                namelists = namelists,
            )
    
    print('Simulated vs experimental impurity ion density profile slider plot prepared.')
    
    return



def slider_plot(
    x,
    y,
    z,
    x_exp,
    z_exp,
    z_exp_unc,
    xmin,
    xmax,
    errorbars,
    print_animation,
    fps,
    filename,
    xlabel,
    ylabel,
    zlabel,
    plot_title,
    labels,
    x_line,
    namelist,
    **kwargs
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
    # if y_line is not None:
    #     a_plot.axhline(y_line, c="r", ls=":", lw=0.5)
        
    if xmin is not None:
        limx_min = xmin
        limx_max = xmax
    else:
        limx_min = x[0]
        limx_max = x[-1]*1.05
    
    a_plot.set_xlim(limx_min,limx_max)
        
    lim_min = np.min(zz.sum(axis=0))
    lim_max = np.max(zz.sum(axis=0))*1.15
    
    if errorbars:
    
        l_err = a_plot.errorbar(
            x_exp,
            zz_exp[:, :, 0].sum(axis=0),
            yerr = zz_exp_unc[:, :, 0].sum(axis=0),
            ecolor=light_blue,
            lw=mpl.rcParams["lines.linewidth"] * 1,
            fmt=' ',
            zorder = 0,
            **kwargs
        )

    l_exp = a_plot.scatter(
            x_exp,
            zz_exp[:, :, 0].sum(axis=0),
            color=blue,
            zorder = 1,
            **kwargs
        )
    _ = a_legend.scatter(
            [],
            [],
            color=blue,
            label="Experiment",
            **kwargs
        )

    (l_sim,) = a_plot.plot(
            x,
            zz[:, :, 0].sum(axis=0),
            c="k",
            lw=3.5,
            zorder = 2,
            **kwargs
        )
    _ = a_legend.plot(
            [],
            [],
            c="k",
            lw=3.5,
            label="Simulation",
            **kwargs
        )

    leg = a_legend.legend(loc="best", fontsize=12).set_draggable(True)
    title = fig.suptitle("")
    a_legend.axis("off")
    a_slider.axis("off")

    def update(dum):

        i = int(slider.val)
        
        # if puff:
            
        #     if y[i] > namelist["src_step_times"][1] + namelist["timing"]["time_start_plot"] and y[i] < namelist["src_step_times"][2] + namelist["timing"]["time_start_plot"]:
            
        #         puff_area = a_plot.axvspan(
        #             limx_min,
        #             limx_max,
        #             color="gainsboro",
        #             zorder = 0,
        #             **kwargs
        #         )
                
        #     else:
                
        #         puff_area = a_plot.axvspan(
        #             limx_min,
        #             limx_max,
        #             color="white",
        #             zorder = 0,
        #             **kwargs
        #         )
        
        if errorbars:
            
            _, _, (bars, ) = l_err
            y_base = zz_exp[:, :, i].sum(axis=0)
            x_base = x_exp

            yerr_top = y_base + zz_exp_unc[:, :, i].sum(axis=0)
            yerr_bot = y_base - zz_exp_unc[:, :, i].sum(axis=0)

            new_segments = [np.array([[x, yt], [x, yb]]) for
                            x, yt, yb in zip(x_base, yerr_top, yerr_bot)]

            bars.set_segments(new_segments)

        l_exp.set_offsets(np.c_[x_exp,zz_exp[:, :, i].sum(axis=0)])
        l_sim.set_ydata(zz[:, :, i].sum(axis=0))

        a_plot.relim()
        a_plot.autoscale()
        
        xmin,xmax = get_xlims()
        if xmin is not None:
            a_plot.set_xlim(xmin,xmax)
        else:
            a_plot.set_xlim(x[0],x[-1]*1.05)
          
        a_plot.set_ylim(lim_min,lim_max)

        if plot_title is not None:
            title.set_text(f"{plot_title}, %s = %.5f" % (ylabel, y[i]) if ylabel else f"{plot_title}, %.5f" % (y[i],))
        else:
            title.set_text("%s = %.5f" % (ylabel, y[i]) if ylabel else "%.5f" % (y[i],))

        fig.canvas.draw()
    
    def get_xlims():
        
        return xmin,xmax

    def arrow_respond(slider, event):
        if event.key == "right":
            slider.set_val(min(slider.val + 1, slider.valmax))
        elif event.key == "left":
            slider.set_val(max(slider.val - 1, slider.valmin))

    slider = mplw.Slider(a_slider, ylabel, 0, len(y) - 1, valinit=0, valfmt="%d") 
    slider.on_changed(update)
    update(0)
    fig.canvas.mpl_connect("key_press_event", lambda evt: arrow_respond(slider, evt))
    
    if print_animation:
        
        fig_anim = plt.figure()
        fig_anim.set_size_inches(7, 5, forward=True)
        a_plot = plt.subplot2grid((10, 10), (0, 0), rowspan=8, colspan=8, fig=fig_anim)
        a_legend = plt.subplot2grid((10, 10), (0, 8), rowspan=8, colspan=2, fig=fig_anim)
        
        a_plot.set_xlabel(xlabel)
        a_plot.set_ylabel(zlabel)
        if x_line is not None:
            a_plot.axvline(x_line, c="r", ls=":", lw=0.5)
        # if y_line is not None:
        #     a_plot.axhline(y_line, c="r", ls=":", lw=0.5)
            
        if xmin is not None:
            a_plot.set_xlim(xmin,xmax)
        else:
            a_plot.set_xlim(x[0],x[-1]*1.05)
            
        lim_min = np.min(zz.sum(axis=0))
        lim_max = np.max(zz.sum(axis=0))*1.15
        a_plot.set_ylim(lim_min,lim_max)
            
        # if puff:
            
        #     if y[0] > namelist["src_step_times"][1] + namelist["timing"]["time_start_plot"] and y[0] < namelist["src_step_times"][2] + namelist["timing"]["time_start_plot"]:
            
        #         puff_area = a_plot.axvspan(
        #             limx_min,
        #             limx_max,
        #             color="gainsboro",
        #             zorder = 0,
        #             **kwargs
        #         )
                
        #     else:
                
        #         puff_area = a_plot.axvspan(
        #             limx_min,
        #             limx_max,
        #             color="white",
        #             zorder = 0,
        #             **kwargs
        #         )
        
        if errorbars:
        
            l_err = a_plot.errorbar(
                x_exp,
                zz_exp[:, :, 0].sum(axis=0),
                yerr = zz_exp_unc[:, :, 0].sum(axis=0),
                ecolor="darkgray",
                lw=mpl.rcParams["lines.linewidth"] * 1,
                fmt=' ',
                zorder = 0,
                **kwargs
            )
        
        l_exp = a_plot.scatter(
                x_exp,
                zz_exp[:, :, 0].sum(axis=0),
                c="k",
                lw=mpl.rcParams["lines.linewidth"] * 2,
                zorder = 1,
                **kwargs
            )
        
        _ = a_legend.scatter(
                [],
                [],
                c="k",
                lw=mpl.rcParams["lines.linewidth"] * 2,
                label="Experiment",
                **kwargs
            )
        
        (l_sim,) = a_plot.plot(
                x,
                zz[:, :, 0].sum(axis=0),
                c="b",
                lw=mpl.rcParams["lines.linewidth"] * 2,
                zorder = 2,
                **kwargs
            )
        
        _ = a_legend.plot(
                [],
                [],
                c="b",
                lw=mpl.rcParams["lines.linewidth"] * 2,
                label="Simulation",
                **kwargs
            )
        
        y_text = a_plot.text(0.75, 0.95, " ", fontsize=14, transform=a_plot.transAxes)
        
        def init():  # initialization function
            
            if errorbars:
                
                _, _, (bars, ) = l_err
                y_base = zz_exp[:, :, 0].sum(axis=0)
                x_base = x_exp

                yerr_top = y_base + zz_exp_unc[:, :, 0].sum(axis=0)
                yerr_bot = y_base - zz_exp_unc[:, :, 0].sum(axis=0)

                new_segments = [np.array([[x, yt], [x, yb]]) for
                                x, yt, yb in zip(x_base, yerr_top, yerr_bot)]

                bars.set_segments(new_segments)
            
            l_exp.set_offsets(np.c_[x_exp,zz_exp[:, :, 0].sum(axis=0)])
            l_sim.set_ydata(zz[:, :, 0].sum(axis=0))
            
            y_text.set_text("")
            
            return bars, l_exp, l_sim, y_text,
        
        def animate(i):  # animation function, called sequentially

            y_text.set_text('time = ' + str(round(y[i], 4)) + 's')
            
            # if puff:
                
            #     if y[i] > namelist["src_step_times"][1] + namelist["timing"]["time_start_plot"] and y[i] < namelist["src_step_times"][2] + namelist["timing"]["time_start_plot"]:
                
            #         puff_area = a_plot.axvspan(
            #             limx_min,
            #             limx_max,
            #             color="gainsboro",
            #             zorder = 0,
            #             **kwargs
            #         )
                    
            #     else:
                    
            #         puff_area = a_plot.axvspan(
            #             limx_min,
            #             limx_max,
            #             color="white",
            #             zorder = 0,
            #             **kwargs
            #         )
            
            if errorbars:
                
                _, _, (bars, ) = l_err
                y_base = zz_exp[:, :, i].sum(axis=0)
                x_base = x_exp

                yerr_top = y_base + zz_exp_unc[:, :, i].sum(axis=0)
                yerr_bot = y_base - zz_exp_unc[:, :, i].sum(axis=0)

                new_segments = [np.array([[x, yt], [x, yb]]) for
                                x, yt, yb in zip(x_base, yerr_top, yerr_bot)]

                bars.set_segments(new_segments)

            l_exp.set_offsets(np.c_[x_exp,zz_exp[:, :, i].sum(axis=0)])
            l_sim.set_ydata(zz[:, :, i].sum(axis=0))

            return bars, l_exp, l_sim, y_text,
    
        a_legend.legend(loc="center").set_draggable(True)
        a_legend.axis("off")
        
        # run animation now:
        anim = animation.FuncAnimation(
            fig_anim, animate, init_func=init, frames=len(y), interval=20, blit=True
        )

        if filename is not None:
            if "gif" in filename:
                anim.save(filename, fps=fps)
            elif "mp4" in filename:
                anim.save(filename, fps=fps)
            else:
                raise ValueError('Select .gif or .mp4 file extension!')




def slider_plot_sensitivity_analysis(
    x,
    y,
    z,
    x_exp,
    z_exp,
    z_exp_unc,
    simlabels,
    xmin,
    xmax,
    errorbars,
    xlabel,
    ylabel,
    zlabel,
    plot_title,
    labels,
    x_line,
    namelists,
    **kwargs
):
    
    colors = plot_tools.load_color_codes_reservoirs()
    blue,light_blue,green,light_green,grey,light_grey,red,light_red = colors
    
    color_indices = np.zeros(len(z))
    colors = ()
    cmap = mpl.colormaps['viridis']
    for i in range(0,len(z)):
        color_indices[i] = (1/(len(z)+1)) * (i+1)
        colors = colors + (cmap(color_indices[i]),) 

    if labels is None:
        labels = ["" for v in z]

    # make sure not to modify the z array in place
    zz = ()

    for j in range(0,len(z)):
        zz = zz + (copy.deepcopy(z[j]),)
    
    zz_exp = copy.deepcopy(z_exp)
    zz_exp_unc = copy.deepcopy(z_exp_unc)

    fig = plt.figure()
    fig.set_size_inches(8, 5, forward=True)

    # separate plot into 3 subgrids
    a_plot = plt.subplot2grid((10, 11), (0, 0), rowspan=8, colspan=8, fig=fig)
    #a_legend = plt.subplot2grid((10, 12), (0, 8), rowspan=8, colspan=2, fig=fig)
    a_slider = plt.subplot2grid((10, 11), (9, 0), rowspan=1, colspan=8, fig=fig)
    a_colorbar = plt.subplot2grid((10, 11), (0, 8), rowspan=8, colspan=1, fig=fig)

    a_plot.set_xlabel(xlabel)
    a_plot.set_ylabel(zlabel)
    if x_line is not None:
        a_plot.axvline(x_line, c="r", ls=":", lw=0.5)
    # if y_line is not None:
    #     a_plot.axhline(y_line, c="r", ls=":", lw=0.5)
        
    if xmin is not None:
        limx_min = xmin
        limx_max = xmax
    else:
        limx_min = x[0]
        limx_max = x[-1]*1.05
    
    a_plot.set_xlim(limx_min,limx_max)
        
    mins = np.zeros(len(z))
    maxs = np.zeros(len(z))
    
    for j in range(0,len(z)):
        mins[j] = np.min(zz[j].sum(axis=0))
        maxs[j] = np.max(zz[j].sum(axis=0))
    
    lim_min = np.min(mins)
    lim_max = np.max(maxs)*1.15
    
    if errorbars:
    
        l_err = a_plot.errorbar(
            x_exp,
            zz_exp[:, :, 0].sum(axis=0),
            yerr = zz_exp_unc[:, :, 0].sum(axis=0),
            ecolor="gainsboro",
            lw=mpl.rcParams["lines.linewidth"] * 1,
            fmt=' ',
            zorder = 0,
            **kwargs
        )

    l_exp = a_plot.scatter(
            x_exp,
            zz_exp[:, :, 0].sum(axis=0),
            color='black',
            zorder = 1,
            **kwargs
        )
    # _ = a_legend.scatter(
    #         [],
    #         [],
    #         color=blue,
    #         label="Experiment",
    #         **kwargs
    #     )

    l_sim = [0]*len(z)
    for j in range(0,len(z)):
        (l_sim[j],) = a_plot.plot(
            x,
            zz[j][:, :, 0].sum(axis=0),
            c=colors[j],
            lw=3.5,
            zorder = 2,
            **kwargs
        )
        # _ = a_legend.plot(
        #     [],
        #     [],
        #     c=colors[j],
        #     lw=3.5,
        #     label=simlabels[j],
        #     **kwargs
        # )

    cmap = plt.cm.viridis
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', colors, cmap.N)
    bounds = np.linspace(0, 1, len(z)+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ticks = np.zeros(len(z))
    i = 0
    tick = 0
    while i < len(z):
        ticks[i] = tick + (1/(len(z)))/2
        tick = ticks[i] + (1/(len(z)))/2
        i = i+1
    cb = mpl.colorbar.ColorbarBase(a_colorbar, cmap=cmap, norm=norm,
        spacing='proportional', ticks=ticks, boundaries=bounds, format='%1i')
    cb.ax.set_yticklabels(simlabels)

    #leg = a_legend.legend(loc="best", fontsize=12).set_draggable(True)
    title = fig.suptitle("")
    #a_legend.axis("off")
    a_slider.axis("off")

    def update(dum):

        i = int(slider.val)
        
        # if puff:
            
        #     if y[i] > namelist["src_step_times"][1] + namelist["timing"]["time_start_plot"] and y[i] < namelist["src_step_times"][2] + namelist["timing"]["time_start_plot"]:
            
        #         puff_area = a_plot.axvspan(
        #             limx_min,
        #             limx_max,
        #             color="gainsboro",
        #             zorder = 0,
        #             **kwargs
        #         )
                
        #     else:
                
        #         puff_area = a_plot.axvspan(
        #             limx_min,
        #             limx_max,
        #             color="white",
        #             zorder = 0,
        #             **kwargs
        #         )
        
        if errorbars:
            
            _, _, (bars, ) = l_err
            y_base = zz_exp[:, :, i].sum(axis=0)
            x_base = x_exp

            yerr_top = y_base + zz_exp_unc[:, :, i].sum(axis=0)
            yerr_bot = y_base - zz_exp_unc[:, :, i].sum(axis=0)

            new_segments = [np.array([[x, yt], [x, yb]]) for
                            x, yt, yb in zip(x_base, yerr_top, yerr_bot)]

            bars.set_segments(new_segments)

        l_exp.set_offsets(np.c_[x_exp,zz_exp[:, :, i].sum(axis=0)])
        for j in range(0,len(z)):
            l_sim[j].set_ydata(zz[j][:, :, i].sum(axis=0))

        a_plot.relim()
        a_plot.autoscale()
        
        xmin,xmax = get_xlims()
        if xmin is not None:
            a_plot.set_xlim(xmin,xmax)
        else:
            a_plot.set_xlim(x[0],x[-1]*1.05)
          
        a_plot.set_ylim(lim_min,lim_max)

        if plot_title is not None:
            title.set_text(f"{plot_title}, %s = %.5f" % (ylabel, y[i]) if ylabel else f"{plot_title}, %.5f" % (y[i],))
        else:
            title.set_text("%s = %.5f" % (ylabel, y[i]) if ylabel else "%.5f" % (y[i],))

        fig.canvas.draw()
    
    def get_xlims():
        
        return xmin,xmax

    def arrow_respond(slider, event):
        if event.key == "right":
            slider.set_val(min(slider.val + 1, slider.valmax))
        elif event.key == "left":
            slider.set_val(max(slider.val - 1, slider.valmin))

    slider = mplw.Slider(a_slider, ylabel, 0, len(y) - 1, valinit=0, valfmt="%d") 
    slider.on_changed(update)
    update(0)
    fig.canvas.mpl_connect("key_press_event", lambda evt: arrow_respond(slider, evt))
    