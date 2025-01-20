import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from aurora import plot_tools


def integrated_fluxes(namelist,asim,reservoirs,time_start=None,time_end=None):
    
    colors = plot_tools.load_color_codes_reservoirs()
    blue,light_blue,green,light_green,grey,light_grey,red,light_red = colors
    
    time = asim.time_out
    
    if time_start is not None and time_end is not None:
        idx_start = np.argmin(np.abs(np.asarray(time) - time_start))
        idx_end = np.argmin(np.abs(np.asarray(time) - time_end))
    else:
        idx_start = 0
        idx_end = -1
    
    # fluxes in the desired time windows
    
    # external source
    external_source = reservoirs["source"]
    
    # plasma balance
    plasma_source = reservoirs["plasma_source"]
    plasma_wall_source = reservoirs["wall_source"]
    plasma_divertor_source = reservoirs["divertor_source"]
    removal_rate = reservoirs["plasma_removal_rate"]
    
    # main wall fluxes
    main_wall_flux = reservoirs["total_flux_mainwall"]
    main_wall_recycling = reservoirs["mainwall_recycling"]
    
    # divertor wall fluxes
    divertor_wall_flux = reservoirs["total_flux_divwall"]
    divertor_wall_recycling = reservoirs["divwall_recycling"]  
    
    # neutrals
    divertor_source = external_source - plasma_source
    recombination_source = (reservoirs["parallel_loss"]+reservoirs["screened_divertor_backflow"])*asim.div_recomb_ratio
    divertor_backflow = reservoirs["divertor_backflow"]+reservoirs["screened_divertor_backflow"]
    div_pump_flux = (reservoirs["particle_density_in_divertor"]-reservoirs["particle_density_in_pump"])*asim.L_divpump
    pump_leakage = reservoirs["pump_leakage"]
    
    # pumping
    pumping = reservoirs["pumping_rate"]
    
    # integrated fluxes
    
    # total balance
    int_external_source = integrate.cumulative_trapezoid(external_source[idx_start:idx_end],time[idx_start:idx_end])[-1]
    int_pumping = integrate.cumulative_trapezoid(pumping[idx_start:idx_end],time[idx_start:idx_end])[-1]
    
    # plasma balance
    int_plasma_source = integrate.cumulative_trapezoid(plasma_source[idx_start:idx_end],time[idx_start:idx_end])[-1]
    int_plasma_wall_source = integrate.cumulative_trapezoid(plasma_wall_source[idx_start:idx_end],time[idx_start:idx_end])[-1]
    int_plasma_divertor_source = integrate.cumulative_trapezoid(plasma_divertor_source[idx_start:idx_end],time[idx_start:idx_end])[-1]
    int_plasma_removal_rate = integrate.cumulative_trapezoid(removal_rate[idx_start:idx_end],time[idx_start:idx_end])[-1]
    
    # main wall balance
    int_main_wall_flux = integrate.cumulative_trapezoid(main_wall_flux[idx_start:idx_end],time[idx_start:idx_end])[-1]
    int_main_wall_recycling = integrate.cumulative_trapezoid(main_wall_recycling[idx_start:idx_end],time[idx_start:idx_end])[-1]
    
    # divertor wall balance
    int_divertor_wall_flux = integrate.cumulative_trapezoid(divertor_wall_flux[idx_start:idx_end],time[idx_start:idx_end])[-1]
    int_divertor_wall_recycling = integrate.cumulative_trapezoid(divertor_wall_recycling[idx_start:idx_end],time[idx_start:idx_end])[-1]
    
    # divertor reservoir balance
    int_divertor_source = integrate.cumulative_trapezoid(divertor_source[idx_start:idx_end],time[idx_start:idx_end])[-1]
    int_divertor_recombination_source = integrate.cumulative_trapezoid(recombination_source[idx_start:idx_end],time[idx_start:idx_end])[-1]
    int_divertor_backflow = integrate.cumulative_trapezoid(divertor_backflow[idx_start:idx_end],time[idx_start:idx_end])[-1]
    int_divpump_flux = integrate.cumulative_trapezoid(div_pump_flux[idx_start:idx_end],time[idx_start:idx_end])[-1]
   
    # pump reservoir balance
    int_pump_leakage = integrate.cumulative_trapezoid(pump_leakage[idx_start:idx_end],time[idx_start:idx_end])[-1]

    # produce plots
    
    fig, ax1 = plt.subplots(nrows=3, ncols=2, figsize=(8, 12))
    fig.suptitle(f'Reservoirs particle balance (time = [{round(time[idx_start],2)}, {round(time[idx_end],2)}] s)',fontsize=16)
    x_ticks_labels = ['$in$','$out$']
    
    index = np.array([0,1])
    bar_width = 0.7

    # total balance

    data = [[int_external_source,int_pumping]]
    
    y_offset = np.zeros(2)
    
    step0 = ax1[0,0].bar(index, data[0], bar_width, bottom=y_offset)
    step0[0].set_color(light_green)
    step0[1].set_color(green)
    if int_external_source > 0.0:
        ax1[0,0].annotate('Tot. fueled\nparticles', xy = (0,int_external_source/2), xytext = (-0.5,int_external_source/2), horizontalalignment='right', verticalalignment='center')
    if int_pumping > 0.0:
        ax1[0,0].annotate('Tot. pumped\nparticles', xy = (1.5,int_pumping/2), xytext = (1.5,int_pumping/2), horizontalalignment='left', verticalalignment='center')
    ax1[0,0].set_xticks(index)
    ax1[0,0].set_xticklabels(x_ticks_labels, fontsize=10)
    ax1[0,0].set_xlim(-2.5,3.5)
    ax1[0,0].set_ylim(0,max(int_external_source,int_pumping)*1.1)
    ax1[0,0].set_title('Total balance', loc='right', fontsize = 11)   
    
    # plasma balance

    data = [[int_plasma_source,np.absolute(int_plasma_removal_rate)],[int_plasma_wall_source,0],[int_plasma_divertor_source,0]]
    
    y_offset = np.zeros(2)
    
    step0 = ax1[0,1].bar(index, data[0], bar_width, bottom=y_offset)
    step0[0].set_color(light_green)
    step0[1].set_color(blue)
    if int_plasma_source > 0.0:
        ax1[0,1].annotate('Fueled part.\ninto plasma', xy = (0,int_plasma_source/2), xytext = (-0.5,int_plasma_source/2), horizontalalignment='right', verticalalignment='center')
    if np.absolute(int_plasma_removal_rate) > 0.0:
        ax1[0,1].annotate('Particles lost\nfrom plasma', xy = (1.5,np.absolute(int_plasma_removal_rate)/2), xytext = (1.5,np.absolute(int_plasma_removal_rate)/2), horizontalalignment='left', verticalalignment='center')    
    y_offset = y_offset + data[0]
    step1 = ax1[0,1].bar(index, data[1], bar_width, bottom=y_offset)
    step1[0].set_color(light_grey)
    if int_plasma_wall_source > 0.0:
        ax1[0,1].annotate('Part. recycled\nfrom main wall', xy = (0,int_plasma_source+int_plasma_wall_source/2), xytext = (-0.5,int_plasma_source+int_plasma_wall_source/2), horizontalalignment='right', verticalalignment='center')
    y_offset = y_offset + data[1]
    step2 = ax1[0,1].bar(index, data[2], bar_width, bottom=y_offset)
    step2[0].set_color(grey)
    if int_plasma_divertor_source > 0.0:
        ax1[0,1].annotate('Part. returned\nfrom divertor', xy = (0,int_plasma_source+int_plasma_wall_source+int_plasma_divertor_source/2), xytext = (-0.5,int_plasma_source+int_plasma_wall_source+int_plasma_divertor_source/2), horizontalalignment='right', verticalalignment='center')
    ax1[0,1].set_xticks(index)
    ax1[0,1].set_xticklabels(x_ticks_labels, fontsize=10)
    ax1[0,1].set_xlim(-2.5,3.5)
    ax1[0,1].set_ylim(0,max(int_plasma_source+int_plasma_wall_source+int_plasma_divertor_source,np.absolute(int_plasma_removal_rate))*1.1)
    ax1[0,1].set_title('Plasma balance', loc='right', fontsize = 11)    
    
    # main wall balance

    data = [[int_main_wall_flux,int_main_wall_recycling]]
    
    y_offset = np.zeros(2)
    
    step0 = ax1[1,0].bar(index, data[0], bar_width, bottom=y_offset)
    step0[0].set_color(blue)
    step0[1].set_color(green)
    if int_main_wall_flux > 0.0:
        ax1[1,0].annotate('Part. flux\ntowards wall', xy = (0,int_main_wall_flux/2), xytext = (-0.5,int_main_wall_flux/2), horizontalalignment='right', verticalalignment='center')
    if int_main_wall_recycling > 0.0:
        ax1[1,0].annotate('Part. recycled\nfrom wall', xy = (1.5,int_main_wall_recycling/2), xytext = (1.5,int_main_wall_recycling/2), horizontalalignment='left', verticalalignment='center')
    ax1[1,0].set_xticks(index)
    ax1[1,0].set_xticklabels(x_ticks_labels, fontsize=10)
    ax1[1,0].set_xlim(-2.5,3.5)
    ax1[1,0].set_ylim(0,max(int_main_wall_flux,int_main_wall_recycling)*1.1)
    ax1[1,0].set_title('Main wall balance', loc='right', fontsize = 11)   
    
    # divertor wall balance

    data = [[int_divertor_wall_flux,int_divertor_wall_recycling]]
    
    y_offset = np.zeros(2)
    
    step0 = ax1[1,1].bar(index, data[0], bar_width, bottom=y_offset)
    step0[0].set_color(blue)
    step0[1].set_color(green)
    if int_divertor_wall_flux > 0.0:
        ax1[1,1].annotate('Part. flux\ntowards wall', xy = (0,int_divertor_wall_flux/2), xytext = (-0.5,int_divertor_wall_flux/2), horizontalalignment='right', verticalalignment='center')
    if int_divertor_wall_recycling > 0.0:
        ax1[1,1].annotate('Part. recycled\nfrom wall', xy = (1.5,int_divertor_wall_recycling/2), xytext = (1.5,int_divertor_wall_recycling/2), horizontalalignment='left', verticalalignment='center')
    ax1[1,1].set_xticks(index)
    ax1[1,1].set_xticklabels(x_ticks_labels, fontsize=10)
    ax1[1,1].set_xlim(-2.5,3.5)
    ax1[1,1].set_ylim(0,max(int_divertor_wall_flux,int_divertor_wall_recycling)*1.1)
    ax1[1,1].set_title('Divertor wall balance', loc='right', fontsize = 11)  
    
    # divertor reservoir balance

    data = [[int_divertor_source,int_divertor_backflow],[int_divertor_recombination_source,int_divpump_flux],[int_divertor_wall_recycling,0]]
    
    y_offset = np.zeros(2)
    
    step0 = ax1[2,0].bar(index, data[0], bar_width, bottom=y_offset)
    step0[0].set_color(light_green)
    step0[1].set_color(green)
    if int_divertor_source > 0.0:
        ax1[2,0].annotate('Fueled part.\ninto divertor', xy = (0,int_divertor_source/2), xytext = (-0.5,int_divertor_source/2), horizontalalignment='right', verticalalignment='center')
    if int_divertor_backflow > 0.0:
        ax1[2,0].annotate('Particles lost\ntowards plasma', xy = (1.5,int_divertor_backflow/2), xytext = (1.5,int_divertor_backflow/2), horizontalalignment='left', verticalalignment='center')
    y_offset = y_offset + data[0]
    step1 = ax1[2,0].bar(index, data[1], bar_width, bottom=y_offset)
    step1[0].set_color(blue)
    step1[1].set_color(light_green)
    if int_divertor_recombination_source > 0.0:
        ax1[2,0].annotate('SOL particles\nrecombined', xy = (0,int_divertor_source+int_divertor_recombination_source/2), xytext = (-0.5,int_divertor_source+int_divertor_recombination_source/2), horizontalalignment='right', verticalalignment='center')
    if int_divpump_flux > 0.0:
        ax1[2,0].annotate('Particles lost\ntowards pump', xy = (1.5,int_divertor_backflow+int_divpump_flux/2), xytext = (1.5,int_divertor_backflow+int_divpump_flux/2), horizontalalignment='left', verticalalignment='center')
    y_offset = y_offset + data[1]
    step2 = ax1[2,0].bar(index, data[2], bar_width, bottom=y_offset)
    step2[0].set_color(grey)
    if int_divertor_wall_recycling > 0.0:
        ax1[2,0].annotate('Part. recycled\nfrom div. wall', xy = (0,int_divertor_source+int_divertor_recombination_source+int_divertor_wall_recycling/2), xytext = (-0.5,int_divertor_source+int_divertor_recombination_source+int_divertor_wall_recycling/2), horizontalalignment='right', verticalalignment='center')
    ax1[2,0].set_xticks(index)
    ax1[2,0].set_xticklabels(x_ticks_labels, fontsize=10)
    ax1[2,0].set_xlim(-2.5,3.5)
    ax1[2,0].set_ylim(0,max(int_divertor_source+int_divertor_recombination_source+int_divertor_wall_recycling,int_divertor_backflow+int_divpump_flux)*1.1)
    ax1[2,0].set_title('Divertor reservoir balance', loc='right', fontsize = 11)   
    
    # pump reservoir balance

    data = [[int_divpump_flux,int_pump_leakage],[0,int_pumping]]
    
    y_offset = np.zeros(2)
    
    step0 = ax1[2,1].bar(index, data[0], bar_width, bottom=y_offset)
    step0[0].set_color(green)
    step0[1].set_color(light_green)
    if int_divpump_flux > 0.0:
        ax1[2,1].annotate('Part. lost\nfrom divertor', xy = (0,int_divpump_flux/2), xytext = (-0.5,int_divpump_flux/2), horizontalalignment='right', verticalalignment='center')
    if int_pump_leakage > 0.0:
        ax1[2,1].annotate('Part. leaked\ntowards plasma', xy = (1.5,int_pump_leakage/2), xytext = (1.5,int_pump_leakage/2), horizontalalignment='left', verticalalignment='center')
    y_offset = y_offset + data[0]
    step1 = ax1[2,1].bar(index, data[1], bar_width, bottom=y_offset)
    step1[1].set_color(grey)
    if int_pumping > 0.0:
        ax1[2,1].annotate('Particles\npumped', xy = (1.5,int_pump_leakage+int_pumping/2), xytext = (1.5,int_pump_leakage+int_pumping/2), horizontalalignment='left', verticalalignment='center')
    ax1[2,1].set_xticks(index)
    ax1[2,1].set_xticklabels(x_ticks_labels, fontsize=10)
    ax1[2,1].set_xlim(-2.5,3.5)
    ax1[2,1].set_ylim(0,max(int_divpump_flux,int_pump_leakage+int_pumping)*1.1)
    ax1[2,1].set_title('Pump reservoir balance', loc='right', fontsize = 11)   
    
    plt.tight_layout()
    
    print('Reservoirs particle balance plots prepared.')


def integrated_fluxes_ELM(namelist,asim,reservoirs,time_start=None,time_end=None,ELM_duration=None):
    
    colors = plot_tools.load_color_codes_reservoirs()
    blue,light_blue,green,light_green,grey,light_grey,red,light_red = colors
    
    time = asim.time_out
    
    if time_start is not None and time_end is not None:
        idx_start = np.argmin(np.abs(np.asarray(time) - time_start))
        idx_end = np.argmin(np.abs(np.asarray(time) - time_end))
    else:
        idx_start = 0
        idx_end = -1
        
    tot_steps = int((time_end-time_start)/namelist["timing"]["dt_start"][0])
 
    ELM_period = 1/namelist["ELM_model"]["ELM_frequency"]
    ELM_onset = round(ELM_period - (namelist["ELM_model"]["crash_duration"]/1000+namelist["ELM_model"]["plateau_duration"]/1000+namelist["ELM_model"]["recovery_duration"]/1000),7)
    ELM_tail = round(ELM_duration - (ELM_period - ELM_onset),7)
   
    steps_ELM = int((ELM_duration)/namelist["timing"]["dt_start"][0])
    steps_ELM_tail = int((ELM_tail)/namelist["timing"]["dt_start"][0])
    steps_inter_ELM = int((ELM_period)/namelist["timing"]["dt_start"][0])-steps_ELM
    
    # fluxes in the desired time windows
    
    # external source
    external_source = reservoirs["source"]
    
    # plasma balance
    plasma_source = reservoirs["plasma_source"]
    plasma_wall_source = reservoirs["wall_source"]
    plasma_divertor_source = reservoirs["divertor_source"]
    removal_rate = reservoirs["plasma_removal_rate"]
    
    # main wall fluxes
    main_wall_flux = reservoirs["total_flux_mainwall"]
    main_wall_recycling = reservoirs["mainwall_recycling"]
    
    # divertor wall fluxes
    divertor_wall_flux = reservoirs["total_flux_divwall"]
    divertor_wall_recycling = reservoirs["divwall_recycling"]  
    
    # neutrals
    divertor_source = external_source - plasma_source
    recombination_source = (reservoirs["parallel_loss"]+reservoirs["screened_divertor_backflow"])*asim.div_recomb_ratio
    divertor_backflow = reservoirs["divertor_backflow"]+reservoirs["screened_divertor_backflow"]
    div_pump_flux = (reservoirs["particle_density_in_divertor"]-reservoirs["particle_density_in_pump"])*asim.L_divpump
    pump_leakage = reservoirs["pump_leakage"]
    
    # pumping
    pumping = reservoirs["pumping_rate"]
    
    # integrated fluxes

    # INITIALIZE INTER-ELM VALUES
    
    # total balance
    int_external_source_inter_ELM = 0
    int_pumping_inter_ELM = 0
    
    # plasma balance
    int_plasma_source_inter_ELM = 0
    int_plasma_wall_source_inter_ELM = 0
    int_plasma_divertor_source_inter_ELM = 0
    int_plasma_removal_rate_inter_ELM = 0
    
    # main wall balance
    int_main_wall_flux_inter_ELM = 0
    int_main_wall_recycling_inter_ELM = 0
    
    # divertor wall balance
    int_divertor_wall_flux_inter_ELM = 0
    int_divertor_wall_recycling_inter_ELM = 0
    
    # divertor reservoir balance
    int_divertor_source_inter_ELM = 0
    int_divertor_recombination_source_inter_ELM = 0
    int_divertor_backflow_inter_ELM = 0
    int_divpump_flux_inter_ELM = 0
   
    # pump reservoir balance
    int_pump_leakage_inter_ELM = 0

    # TAIL OF THE PREVIOUS ELM IN THE FIRST ELM PERIOD
    
    idx_start_cycle = idx_start
    idx_end_cycle = idx_start_cycle + steps_ELM_tail
    
    # total balance
    int_external_source_intra_ELM = integrate.cumulative_trapezoid(external_source[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    int_pumping_intra_ELM = integrate.cumulative_trapezoid(pumping[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    
    # plasma balance
    int_plasma_source_intra_ELM = integrate.cumulative_trapezoid(plasma_source[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    int_plasma_wall_source_intra_ELM = integrate.cumulative_trapezoid(plasma_wall_source[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    int_plasma_divertor_source_intra_ELM = integrate.cumulative_trapezoid(plasma_divertor_source[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    int_plasma_removal_rate_intra_ELM = integrate.cumulative_trapezoid(removal_rate[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    
    # main wall balance
    int_main_wall_flux_intra_ELM = integrate.cumulative_trapezoid(main_wall_flux[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    int_main_wall_recycling_intra_ELM = integrate.cumulative_trapezoid(main_wall_recycling[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    
    # divertor wall balance
    int_divertor_wall_flux_intra_ELM = integrate.cumulative_trapezoid(divertor_wall_flux[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    int_divertor_wall_recycling_intra_ELM = integrate.cumulative_trapezoid(divertor_wall_recycling[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    
    # divertor reservoir balance
    int_divertor_source_intra_ELM = integrate.cumulative_trapezoid(divertor_source[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    int_divertor_recombination_source_intra_ELM = integrate.cumulative_trapezoid(recombination_source[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    int_divertor_backflow_intra_ELM = integrate.cumulative_trapezoid(divertor_backflow[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    int_divpump_flux_intra_ELM = integrate.cumulative_trapezoid(div_pump_flux[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
   
    # pump reservoir balance
    int_pump_leakage_intra_ELM = integrate.cumulative_trapezoid(pump_leakage[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    
    # update
    
    idx_start_cycle = idx_end_cycle
    idx_end_cycle = idx_start_cycle + steps_inter_ELM
        
    while time[idx_end_cycle] < time[idx_end]:
        
        # INTER-ELM CYCLES
        
        # total balance
        int_external_source_inter_ELM = int_external_source_inter_ELM + integrate.cumulative_trapezoid(external_source[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        int_pumping_inter_ELM = int_pumping_inter_ELM + integrate.cumulative_trapezoid(pumping[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        
        # plasma balance
        int_plasma_source_inter_ELM = int_plasma_source_inter_ELM + integrate.cumulative_trapezoid(plasma_source[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        int_plasma_wall_source_inter_ELM = int_plasma_wall_source_inter_ELM + integrate.cumulative_trapezoid(plasma_wall_source[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        int_plasma_divertor_source_inter_ELM = int_plasma_divertor_source_inter_ELM + integrate.cumulative_trapezoid(plasma_divertor_source[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        int_plasma_removal_rate_inter_ELM = int_plasma_removal_rate_inter_ELM + integrate.cumulative_trapezoid(removal_rate[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        
        # main wall balance
        int_main_wall_flux_inter_ELM = int_main_wall_flux_inter_ELM + integrate.cumulative_trapezoid(main_wall_flux[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        int_main_wall_recycling_inter_ELM = int_main_wall_recycling_inter_ELM + integrate.cumulative_trapezoid(main_wall_recycling[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        
        # divertor wall balance
        int_divertor_wall_flux_inter_ELM = int_divertor_wall_flux_inter_ELM + integrate.cumulative_trapezoid(divertor_wall_flux[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        int_divertor_wall_recycling_inter_ELM = int_divertor_wall_recycling_inter_ELM + integrate.cumulative_trapezoid(divertor_wall_recycling[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        
        # divertor reservoir balance
        int_divertor_source_inter_ELM = int_divertor_source_inter_ELM + integrate.cumulative_trapezoid(divertor_source[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        int_divertor_recombination_source_inter_ELM = int_divertor_recombination_source_inter_ELM + integrate.cumulative_trapezoid(recombination_source[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        int_divertor_backflow_inter_ELM = int_divertor_backflow_inter_ELM + integrate.cumulative_trapezoid(divertor_backflow[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        int_divpump_flux_inter_ELM = int_divpump_flux_inter_ELM + integrate.cumulative_trapezoid(div_pump_flux[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
       
        # pump reservoir balance
        int_pump_leakage_inter_ELM = int_pump_leakage_inter_ELM + integrate.cumulative_trapezoid(pump_leakage[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        
        # update
        
        idx_start_cycle = idx_end_cycle
        idx_end_cycle = idx_start_cycle + steps_ELM
        
        if time[idx_end_cycle] > time[idx_end]:
            break
        
        # INTRA-ELM CYCLES
        
        # total balance
        int_external_source_intra_ELM = int_external_source_intra_ELM + integrate.cumulative_trapezoid(external_source[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        int_pumping_intra_ELM = int_pumping_intra_ELM + integrate.cumulative_trapezoid(pumping[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        
        # plasma balance
        int_plasma_source_intra_ELM = int_plasma_source_intra_ELM + integrate.cumulative_trapezoid(plasma_source[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        int_plasma_wall_source_intra_ELM = int_plasma_wall_source_intra_ELM + integrate.cumulative_trapezoid(plasma_wall_source[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        int_plasma_divertor_source_intra_ELM = int_plasma_divertor_source_intra_ELM + integrate.cumulative_trapezoid(plasma_divertor_source[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        int_plasma_removal_rate_intra_ELM = int_plasma_removal_rate_intra_ELM + integrate.cumulative_trapezoid(removal_rate[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        
        # main wall balance
        int_main_wall_flux_intra_ELM = int_main_wall_flux_intra_ELM + integrate.cumulative_trapezoid(main_wall_flux[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        int_main_wall_recycling_intra_ELM = int_main_wall_recycling_intra_ELM + integrate.cumulative_trapezoid(main_wall_recycling[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        
        # divertor wall balance
        int_divertor_wall_flux_intra_ELM = int_divertor_wall_flux_intra_ELM + integrate.cumulative_trapezoid(divertor_wall_flux[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        int_divertor_wall_recycling_intra_ELM = int_divertor_wall_recycling_intra_ELM + integrate.cumulative_trapezoid(divertor_wall_recycling[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        
        # divertor reservoir balance
        int_divertor_source_intra_ELM = int_divertor_source_intra_ELM + integrate.cumulative_trapezoid(divertor_source[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        int_divertor_recombination_source_intra_ELM = int_divertor_recombination_source_intra_ELM + integrate.cumulative_trapezoid(recombination_source[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        int_divertor_backflow_intra_ELM = int_divertor_backflow_intra_ELM + integrate.cumulative_trapezoid(divertor_backflow[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        int_divpump_flux_intra_ELM = int_divpump_flux_intra_ELM + integrate.cumulative_trapezoid(div_pump_flux[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
       
        # pump reservoir balance
        int_pump_leakage_intra_ELM = int_pump_leakage_intra_ELM + integrate.cumulative_trapezoid(pump_leakage[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        
        # update
        
        idx_start_cycle = idx_end_cycle
        idx_end_cycle = idx_start_cycle + steps_inter_ELM
        
    # FIRST PART OF THE ELM IN THE LAST ELM PERIOD
    
    idx_end_cycle = idx_start_cycle + (steps_ELM - steps_ELM_tail)
    
    # total balance
    int_external_source_intra_ELM = int_external_source_intra_ELM + integrate.cumulative_trapezoid(external_source[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    int_pumping_intra_ELM = int_pumping_intra_ELM + integrate.cumulative_trapezoid(pumping[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    
    # plasma balance
    int_plasma_source_intra_ELM = int_plasma_source_intra_ELM + integrate.cumulative_trapezoid(plasma_source[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    int_plasma_wall_source_intra_ELM = int_plasma_wall_source_intra_ELM + integrate.cumulative_trapezoid(plasma_wall_source[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    int_plasma_divertor_source_intra_ELM = int_plasma_divertor_source_intra_ELM + integrate.cumulative_trapezoid(plasma_divertor_source[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    int_plasma_removal_rate_intra_ELM = int_plasma_removal_rate_intra_ELM + integrate.cumulative_trapezoid(removal_rate[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    
    # main wall balance
    int_main_wall_flux_intra_ELM = int_main_wall_flux_intra_ELM + integrate.cumulative_trapezoid(main_wall_flux[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    int_main_wall_recycling_intra_ELM = int_main_wall_recycling_intra_ELM + integrate.cumulative_trapezoid(main_wall_recycling[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    
    # divertor wall balance
    int_divertor_wall_flux_intra_ELM = int_divertor_wall_flux_intra_ELM + integrate.cumulative_trapezoid(divertor_wall_flux[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    int_divertor_wall_recycling_intra_ELM = int_divertor_wall_recycling_intra_ELM + integrate.cumulative_trapezoid(divertor_wall_recycling[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    
    # divertor reservoir balance
    int_divertor_source_intra_ELM = int_divertor_source_intra_ELM + integrate.cumulative_trapezoid(divertor_source[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    int_divertor_recombination_source_intra_ELM = int_divertor_recombination_source_intra_ELM + integrate.cumulative_trapezoid(recombination_source[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    int_divertor_backflow_intra_ELM = int_divertor_backflow_intra_ELM + integrate.cumulative_trapezoid(divertor_backflow[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    int_divpump_flux_intra_ELM = int_divpump_flux_intra_ELM + integrate.cumulative_trapezoid(div_pump_flux[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
   
    # pump reservoir balance
    int_pump_leakage_intra_ELM = int_pump_leakage_intra_ELM + integrate.cumulative_trapezoid(pump_leakage[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    
    # produce plots
    
    fig, ax1 = plt.subplots(nrows=3, ncols=2, figsize=(10, 12))
    fig.suptitle(f'ELM-resolved reservoirs particle balance (time = [{round(time[idx_start],2)}, {round(time[idx_end],2)}] s)',fontsize=16)
    x_ticks_labels = ['$in$','$in_{ELM}$','$out_{ELM}$','$out$']
    
    index = np.array([-0.7,0,1,1.7])
    bar_width = 0.7

    # total balance

    data = [[int_external_source_inter_ELM,int_external_source_intra_ELM,int_pumping_intra_ELM,int_pumping_inter_ELM]]
    
    y_offset = np.zeros(4)
    
    step0 = ax1[0,0].bar(index, data[0], bar_width, bottom=y_offset)
    step0[0].set_color(light_green)
    step0[1].set_color(light_green)
    step0[2].set_color(green)
    step0[3].set_color(green)
    if int_external_source_inter_ELM > 0.0:
        ax1[0,0].annotate('Tot. fueled\nparticles', xy = (0,int_external_source_inter_ELM/2), xytext = (-1.2,int_external_source_inter_ELM/2), horizontalalignment='right', verticalalignment='center')
    if int_pumping_inter_ELM > 0.0:
        ax1[0,0].annotate('Tot. pumped\nparticles', xy = (1.5,int_pumping_inter_ELM/2), xytext = (2.2,int_pumping_inter_ELM/2), horizontalalignment='left', verticalalignment='center')
    ax1[0,0].set_xticks(index)
    ax1[0,0].set_xticklabels(x_ticks_labels, fontsize=10)
    ax1[0,0].set_xlim(-3.2,4.2)
    ax1[0,0].set_ylim(0,max(int_external_source_inter_ELM,int_external_source_intra_ELM,int_pumping_intra_ELM,int_pumping_inter_ELM)*1.1)
    ax1[0,0].set_title('Total balance', loc='right', fontsize = 11)   
    
    # plasma balance

    data = [[int_plasma_source_inter_ELM,int_plasma_source_intra_ELM,np.absolute(int_plasma_removal_rate_intra_ELM),np.absolute(int_plasma_removal_rate_inter_ELM)],
            [int_plasma_wall_source_inter_ELM,int_plasma_wall_source_intra_ELM,0,0],[int_plasma_divertor_source_inter_ELM,int_plasma_divertor_source_intra_ELM,0,0]]
    
    y_offset = np.zeros(4)
    
    step0 = ax1[0,1].bar(index, data[0], bar_width, bottom=y_offset)
    step0[0].set_color(light_green)
    step0[1].set_color(light_green)
    step0[2].set_color(blue)
    step0[3].set_color(blue)
    if int_plasma_source_inter_ELM > 0.0:
        ax1[0,1].annotate('Fueled part.\ninto plasma', xy = (0,int_plasma_source_inter_ELM/2), xytext = (-1.2,int_plasma_source_inter_ELM/2), horizontalalignment='right', verticalalignment='center')
    if np.absolute(int_plasma_removal_rate_inter_ELM) > 0.0:
        ax1[0,1].annotate('Particles lost\nfrom plasma', xy = (1.5,np.absolute(int_plasma_removal_rate_inter_ELM)/2), xytext = (2.2,np.absolute(int_plasma_removal_rate_inter_ELM)/2), horizontalalignment='left', verticalalignment='center')    
    y_offset = y_offset + data[0]
    step1 = ax1[0,1].bar(index, data[1], bar_width, bottom=y_offset)
    step1[0].set_color(light_grey)
    step1[1].set_color(light_grey)
    if int_plasma_wall_source_inter_ELM > 0.0:
        ax1[0,1].annotate('Part. recycled\nfrom main wall', xy = (0,int_plasma_source_inter_ELM+int_plasma_wall_source_inter_ELM/2), xytext = (-1.2,int_plasma_source_inter_ELM+int_plasma_wall_source_inter_ELM/2), horizontalalignment='right', verticalalignment='center')
    y_offset = y_offset + data[1]
    step2 = ax1[0,1].bar(index, data[2], bar_width, bottom=y_offset)
    step2[0].set_color(grey)
    step2[1].set_color(grey)
    if int_plasma_divertor_source_inter_ELM > 0.0:
        ax1[0,1].annotate('Part. returned\nfrom divertor', xy = (0,int_plasma_source_inter_ELM+int_plasma_wall_source_inter_ELM+int_plasma_divertor_source_inter_ELM/2), xytext = (-1.2,int_plasma_source_inter_ELM+int_plasma_wall_source_inter_ELM+int_plasma_divertor_source_inter_ELM/2), horizontalalignment='right', verticalalignment='center')
    ax1[0,1].set_xticks(index)
    ax1[0,1].set_xticklabels(x_ticks_labels, fontsize=10)
    ax1[0,1].set_xlim(-3.2,4.2)
    ax1[0,1].set_ylim(0,max(int_plasma_source_inter_ELM+int_plasma_wall_source_inter_ELM+int_plasma_divertor_source_inter_ELM,
                            int_plasma_source_intra_ELM+int_plasma_wall_source_intra_ELM+int_plasma_divertor_source_intra_ELM,
                            np.absolute(int_plasma_removal_rate_inter_ELM),
                            np.absolute(int_plasma_removal_rate_intra_ELM))*1.1)
    ax1[0,1].set_title('Plasma balance', loc='right', fontsize = 11)    
    
    # main wall balance

    data = [[int_main_wall_flux_inter_ELM,int_main_wall_flux_intra_ELM,int_main_wall_recycling_intra_ELM,int_main_wall_recycling_inter_ELM]]
    
    y_offset = np.zeros(4)
    
    step0 = ax1[1,0].bar(index, data[0], bar_width, bottom=y_offset)
    step0[0].set_color(blue)
    step0[1].set_color(blue)
    step0[2].set_color(green)
    step0[3].set_color(green)
    if int_main_wall_flux_inter_ELM > 0.0:
        ax1[1,0].annotate('Part. flux\ntowards wall', xy = (0,int_main_wall_flux_inter_ELM/2), xytext = (-1.2,int_main_wall_flux_inter_ELM/2), horizontalalignment='right', verticalalignment='center')
    if int_main_wall_recycling_inter_ELM > 0.0:
        ax1[1,0].annotate('Part. recycled\nfrom wall', xy = (1.5,int_main_wall_recycling_inter_ELM/2), xytext = (2.2,int_main_wall_recycling_inter_ELM/2), horizontalalignment='left', verticalalignment='center')
    ax1[1,0].set_xticks(index)
    ax1[1,0].set_xticklabels(x_ticks_labels, fontsize=10)
    ax1[1,0].set_xlim(-3.2,4.2)
    ax1[1,0].set_ylim(0,max(int_main_wall_flux_inter_ELM,int_main_wall_recycling_inter_ELM,int_main_wall_flux_intra_ELM,int_main_wall_recycling_intra_ELM)*1.1)
    ax1[1,0].set_title('Main wall balance', loc='right', fontsize = 11)   
    
    # divertor wall balance

    data = [[int_divertor_wall_flux_inter_ELM,int_divertor_wall_flux_intra_ELM,int_divertor_wall_recycling_intra_ELM,int_divertor_wall_recycling_inter_ELM]]
    
    y_offset = np.zeros(4)
    
    step0 = ax1[1,1].bar(index, data[0], bar_width, bottom=y_offset)
    step0[0].set_color(blue)
    step0[1].set_color(blue)
    step0[2].set_color(green)
    step0[3].set_color(green)
    if int_divertor_wall_flux_inter_ELM > 0.0:
        ax1[1,1].annotate('Part. flux\ntowards wall', xy = (0,int_divertor_wall_flux_inter_ELM/2), xytext = (-1.2,int_divertor_wall_flux_inter_ELM/2), horizontalalignment='right', verticalalignment='center')
    if int_divertor_wall_recycling_inter_ELM > 0.0:
        ax1[1,1].annotate('Part. recycled\nfrom wall', xy = (1.5,int_divertor_wall_recycling_inter_ELM/2), xytext = (2.2,int_divertor_wall_recycling_inter_ELM/2), horizontalalignment='left', verticalalignment='center')
    ax1[1,1].set_xticks(index)
    ax1[1,1].set_xticklabels(x_ticks_labels, fontsize=10)
    ax1[1,1].set_xlim(-3.2,4.2)
    ax1[1,1].set_ylim(0,max(int_divertor_wall_flux_inter_ELM,int_divertor_wall_recycling_inter_ELM,int_divertor_wall_flux_intra_ELM,int_divertor_wall_recycling_intra_ELM)*1.1)
    ax1[1,1].set_title('Divertor wall balance', loc='right', fontsize = 11)  
    
    # divertor reservoir balance

    data = [[int_divertor_source_inter_ELM,int_divertor_source_intra_ELM,int_divertor_backflow_intra_ELM,int_divertor_backflow_inter_ELM],
            [int_divertor_recombination_source_inter_ELM,int_divertor_recombination_source_intra_ELM,int_divpump_flux_intra_ELM,int_divpump_flux_inter_ELM],
            [int_divertor_wall_recycling_inter_ELM,int_divertor_wall_recycling_intra_ELM,0,0]]
    
    y_offset = np.zeros(4)
    
    step0 = ax1[2,0].bar(index, data[0], bar_width, bottom=y_offset)
    step0[0].set_color(light_green)
    step0[1].set_color(light_green)
    step0[2].set_color(green)
    step0[3].set_color(green)
    if int_divertor_source_inter_ELM > 0.0:
        ax1[2,0].annotate('Fueled part.\ninto divertor', xy = (0,int_divertor_source_inter_ELM/2), xytext = (-1.2,int_divertor_source_inter_ELM/2), horizontalalignment='right', verticalalignment='center')
    if int_divertor_backflow_inter_ELM > 0.0:
        ax1[2,0].annotate('Particles lost\ntowards plasma', xy = (1.5,int_divertor_backflow_inter_ELM/2), xytext = (2.2,int_divertor_backflow_inter_ELM/2), horizontalalignment='left', verticalalignment='center')
    y_offset = y_offset + data[0]
    step1 = ax1[2,0].bar(index, data[1], bar_width, bottom=y_offset)
    step1[0].set_color(blue)
    step1[1].set_color(blue)
    step1[2].set_color(light_green)
    step1[3].set_color(light_green)
    if int_divertor_recombination_source_inter_ELM > 0.0:
        ax1[2,0].annotate('SOL particles\nrecombined', xy = (0,int_divertor_source_inter_ELM+int_divertor_recombination_source_inter_ELM/2), xytext = (-1.2,int_divertor_source_inter_ELM+int_divertor_recombination_source_inter_ELM/2), horizontalalignment='right', verticalalignment='center')
    if int_divpump_flux_inter_ELM > 0.0:
        ax1[2,0].annotate('Particles lost\ntowards pump', xy = (1.5,int_divertor_backflow_inter_ELM+int_divpump_flux_inter_ELM/2), xytext = (2.2,int_divertor_backflow_inter_ELM+int_divpump_flux_inter_ELM/2), horizontalalignment='left', verticalalignment='center')
    y_offset = y_offset + data[1]
    step2 = ax1[2,0].bar(index, data[2], bar_width, bottom=y_offset)
    step2[0].set_color(grey)
    step2[1].set_color(grey)
    if int_divertor_wall_recycling_inter_ELM > 0.0:
        ax1[2,0].annotate('Part. recycled\nfrom div. wall', xy = (0,int_divertor_source_inter_ELM+int_divertor_recombination_source_inter_ELM+int_divertor_wall_recycling_inter_ELM/2), xytext = (-1.2,int_divertor_source_inter_ELM+int_divertor_recombination_source_inter_ELM+int_divertor_wall_recycling_inter_ELM/2), horizontalalignment='right', verticalalignment='center')
    ax1[2,0].set_xticks(index)
    ax1[2,0].set_xticklabels(x_ticks_labels, fontsize=10)
    ax1[2,0].set_xlim(-3.2,4.2)
    ax1[2,0].set_ylim(0,max(int_divertor_source_inter_ELM+int_divertor_recombination_source_inter_ELM+int_divertor_wall_recycling_inter_ELM,
                            int_divertor_source_intra_ELM+int_divertor_recombination_source_intra_ELM+int_divertor_wall_recycling_intra_ELM,
                            int_divertor_backflow_inter_ELM+int_divpump_flux_inter_ELM,
                            int_divertor_backflow_intra_ELM+int_divpump_flux_intra_ELM,)*1.1)
    ax1[2,0].set_title('Divertor reservoir balance', loc='right', fontsize = 11)   
    
    # pump reservoir balance

    data = [[int_divpump_flux_inter_ELM,int_divpump_flux_intra_ELM,int_pump_leakage_intra_ELM,int_pump_leakage_inter_ELM],
            [0,0,int_pumping_intra_ELM,int_pumping_inter_ELM]]
    
    y_offset = np.zeros(4)
    
    step0 = ax1[2,1].bar(index, data[0], bar_width, bottom=y_offset)
    step0[0].set_color(green)
    step0[1].set_color(green)
    step0[2].set_color(light_green)
    step0[3].set_color(light_green)
    if int_divpump_flux_inter_ELM > 0.0:
        ax1[2,1].annotate('Part. lost\nfrom divertor', xy = (0,int_divpump_flux_inter_ELM/2), xytext = (-1.2,int_divpump_flux_inter_ELM/2), horizontalalignment='right', verticalalignment='center')
    if int_pump_leakage_inter_ELM > 0.0:
        ax1[2,1].annotate('Part. leaked\ntowards plasma', xy = (1.5,int_pump_leakage_inter_ELM/2), xytext = (2.2,int_pump_leakage_inter_ELM/2), horizontalalignment='left', verticalalignment='center')
    y_offset = y_offset + data[0]
    step1 = ax1[2,1].bar(index, data[1], bar_width, bottom=y_offset)
    step1[2].set_color(grey)
    step1[3].set_color(grey)
    if int_pumping_inter_ELM > 0.0:
        ax1[2,1].annotate('Particles\npumped', xy = (1.5,int_pump_leakage_inter_ELM+int_pumping_inter_ELM/2), xytext = (2.2,int_pump_leakage_inter_ELM+int_pumping_inter_ELM/2), horizontalalignment='left', verticalalignment='center')
    ax1[2,1].set_xticks(index)
    ax1[2,1].set_xticklabels(x_ticks_labels, fontsize=10)
    ax1[2,1].set_xlim(-3.2,4.2)
    ax1[2,1].set_ylim(0,max(int_divpump_flux_inter_ELM,int_divpump_flux_intra_ELM,
                            int_pump_leakage_inter_ELM+int_pumping_inter_ELM,int_pump_leakage_intra_ELM+int_pumping_intra_ELM)*1.1)
    ax1[2,1].set_title('Pump reservoir balance', loc='right', fontsize = 11)   
    
    plt.tight_layout()
    
    print('Reservoirs particle balance plots prepared.')
    
    
def wall_balance(namelist,asim,PWI_traces,time_start=None,time_end=None):
    
    colors_PWI = plot_tools.load_color_codes_PWI()
    reds, blues, light_blues, greens = colors_PWI
    
    time = asim.time_out
    
    if time_start is not None and time_end is not None:
        idx_start = np.argmin(np.abs(np.asarray(time) - time_start))
        idx_end = np.argmin(np.abs(np.asarray(time) - time_end))
    else:
        idx_start = 0
        idx_end = -1
    
    # fluxes in the desired time windows
    
    # main wall
    implantation_rate_main = PWI_traces["impurity_implantation_rate_mainwall"]
    sputtering_rates_main = PWI_traces["impurity_sputtering_rates_mainwall"]
    
    # divertor wall
    implantation_rate_div = PWI_traces["impurity_implantation_rate_divwall"]
    sputtering_rates_div = PWI_traces["impurity_sputtering_rates_divwall"]
    
    # integrated fluxes
    
    # main wall
    int_implantation_rate_main = integrate.cumulative_trapezoid(implantation_rate_main[idx_start:idx_end],time[idx_start:idx_end])[-1]
    int_sputtering_rates_main = np.zeros(sputtering_rates_main.shape[1])
    for i in range(0,sputtering_rates_main.shape[1]):
        int_sputtering_rates_main[i] = integrate.cumulative_trapezoid(sputtering_rates_main[idx_start:idx_end,i],time[idx_start:idx_end])[-1]
        
    # divertor wall
    int_implantation_rate_div = integrate.cumulative_trapezoid(implantation_rate_div[idx_start:idx_end],time[idx_start:idx_end])[-1]
    int_sputtering_rates_div = np.zeros(sputtering_rates_div.shape[1])
    for i in range(0,sputtering_rates_div.shape[1]):
        int_sputtering_rates_div[i] = integrate.cumulative_trapezoid(sputtering_rates_div[idx_start:idx_end,i],time[idx_start:idx_end])[-1]
    
    # produce plots
    
    fig, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
    fig.suptitle(f'Wall retention balance (time = [{round(time[idx_start],2)}, {round(time[idx_end],2)}] s)',fontsize=16)
    x_ticks_labels = ['$in$','$out$']
    
    index = np.array([0,1])
    bar_width = 0.7

    # main wall

    data = [[int_implantation_rate_main,int_sputtering_rates_main[0]]]
    i=1
    while i < sputtering_rates_main.shape[1]:
        data.append([0,int_sputtering_rates_main[i]])
        i = i+1
    
    y_offset = np.zeros(2)
    
    step0 = ax1[0].bar(index, data[0], bar_width, bottom=y_offset)
    step0[0].set_color(greens[0])
    step0[1].set_color(blues[0])
    ax1[0].annotate('Absorbed\nparticles', xy = (0,int_implantation_rate_main/2), xytext = (-0.5,int_implantation_rate_main/2), horizontalalignment='right', verticalalignment='center')
    ax1[0].annotate(f'Part. sputtered\nfrom {namelist["imp"]}', xy = (1.5,int_sputtering_rates_main[0]/2), xytext = (1.5,int_sputtering_rates_main[0]/2), horizontalalignment='left', verticalalignment='center')
    y_offset = y_offset + data[0]
    tot = data[0][1]
    i=1
    while i < sputtering_rates_main.shape[1]:
        steps = ax1[0].bar(index, data[i], bar_width, bottom=y_offset)
        steps[1].set_color(blues[i])
        tot = tot + data[i][1]/2
        ax1[0].annotate(f'Part. sputtered\nfrom {namelist["full_PWI"]["background_species"][i-1]}', xy = (1.5,tot), xytext = (1.5,tot), horizontalalignment='left', verticalalignment='center')
        y_offset = y_offset + data[i]
        tot = tot + data[i][1]/2
        i = i+1
    ax1[0].set_xticks(index)
    ax1[0].set_xticklabels(x_ticks_labels, fontsize=14)
    ax1[0].set_xlim(-2.5,3.5)
    #ax1[0,0].set_ylim(0,max(int_external_source,int_pumping)*1.1)
    ax1[0].set_title('Main wall balance', loc='right', fontsize = 10)   
    
    # divertor wall

    data = [[int_implantation_rate_div,int_sputtering_rates_div[0]]]
    i=1
    while i < sputtering_rates_div.shape[1]:
        data.append([0,int_sputtering_rates_div[i]])
        i = i+1
    
    y_offset = np.zeros(2)
    
    step0 = ax1[1].bar(index, data[0], bar_width, bottom=y_offset)
    step0[0].set_color(greens[0])
    step0[1].set_color(blues[0])
    ax1[1].annotate('Absorbed\nparticles', xy = (0,int_implantation_rate_div/2), xytext = (-0.5,int_implantation_rate_div/2), horizontalalignment='right', verticalalignment='center')
    ax1[1].annotate(f'Part. sputtered\nfrom {namelist["imp"]}', xy = (1.5,int_sputtering_rates_div[0]/2), xytext = (1.5,int_sputtering_rates_div[0]/2), horizontalalignment='left', verticalalignment='center')
    y_offset = y_offset + data[0]
    tot = data[0][1]
    i=1
    while i < sputtering_rates_div.shape[1]:
        steps = ax1[1].bar(index, data[i], bar_width, bottom=y_offset)
        steps[1].set_color(blues[i])
        tot = tot + data[i][1]/2
        ax1[1].annotate(f'Part. sputtered\nfrom {namelist["full_PWI"]["background_species"][i-1]}', xy = (1.5,tot), xytext = (1.5,tot), horizontalalignment='left', verticalalignment='center')
        y_offset = y_offset + data[i]
        tot = tot + data[i][1]/2
        i = i+1
    ax1[1].set_xticks(index)
    ax1[1].set_xticklabels(x_ticks_labels, fontsize=14)
    ax1[1].set_xlim(-2.5,3.5)
    #ax1[0,0].set_ylim(0,max(int_external_source,int_pumping)*1.1)
    ax1[1].set_title('Divertor wall balance', loc='right', fontsize = 10)
    
    plt.tight_layout()
    
    print('Walls particle balance plots prepared.')
    
    
def wall_balance_ELM(namelist,asim,PWI_traces,time_start=None,time_end=None,ELM_duration=None):
    
    colors_PWI = plot_tools.load_color_codes_PWI()
    reds, blues, light_blues, greens = colors_PWI
    
    time = asim.time_out
    
    if time_start is not None and time_end is not None:
        idx_start = np.argmin(np.abs(np.asarray(time) - time_start))
        idx_end = np.argmin(np.abs(np.asarray(time) - time_end))
    else:
        idx_start = 0
        idx_end = -1
        
    tot_steps = int((time_end-time_start)/namelist["timing"]["dt_start"][0])
 
    ELM_period = 1/namelist["ELM_model"]["ELM_frequency"]
    ELM_onset = round(ELM_period - (namelist["ELM_model"]["crash_duration"]/1000+namelist["ELM_model"]["plateau_duration"]/1000+namelist["ELM_model"]["recovery_duration"]/1000),7)
    ELM_tail = round(ELM_duration - (ELM_period - ELM_onset),7)
   
    steps_ELM = int((ELM_duration)/namelist["timing"]["dt_start"][0])
    steps_ELM_tail = int((ELM_tail)/namelist["timing"]["dt_start"][0])
    steps_inter_ELM = int((ELM_period)/namelist["timing"]["dt_start"][0])-steps_ELM
    
    # fluxes in the desired time windows
    
    # main wall
    implantation_rate_main = PWI_traces["impurity_implantation_rate_mainwall"]
    sputtering_rates_main = PWI_traces["impurity_sputtering_rates_mainwall"]
    
    # divertor wall
    implantation_rate_div = PWI_traces["impurity_implantation_rate_divwall"]
    sputtering_rates_div = PWI_traces["impurity_sputtering_rates_divwall"]
    
    # INITIALIZE INTER-ELM VALUES
    
    # main wall
    int_implantation_rate_main_inter_ELM = 0
    int_sputtering_rates_main_inter_ELM = np.zeros(sputtering_rates_main.shape[1])
    
    # divertor wall
    int_implantation_rate_div_inter_ELM = 0
    int_sputtering_rates_div_inter_ELM = np.zeros(sputtering_rates_div.shape[1])

    # TAIL OF THE PREVIOUS ELM IN THE FIRST ELM PERIOD
    
    idx_start_cycle = idx_start
    idx_end_cycle = idx_start_cycle + steps_ELM_tail
    
    # main wall
    int_implantation_rate_main_intra_ELM = integrate.cumulative_trapezoid(implantation_rate_main[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    int_sputtering_rates_main_intra_ELM = np.zeros(sputtering_rates_main.shape[1])
    for i in range(0,sputtering_rates_main.shape[1]):
        int_sputtering_rates_main_intra_ELM[i] = integrate.cumulative_trapezoid(sputtering_rates_main[idx_start_cycle:idx_end_cycle,i],time[idx_start_cycle:idx_end_cycle])[-1]

    # divertor wall
    int_implantation_rate_div_intra_ELM = integrate.cumulative_trapezoid(implantation_rate_div[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    int_sputtering_rates_div_intra_ELM = np.zeros(sputtering_rates_div.shape[1])
    for i in range(0,sputtering_rates_div.shape[1]):
        int_sputtering_rates_div_intra_ELM[i] = integrate.cumulative_trapezoid(sputtering_rates_div[idx_start_cycle:idx_end_cycle,i],time[idx_start_cycle:idx_end_cycle])[-1]
    
    # update
    
    idx_start_cycle = idx_end_cycle
    idx_end_cycle = idx_start_cycle + steps_inter_ELM
        
    while time[idx_end_cycle] < time[idx_end]:
        
        # INTER-ELM CYCLES
        
        # main wall
        int_implantation_rate_main_inter_ELM = int_implantation_rate_main_inter_ELM + integrate.cumulative_trapezoid(implantation_rate_main[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        for i in range(0,sputtering_rates_main.shape[1]):
            int_sputtering_rates_main_inter_ELM[i] = int_sputtering_rates_main_inter_ELM[i] + integrate.cumulative_trapezoid(sputtering_rates_main[idx_start_cycle:idx_end_cycle,i],time[idx_start_cycle:idx_end_cycle])[-1]
        
        # divertor wall
        int_implantation_rate_div_inter_ELM = int_implantation_rate_div_inter_ELM + integrate.cumulative_trapezoid(implantation_rate_div[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        for i in range(0,sputtering_rates_div.shape[1]):
            int_sputtering_rates_div_inter_ELM[i] = int_sputtering_rates_div_inter_ELM[i] + integrate.cumulative_trapezoid(sputtering_rates_div[idx_start_cycle:idx_end_cycle,i],time[idx_start_cycle:idx_end_cycle])[-1]
        
        # update
        
        idx_start_cycle = idx_end_cycle
        idx_end_cycle = idx_start_cycle + steps_ELM
        
        if time[idx_end_cycle] > time[idx_end]:
            break
        
        # INTRA-ELM CYCLES
        
        # main wall
        int_implantation_rate_main_intra_ELM = int_implantation_rate_main_intra_ELM + integrate.cumulative_trapezoid(implantation_rate_main[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        for i in range(0,sputtering_rates_main.shape[1]):
            int_sputtering_rates_main_intra_ELM[i] = int_sputtering_rates_main_intra_ELM[i] + integrate.cumulative_trapezoid(sputtering_rates_main[idx_start_cycle:idx_end_cycle,i],time[idx_start_cycle:idx_end_cycle])[-1]
        
        # divertor wall
        int_implantation_rate_div_intra_ELM = int_implantation_rate_div_intra_ELM + integrate.cumulative_trapezoid(implantation_rate_div[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
        for i in range(0,sputtering_rates_div.shape[1]):
            int_sputtering_rates_div_intra_ELM[i] = int_sputtering_rates_div_intra_ELM[i] + integrate.cumulative_trapezoid(sputtering_rates_div[idx_start_cycle:idx_end_cycle,i],time[idx_start_cycle:idx_end_cycle])[-1]      
        
        # update
        
        idx_start_cycle = idx_end_cycle
        idx_end_cycle = idx_start_cycle + steps_inter_ELM
        
    # FIRST PART OF THE ELM IN THE LAST ELM PERIOD
    
    idx_end_cycle = idx_start_cycle + (steps_ELM - steps_ELM_tail)
    
    # main wall
    int_implantation_rate_main_intra_ELM = int_implantation_rate_main_intra_ELM + integrate.cumulative_trapezoid(implantation_rate_main[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    for i in range(0,sputtering_rates_main.shape[1]):
        int_sputtering_rates_main_intra_ELM[i] = int_sputtering_rates_main_intra_ELM[i] + integrate.cumulative_trapezoid(sputtering_rates_main[idx_start_cycle:idx_end_cycle,i],time[idx_start_cycle:idx_end_cycle])[-1]
    
    # divertor wall
    int_implantation_rate_div_intra_ELM = int_implantation_rate_div_intra_ELM + integrate.cumulative_trapezoid(implantation_rate_div[idx_start_cycle:idx_end_cycle],time[idx_start_cycle:idx_end_cycle])[-1]
    for i in range(0,sputtering_rates_div.shape[1]):
        int_sputtering_rates_div_intra_ELM[i] = int_sputtering_rates_div_intra_ELM[i] + integrate.cumulative_trapezoid(sputtering_rates_div[idx_start_cycle:idx_end_cycle,i],time[idx_start_cycle:idx_end_cycle])[-1]  
    
    # produce plots
    
    fig, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    fig.suptitle(f'ELM-resolved wall retention balance (time = [{round(time[idx_start],2)}, {round(time[idx_end],2)}] s)',fontsize=16)
    x_ticks_labels = ['$in$','$in_{ELM}$','$out_{ELM}$','$out$']
    
    index = np.array([-0.7,0,1,1.7])
    bar_width = 0.7

    # main wall

    data = [[int_implantation_rate_main_inter_ELM,int_implantation_rate_main_intra_ELM,int_sputtering_rates_main_intra_ELM[0],int_sputtering_rates_main_inter_ELM[0]]]
    
    i=1
    while i < sputtering_rates_main.shape[1]:
        data.append([0,0,int_sputtering_rates_main_intra_ELM[i],int_sputtering_rates_main_inter_ELM[i]])
        i = i+1
    
    y_offset = np.zeros(4)
    
    step0 = ax1[0].bar(index, data[0], bar_width, bottom=y_offset)
    step0[0].set_color(greens[0])
    step0[1].set_color(greens[0])
    step0[2].set_color(blues[0])
    step0[3].set_color(blues[0])
    ax1[0].annotate('Absorbed\nparticles', xy = (0,int_implantation_rate_main_inter_ELM/2), xytext = (-1.2,int_implantation_rate_main_inter_ELM/2), horizontalalignment='right', verticalalignment='center')
    ax1[0].annotate(f'Part. sputtered\nfrom {namelist["imp"]}', xy = (1.5,int_sputtering_rates_main_inter_ELM[0]/2), xytext = (2.2,int_sputtering_rates_main_inter_ELM[0]/2), horizontalalignment='left', verticalalignment='center')
    y_offset = y_offset + data[0]
    tot = data[0][3]
    i = 1
    while i < sputtering_rates_main.shape[1]:
        steps = ax1[0].bar(index, data[i], bar_width, bottom=y_offset)
        steps[2].set_color(blues[i])
        steps[3].set_color(blues[i])
        tot = tot + data[i][3]/2
        ax1[0].annotate(f'Part. sputtered\nfrom {namelist["full_PWI"]["background_species"][i-1]}', xy = (1.5,tot), xytext = (2.2,tot), horizontalalignment='left', verticalalignment='center')
        y_offset = y_offset + data[i]
        tot = tot + data[i][3]/2
        i = i+1
    ax1[0].set_xticks(index)
    ax1[0].set_xticklabels(x_ticks_labels, fontsize=10)
    ax1[0].set_xlim(-3.2,4.2)
    #ax1[0].set_ylim(0,max(int_external_source_inter_ELM,int_external_source_intra_ELM,int_pumping_intra_ELM,int_pumping_inter_ELM)*1.1)
    ax1[0].set_title('Main wall balance', loc='right', fontsize = 11)   
    
    # divertor wall
    
    data = [[int_implantation_rate_div_inter_ELM,int_implantation_rate_div_intra_ELM,int_sputtering_rates_div_intra_ELM[0],int_sputtering_rates_div_inter_ELM[0]]]
    
    i=1
    while i < sputtering_rates_div.shape[1]:
        data.append([0,0,int_sputtering_rates_div_intra_ELM[i],int_sputtering_rates_div_inter_ELM[i]])
        i = i+1
    
    y_offset = np.zeros(4)
    
    step0 = ax1[1].bar(index, data[0], bar_width, bottom=y_offset)
    step0[0].set_color(greens[0])
    step0[1].set_color(greens[0])
    step0[2].set_color(blues[0])
    step0[3].set_color(blues[0])
    ax1[1].annotate('Absorbed\nparticles', xy = (0,int_implantation_rate_div_inter_ELM/2), xytext = (-1.2,int_implantation_rate_div_inter_ELM/2), horizontalalignment='right', verticalalignment='center')
    ax1[1].annotate(f'Part. sputtered\nfrom {namelist["imp"]}', xy = (1.5,int_sputtering_rates_div_inter_ELM[0]/2), xytext = (2.2,int_sputtering_rates_div_inter_ELM[0]/2), horizontalalignment='left', verticalalignment='center')
    y_offset = y_offset + data[0]
    tot = data[0][3]
    i = 1
    while i < sputtering_rates_div.shape[1]:
        steps = ax1[1].bar(index, data[i], bar_width, bottom=y_offset)
        steps[2].set_color(blues[i])
        steps[3].set_color(blues[i])
        tot = tot + data[i][3]/2
        ax1[1].annotate(f'Part. sputtered\nfrom {namelist["full_PWI"]["background_species"][i-1]}', xy = (1.5,tot), xytext = (2.2,tot), horizontalalignment='left', verticalalignment='center')
        y_offset = y_offset + data[i]
        tot = tot + data[i][3]/2
        i = i+1
    ax1[1].set_xticks(index)
    ax1[1].set_xticklabels(x_ticks_labels, fontsize=10)
    ax1[1].set_xlim(-3.2,4.2)
    #ax1[1].set_ylim(0,max(int_external_source_inter_ELM,int_external_source_intra_ELM,int_pumping_intra_ELM,int_pumping_inter_ELM)*1.1)
    ax1[1].set_title('Divertor wall balance', loc='right', fontsize = 11)   

    plt.tight_layout()
    
    print('Walls particle balance plots prepared.')