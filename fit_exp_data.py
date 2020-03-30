from IPython.display import display, Math, Latex
from pylab import *
from part2 import *
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import pandas as pd


def format_axe(axe, ylabel = None, set_ylim=False):
    labelsize = 30
    if set_ylim:
        axe.set_ylim((1e-4,1e3))
    axe.set_yscale('log')
    axe.set_xlim((0.2,-1.2))
    if ylabel:
        axe.set_ylabel(ylabel, fontsize = labelsize)
    else:
        axe.set_ylabel(r'$\frac{R_{e\Delta V}}{R_{(e\Delta V=0)}}$', fontsize = labelsize)
    axe.set_xlabel('$e\Delta V$ $[eV]$', fontsize = labelsize)
    axe.tick_params(axis='both', which='both', labelsize=labelsize)
    axe.grid()

    



def fit_exp_with_num_data(dF_exp, calc_dF, sens='SensorName', caption_figure=None):
    #instead of unsing the row number
    #each row has the value of dV as index
    dF_exp.index = dF_exp['dV']

    v_exp = dF_exp['dV']
    res_exp = dF_exp['res']

    #get the value of the flatband (if needed)
    #by interpolation 
    interp_res = interp1d(v_exp,res_exp)
    res_flatband = interp_res(0)

    #calcualte the rel. res change
    rel_res_exp = dF_exp['res']/res_flatband


    #The dataframe to hold the different
    #of the exp. values to the numerical ones
    #Will be used to find the best fitting num. solution
    num_data_at_exp_pos_dF = pd.DataFrame(index = v_exp)

    #group the num. data by its paramters (T, R and ND)
    data_by_grain = calc_dF.groupby(['temp','R','ND'])

    for (T, R,ND), calc_dF_grain in data_by_grain:

        num_data_at_exp_pos_dF[(T, R,ND)] = None

        grain = create_grain_from_data(calc_dF_grain)

        flat_band_data = calc_dF_grain[calc_dF_grain['Einit_kT']==0].iloc[0]

        rel_res_num = calc_dF_grain['rel_res_change']

        #express the surace potential in eV
        #to be comparable with the exp. data
        v_num = calc_dF_grain['Einit_kT']*CONST.J_to_eV(grain.material.kT)

        #use interpolation to get the values for the positions
        #of the experiment data points
        interp_rs_num = interp1d(v_num, rel_res_num,bounds_error=False)
        interp_v_num = interp1d(rel_res_num,v_num, bounds_error=False)

        #caculate the numerical value of rel. res at the position
        # of V from the experiment
        res_num_at_exp_pos = interp_rs_num(v_exp)

        #save those values in the new DataFrame
        num_data_at_exp_pos_dF.loc[:,(T, R,ND)] = res_num_at_exp_pos







    abs_error = num_data_at_exp_pos_dF.subtract(rel_res_exp, axis='index')
    rel_error = abs_error.divide(rel_res_exp, axis='index')
    rel_error_square = rel_error**2
    sum_of_squares = rel_error_square.sum()

    #valid_index = [i for i in sum_of_squares.index if i[1] in [50e-9,100e-9,55e-9]]
    valid_index = [i for i in sum_of_squares.index if i[1] in [50e-9,100e-9,55e-9]]

    sum_of_squares_grainsize = sum_of_squares.loc[valid_index].sort_values()

    grain_min_error_tuple = sum_of_squares_grainsize.idxmin()








    fig, axe = subplots(figsize = (16,10));
    #for grain_tuple in num_data_at_exp_pos_dF.keys():
    for grain_tuple in sum_of_squares.index:   
        if grain_tuple == grain_min_error_tuple:
            linestyle = '*-'
            linewidth = 5
            alpha = 0.5
            label = 'Best Fit'
        elif grain_tuple == sum_of_squares_grainsize.index[1] and False:
            linestyle = '-o'
            linewidth = 5
            alpha = 0.3
            label = 'Second best fit'
        else:
            linestyle = '-.'
            linewidth = 1
            alpha = 0.3
            label = 'Other solution'


        calc_dF_grain = data_by_grain.get_group(grain_tuple)
        grain = create_grain_from_data(calc_dF_grain)
        r_temp = calc_dF_grain['rel_res_change']
        v_temp = calc_dF_grain['Einit_kT']*CONST.J_to_eV(grain.material.kT)

        axe.plot(v_temp,
                    r_temp,
                linestyle, linewidth=linewidth, alpha = alpha,
                label ='model')        

        #axe.plot(num_data_at_exp_pos_dF.index,
        #            num_data_at_exp_pos_dF[grain_tuple],
        #        linestyle, linewidth=linewidth, alpha = alpha,
        #        label =label)


        last_x  = num_data_at_exp_pos_dF.index[0] 
        last_y  = num_data_at_exp_pos_dF.iloc[0][grain_tuple]

        if grain_tuple in sum_of_squares_grainsize.index[0:1]:
            axe.text(last_x-0.05,last_y,
                f'Radius:{grain_tuple[1]*1e9:.0f}nm\n$N_D$:{grain_tuple[2]:.2} 1/m³',
                    fontsize = 22)
    format_axe(axe)

    axe.scatter(rel_res_exp.index,
                rel_res_exp,
                s=100,
                label = 'Exp. data'
               )

    axe.set_ylim(rel_res_exp.min()/10,
                 rel_res_exp.max()*2);


    l = {h[1]:h[0] for h in zip(*axe.get_legend_handles_labels())}.keys()
    h = {h[1]:h[0] for h in zip(*axe.get_legend_handles_labels())}.values()
    axe.legend(h,l,loc=1, fontsize = 22)
    axe.set_title(sens, fontsize = 30);

    close()
    display(fig);

    caption = Latex(r'\begin{center}'+'\n'+caption_figure+'\n'+r'\end{center}')
    if caption:
        display(caption)


    #display the numerical data
    best_fits_error_dF = pd.DataFrame({'error':sum_of_squares_grainsize.iloc[0:10]})
    
    T_temp = [i[0] for i in best_fits_error_dF.index]
    R_temp = [round(i[1]*1e9,2) for i in best_fits_error_dF.index]
    ND_temp = [i[2] for i in best_fits_error_dF.index]
    best_fits_error_dF['T [°C]'] = T_temp
    best_fits_error_dF['Radius [nm]'] = R_temp
    best_fits_error_dF['$N_D$ [$1/m³$]'] = ND_temp
    best_fits_error_dF.index = range(len(best_fits_error_dF))

    
    display(best_fits_error_dF)

    caption = Latex(r'''\begin{center}
    Table with the best fitting simulation parameters
    \end{center}''')
    display(caption)



    grain_data = data_by_grain.get_group(sum_of_squares_grainsize.index[0])
    best_fit_grain = create_grain_from_data(grain_data)

    data = {'$E_C$ - $E_F$': f'{best_fit_grain.material.Diff_EF_EC_evolt:.2f} eV',
        '$N_D$' : f'{best_fit_grain.material.ND:.2e} 1/m³',
         '$n_b$' : f'{best_fit_grain.material.nb:.2e} 1/m³',
         '$L_D$' : f'{best_fit_grain.material.LD*1e9:.2f} nm'}
    best_fit_data = pd.DataFrame(pd.Series(data, name = sens))
    display(best_fit_data)


    caption = Latex(r'''\begin{center}
    Table with material properties of the best fitting simulation.
    \end{center}''')
    display(caption)
    return best_fits_error_dF
    
    
