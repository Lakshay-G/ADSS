import numpy as np


class Config():
    dataset = 'dataset/'
    output = 'outputs/'
    label_font_size = 16
    legend_font_size = 14
    figsize = (9, 6)
    bbox_anchor = (1, 1)
    dpi_quality = 300
    nmax = 10
    nmin = -nmax
    mmax = nmax
    mmin = nmin
    sum = 0
    for n in range(nmin, nmax+1):
        sum += (-1)**(2*n)

    dataType = 'versus_acceleration'
    # dataType = 'versus_energy_gap'
    # dataType = 'versus_length_ratio'

    if dataType == 'versus_length_ratio':
        # sigma here is 0.1 implicitly
        omega = 0.01
        l_a = 10
        step_length_ratio = 0.1
        l_b_temp_values = np.arange(5, 17.6, step=step_length_ratio)
        l_b_vals = np.around(l_b_temp_values, decimals=2)
        gamma_vals = l_b_vals/l_a
        a_vals = [0.0]
        plotting_a = 0.0

    elif dataType == "versus_energy_gap":
        l_a = 0.75
        l_b = 0.25
        omega_values = np.arange(-100, 5+1, step=1)
        omega_values = omega_values/2
        a_vals = [0.0]
        plotting_a = 0.0

    elif dataType == "versus_acceleration":
        l_a = 0.75
        l_b = 0.25
        omega_vals = [-50, 50]
        step_acceleration = 1
        a_temp_values = np.arange(
            0, 100+step_acceleration, step=step_acceleration)
        a_vals = np.around(a_temp_values, decimals=1)
