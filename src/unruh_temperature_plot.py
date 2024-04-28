import mpmath
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, fixed_quad, nquad
import pandas as pd
from math import erfc, erf
from scipy.special import erf as Erf
from sympy import Symbol, integrate
plt.rcParams["font.family"] = "Times New Roman"

if __name__ == "__main__":

    nmax = 10
    step = 1
    a_vals = np.arange(0, 100+step, step=step)
    a_vals = np.around(a_vals, decimals=1)
    omega = [50]

    df_plus = pd.read_excel(
        f"/Users/lakshaygoel/Documents/Fall 2023/Phys 437A/python_code/new_fig_data_for_n_{nmax}_omega_{omega[0]}_step_{step}_amax_{a_vals[-1]}.xlsx")
    df_minus = pd.read_excel(
        f"/Users/lakshaygoel/Documents/Fall 2023/Phys 437A/python_code/new_fig_data_for_n_{nmax}_omega_-{omega[0]}_step_{step}_amax_{a_vals[-1]}.xlsx")

    a_vals_plus, pa_plus, pb_plus, lab_plus, peplus_plus, peminus_plus = df_plus.T[
        0], df_plus.T[1], df_plus.T[2], df_plus.T[3], df_plus.T[4], df_plus.T[5]

    a_vals_minus, pa_minus, pb_minus, lab_minus, peplus_minus, peminus_minus = df_minus.T[
        0], df_minus.T[1], df_minus.T[2], df_minus.T[3], df_minus.T[4], df_minus.T[5]

    print(peplus_plus, peplus_minus)

    ratio_vals_rhs = []

    for a in a_vals:
        if a == 0:
            ratio = 0
        else:
            ratio = np.exp(-2*np.pi*omega[0]/a)

        ratio_vals_rhs.append(ratio)

    ratio_vals_lhs = []

    for i in range(0, len(peplus_plus)-1):
        # print(i)
        ratio_vals_lhs.append(peplus_plus[i]/peplus_minus[i])

    plt.plot(a_vals, ratio_vals_rhs, '-k',
             label=r"$e^{-\beta \Omega}$", lw=1)
    plt.plot(a_vals, ratio_vals_lhs, 'r-o',
             label=r"$\frac{Pr(\Omega)}{Pr(-\Omega)}$", markersize=2, lw=0.5)
    plt.grid()
    # plt.title("Energy gap: "+r"$|\Omega\sigma| = " +
    #           str(omega[0]) + r"$", fontsize=16)
    plt.xlabel(r"Acceleration, $a\sigma$", fontsize=16)
    plt.ylabel(r"Ratio, $e^{-\beta \Omega}$", fontsize=16)
    plt.legend(loc='lower right', fontsize=14)
    # plt.xscale('log')
    plt.tight_layout()
    plt.savefig(
        f"unruh_temp_to_probab_omega_{omega[0]}_amax_{a_vals[-1]}_nmax_{nmax}_step_{step}_upload.png", format='PNG')

    plt.show()
