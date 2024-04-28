import mpmath
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, fixed_quad, nquad
import pandas as pd
from math import erfc, erf
from scipy.special import erf as Erf
from sympy import Symbol, integrate
import time
import logging
import os
plt.rcParams["font.family"] = "Times New Roman"
font = {'family': 'Times New Roman'}
plt.rc('font', **font)
if __name__ == "__main__":
    df_0 = pd.read_excel(
        "/Users/lakshaygoel/Documents/Fall 2023/Phys 437A/python_code/fig_1_data_for_a_0.0_n_30_step_0.1.xlsx")
    df_2 = pd.read_excel(
        "/Users/lakshaygoel/Documents/Fall 2023/Phys 437A/python_code/fig_1_data_for_a_2.0_n_20_step_0.1.xlsx")
    df_10 = pd.read_excel(
        "/Users/lakshaygoel/Documents/Fall 2023/Phys 437A/python_code/fig_1_data_for_a_10.0_n_35_step_0.001.xlsx")
    df_100 = pd.read_excel(
        "/Users/lakshaygoel/Documents/Fall 2023/Phys 437A/python_code/fig_1_data_for_a_100.0_n_20_step_0.1.xlsx")
    print(df_0.T)
    print(df_10.T)
    # [gamma_vals, l_b_values, pa_values, pb_values,
    #     lab_values, pe_plus_values, pe_minus_values] = df_0
    gamma_vals_0, l_b_values_0, pa_values_0, pb_values_0, lab_values_0, pe_plus_values_0, pe_minus_values_0 = df_0.T[
        0], df_0.T[1], df_0.T[2], df_0.T[3], df_0.T[4], df_0.T[5], df_0.T[6]

    gamma_vals_2, l_b_values_2, pa_values_2, pb_values_2, lab_values_2, pe_plus_values_2, pe_minus_values_2 = df_2.T[
        0], df_2.T[1], df_2.T[2], df_2.T[3], df_2.T[4], df_2.T[5], df_2.T[6]

    gamma_vals_10, l_b_values_10, pa_values_10, pb_values_10, lab_values_10, pe_plus_values_10, pe_minus_values_10 = df_10.T[
        0], df_10.T[1], df_10.T[2], df_10.T[3], df_10.T[4], df_10.T[5], df_10.T[6]

    # gamma_vals_100, l_b_values_100, pa_values_100, pb_values_100, lab_values_100, pe_plus_values_100, pe_minus_values_100 = df_100.T[
    #     0], df_100.T[1], df_100.T[2], df_100.T[3], df_100.T[4], df_100.T[5], df_100.T[6]

    # gamma_vals = df_0.T[0]
    v_line1 = [(n - 1) / n for n in range(2, 10)]
    v_line2 = [n/(n-1) for n in range(3, 11)]
    v_line = v_line1+v_line2+[1.0]
    print(v_line)

    scaling = False
    print(len(gamma_vals_0))
    if scaling:
        # plt.plot(gamma_vals_0[1:122], pe_plus_values_0[1:122]/pa_values_0[1:122],
        #          '-r', label=r"$a \sigma = 0$", lw=0.75)
        # plt.plot(gamma_vals_2[1:122], pe_plus_values_2[1:122]/pa_values_2[1:122],
        #          '-b', label=r"$a \sigma = 2$", lw=0.75)
        # plt.plot(gamma_vals_10[1+3000:], pe_plus_values_10[1+3000:]/pa_values_10[1+3000:],
        #          '-g', label=r"$a \sigma = 10$", lw=0.75)
        # plt.plot(gamma_vals_10[1:], pe_plus_values_10[1:]/pa_values_10[1:],
        #          '-g', label=r"$a = 10$", lw=0.75)
        # plt.plot(gamma_vals_100[1:], pe_plus_values_100[1:]/pa_values_100[1:],
        #          '-y', label=r"$a = 100$", lw=0.75)
        plt.ylabel(r"Detector response $\propto P_E^{(+)} / P_A$", fontsize=14)
    else:
        # plt.plot(gamma_vals_0[1:122], pe_plus_values_0[1:122],
        #          '-r', label=r"$a \sigma = 0$", lw=0.75)
        # plt.plot(gamma_vals_2[1:122], pe_plus_values_2[1:122],
        #          '-b', label=r"$a \sigma = 2$", lw=0.75)
        # plt.plot(gamma_vals_10[1+3000:], pe_plus_values_10[1+3000:],
        #          '-g', label=r"$a \sigma = 10$", lw=0.75)
        plt.plot(gamma_vals_10[1:], pa_values_10[1:],
                 '-y', label=r"$P_A$", lw=0.75)
        plt.plot(gamma_vals_10[1:], pb_values_10[1:],
                 '-r', label=r"$P_B$", lw=0.75)
        plt.plot(gamma_vals_10[1:], pe_plus_values_10[1:], '-b',
                 label=r"$P_E ^{+}$", lw=0.75)
        plt.plot(gamma_vals_10[1:], lab_values_10[1:],
                 '-g', label=r"$L_{AB}$", lw=0.75)
        # plt.plot(gamma_vals_10[1:], pe_minus_values_10[1:],
        #          '-k', label=r"$P_E ^{-}$")
        # plt.plot(gamma_vals_10[1:], pe_plus_values_10[1:],
        #          '-g', label=r"$a = 10$", lw=0.75)
        # plt.plot(gamma_vals_100[1:], pe_plus_values_100[1:],
        #          '-y', label=r"$a = 100$", lw=0.75)
        plt.ylabel(
            r"Transition probability, $\,\mathregular{P_E^{(+)}/\,\lambda^2}$", fontsize=14)
    plt.vlines(v_line, ymin=0.0, ymax=0.525, colors="k", ls="--", lw=0.5)
    plt.xlabel(
        r"Length ratio, $\,\mathregular{\gamma = \frac{l_B}{l_A}}$", fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.ylim(0.0, 0.525)
    plt.xlim(0.175, 1.725)
    # plt.grid()
    plt.tight_layout()
    plt.savefig(
        f"fig_1_acc_rescaled_{scaling}_new_upload.png", format='PNG')
    plt.show()
