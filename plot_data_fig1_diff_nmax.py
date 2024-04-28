import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams["font.family"] = "Times New Roman"
font = {'family': 'Times New Roman'}
plt.rc('font', **font)
if __name__ == "__main__":
    df_10 = pd.read_excel(
        "/Users/lakshaygoel/Documents/Fall 2023/Phys 437A/python_code/fig_1_data_for_a_10.0_n_10_step_0.005.xlsx")
    df_20 = pd.read_excel(
        "/Users/lakshaygoel/Documents/Fall 2023/Phys 437A/python_code/fig_1_data_for_a_10.0_n_20_step_0.005.xlsx")
    df_35 = pd.read_excel(
        "/Users/lakshaygoel/Documents/Fall 2023/Phys 437A/python_code/fig_1_data_for_a_10.0_n_35_step_0.001.xlsx")
    df_50 = pd.read_excel(
        "/Users/lakshaygoel/Documents/Fall 2023/Phys 437A/python_code/fig_1_data_for_a_10.0_n_50_step_0.005.xlsx")

    print(df_10.T)

    a = 10.0

    gamma_vals_10, l_b_values_10, pa_values_10, pb_values_10, lab_values_10, pe_plus_values_10, pe_minus_values_10 = df_10.T[
        0], df_10.T[1], df_10.T[2], df_10.T[3], df_10.T[4], df_10.T[5], df_10.T[6]

    gamma_vals_20, l_b_values_20, pa_values_20, pb_values_20, lab_values_20, pe_plus_values_20, pe_minus_values_20 = df_20.T[
        0], df_20.T[1], df_20.T[2], df_20.T[3], df_20.T[4], df_20.T[5], df_20.T[6]

    gamma_vals_35, l_b_values_35, pa_values_35, pb_values_35, lab_values_35, pe_plus_values_35, pe_minus_values_35 = df_35.T[
        0], df_35.T[1], df_35.T[2], df_35.T[3], df_35.T[4], df_35.T[5], df_35.T[6]

    gamma_vals_50, l_b_values_50, pa_values_50, pb_values_50, lab_values_50, pe_plus_values_50, pe_minus_values_50 = df_50.T[
        0], df_50.T[1], df_50.T[2], df_50.T[3], df_50.T[4], df_50.T[5], df_50.T[6]

    print(len(gamma_vals_10))

    # plt.figure(figsize=(10, 6))

    # plt.plot(gamma_vals_10[1:], pe_plus_values_10[1:],
    #          '-r', label=r"$n_{max} = 10$", lw=0.75)
    # plt.plot(gamma_vals_20[1:], pe_plus_values_20[1:],
    #          '-b', label=r"$n_{max} = 20$", lw=0.75)
    # plt.plot(gamma_vals_35[1:], pe_plus_values_35[1:],
    #          '-g', label=r"$n_{max} = 35$", lw=0.75)
    # plt.plot(gamma_vals_50[1:], pe_plus_values_50[1:],
    #          '-y', label=r"$n_{max} = 50$", lw=0.75)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each x-y plane for different n_max
    ax.plot(gamma_vals_10[1:], [10]*len(gamma_vals_10[1:]),
            pe_plus_values_10[1:], '-r', label=r"$n_{max} = 10$", lw=0.75)
    ax.plot(gamma_vals_20[1:], [20]*len(gamma_vals_20[1:]),
            pe_plus_values_20[1:], '-b', label=r"$n_{max} = 20$", lw=0.75)
    ax.plot(gamma_vals_35[1:], [35]*len(gamma_vals_35[1:]),
            pe_plus_values_35[1:], '-g', label=r"$n_{max} = 35$", lw=0.75)
    ax.plot(gamma_vals_50[1:], [50]*len(gamma_vals_50[1:]),
            pe_plus_values_50[1:], '-y', label=r"$n_{max} = 50$", lw=0.75)

    # Set labels and legend
    ax.set_xlabel(
        r"Length ratio, $\,\mathregular{\gamma = \frac{\ell _B}{\ell _A}}$", fontsize=14, labelpad=7)
    ax.set_ylabel(r"n$_{max}$", fontsize=14, labelpad=7)
    ax.set_zlabel(
        r"Transition probability, $\,\mathregular{P_E^{(+)}/\,\lambda^2}$", fontsize=14, labelpad=7)
    # Adjust padding between axis numbers and labels
    # ax.tick_params(axis='x', pad=5)
    # ax.tick_params(axis='y', pad=5)
    # ax.tick_params(axis='z', pad=5)

    # ax.legend()

    # plt.plot(gamma_vals_10[1:], pe_minus_values_10[1:],
    #          '-r', label=r"$n_{max} = 10$", lw=0.75)
    # plt.plot(gamma_vals_20[1:], pe_minus_values_20[1:],
    #          '-b', label=r"$n_{max} = 20$", lw=0.75)
    # plt.plot(gamma_vals_35[1:], pe_minus_values_35[1:],
    #          '-g', label=r"$n_{max} = 35$", lw=0.75)

    # plt.ylabel(
    #     r"Transition probability, $\,\mathregular{P_E^{(+)}/\,\lambda^2}$", fontsize=16)
    # plt.xlabel(
    #     r"Length ratio, $\,\mathregular{\gamma = \frac{l_B}{l_A}}$", fontsize=16)
    # plt.title(f"Acceleration a = {a}")
    plt.legend(loc='upper right', fontsize=14)
    plt.grid()
    # plt.ylim(0.23, 0.47)
    # plt.xlim(0.19, 1.71)
    plt.tight_layout()
    plt.savefig(
        f"fig_1_nmax_comparison_for_a_{a}_nmax_{50}_3D.png", dpi=300, format="PNG")
    plt.show()
