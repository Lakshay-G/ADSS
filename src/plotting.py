import mpmath
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import pandas as pd
import time
import logging
from mpmath import mp
from modules import CrossTerms, SingleTerms
from generate_data import Config
plt.rcParams["font.family"] = "Times New Roman"


def plot_length_ratio(df, a, label_font_size, legend_font_size, bbox_anchor):
    gamma_vals, l_b_vals, pa_values, pb_values, lab_values, pe_plus_values, pe_minus_values = df[
        'gamma'], df['l_b'], df['pa'], df['pb'], df['lab'], df['pe_plus'], df['pe_minus']
    plt.plot(gamma_vals, lab_values, '-g', label=r"$L_{AB}$", lw=1)
    plt.plot(gamma_vals, pa_values, '-y', label=r"$P_A$", lw=1)
    plt.plot(gamma_vals, pb_values, '-r', label=r"$P_B$", lw=1)
    plt.plot(gamma_vals, pe_plus_values, '-b', label=r"$P_E ^{+}$", lw=1)
    plt.grid()
    plt.xlabel(
        r"Length ratio, $\,\mathregular{\gamma = \frac{l_B}{l_A}}$", fontsize=label_font_size)
    plt.ylabel(
        r"Transition probability, $\,\mathregular{P_E/\,\lambda^2}$", fontsize=label_font_size)
    plt.legend(loc='upper right', bbox_to_anchor=bbox_anchor,
               fontsize=legend_font_size)
    plt.title("Acceleration: "+r"$a\sigma = " +
              str(a) + r"$", fontsize=label_font_size)
    plt.tight_layout()
    # plt.show()


def plot_acceleration_unruh(df_minus, df_plus, omega, label_font_size, legend_font_size, bbox_anchor):
    a_vals_plus, pa_values_plus, pb_values_plus, lab_values_plus, pe_plus_values_plus, pe_minus_values_plus = df_plus[
        'a_vals'], df_plus['pa'], df_plus['pb'], df_plus['lab'], df_plus['pe_plus'], df_plus['pe_minus']
    a_vals_minus, pa_values_minus, pb_values_minus, lab_values_minus, pe_plus_values_minus, pe_minus_values_minus = df_minus[
        'a_vals'], df_minus['pa'], df_minus['pb'], df_minus['lab'], df_minus['pe_plus'], df_minus['pe_minus']

    ratio_vals_rhs = []
    for a in a_vals_plus:
        if a == 0:
            ratio = 0
        else:
            ratio = np.exp(-2*np.pi*omega/a)

        ratio_vals_rhs.append(ratio)

    ratio_vals_lhs = []
    for i in range(0, len(pe_plus_values_plus)):
        ratio_vals_lhs.append(pe_plus_values_plus[i]/pe_plus_values_minus[i])

    plt.plot(a_vals_plus, ratio_vals_rhs, '-k',
             label=r"$e^{-\beta \Omega}$", lw=1)
    plt.plot(a_vals_plus, ratio_vals_lhs, 'r-o',
             label=r"$\frac{Pr(\Omega)}{Pr(-\Omega)}$", markersize=2, lw=0.5)
    plt.grid()
    plt.xlabel(r"Acceleration, $a\sigma$", fontsize=label_font_size)
    plt.ylabel(r"Ratio, $e^{-\beta \Omega}$", fontsize=label_font_size)
    plt.legend(loc='lower right',
               fontsize=legend_font_size)
    plt.title("Energy gap: "+r"$|\Omega\sigma| = " +
              str(omega) + r"$", fontsize=label_font_size)

    plt.tight_layout()
    # plt.show()


def plot_energy_gap(df, a, label_font_size, legend_font_size, bbox_anchor):
    omega_vals, pa_values, pb_values, lab_values, pe_plus_values, pe_minus_values = df[
        'omegas'], df['pa'], df['pb'], df['lab'], df['pe_plus'], df['pe_minus']
    plt.plot(omega_vals, lab_values, '-g', label=r"$L_{AB}$", lw=1)
    plt.plot(omega_vals, pa_values, '-y', label=r"$P_A$", lw=1)
    plt.plot(omega_vals, pb_values, '-r', label=r"$P_B$", lw=1)
    plt.plot(omega_vals, pe_plus_values, '-b', label=r"$P_E ^{+}$", lw=1)
    plt.plot(omega_vals, pe_minus_values, '-b', label=r"$P_E ^{-}$", lw=1)
    plt.grid()
    plt.xlabel(
        r"Energy gap, $\mathregular{\Omega \sigma}$", fontsize=label_font_size)
    plt.ylabel(
        r"Transition probability, $\mathregular{P_E/\,\lambda^2}$", fontsize=label_font_size)
    plt.title("Acceleration: "+r"$a\sigma = " +
              str(a) + r"$", fontsize=label_font_size)
    plt.legend(loc='upper right', bbox_to_anchor=bbox_anchor,
               fontsize=legend_font_size)
    plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    # print('starting!')
    cfg = Config()
    # print(cfg)
    plt.figure(figsize=cfg.figsize, dpi=cfg.dpi_quality)
    if cfg.dataType == 'versus_acceleration':
        # print('acceleration')
        df_acceleration_minus = pd.read_excel(
            f'{cfg.dataset}{cfg.dataType}/n_{cfg.nmax}_amax_{cfg.a_vals[-1]}_omega_{cfg.omega_vals[0]}_step_{cfg.step_acceleration}.xlsx')
        df_acceleration_plus = pd.read_excel(
            f'{cfg.dataset}{cfg.dataType}/n_{cfg.nmax}_amax_{cfg.a_vals[-1]}_omega_{cfg.omega_vals[1]}_step_{cfg.step_acceleration}.xlsx')
        plot_acceleration_unruh(
            df_minus=df_acceleration_minus,
            df_plus=df_acceleration_plus,
            omega=cfg.omega_vals[1],
            label_font_size=cfg.label_font_size,
            legend_font_size=cfg.legend_font_size,
            bbox_anchor=cfg.bbox_anchor)
        plt.savefig(
            f'{cfg.output}{cfg.dataType}/n_{cfg.nmax}_amax_{cfg.a_vals[-1]}_omega_{cfg.omega_vals[1]}_step_{cfg.step_acceleration}.png', format='PNG')

    elif cfg.dataType == "versus_length_ratio":
        # print('versus_length_ratio')
        df_length_ratio = pd.read_excel(
            f'{cfg.dataset}{cfg.dataType}/n_{cfg.nmax}_omega_{cfg.omega}_step_{cfg.step_length_ratio}_a_{cfg.plotting_a}.xlsx')
        plot_length_ratio(
            df=df_length_ratio,
            a=cfg.plotting_a,
            label_font_size=cfg.label_font_size,
            legend_font_size=cfg.legend_font_size,
            bbox_anchor=cfg.bbox_anchor)
        plt.savefig(
            f"{cfg.output}{cfg.dataType}/n_{cfg.nmax}_omega_{cfg.omega}_step_{cfg.step_length_ratio}_a_{cfg.plotting_a}.png", format='PNG')

    elif cfg.dataType == "versus_energy_gap":
        # print('versus_energy_gap')
        df_energy_gap = pd.read_excel(
            f'{cfg.dataset}{cfg.dataType}/n_{cfg.nmax}_a_{cfg.plotting_a}_la_{cfg.l_a}_lb_{cfg.l_b}.xlsx'
        )
        plot_energy_gap(
            df=df_energy_gap,
            a=cfg.plotting_a,
            label_font_size=cfg.label_font_size,
            legend_font_size=cfg.legend_font_size,
            bbox_anchor=cfg.bbox_anchor
        )
        plt.savefig(
            f"{cfg.output}{cfg.dataType}/n_{cfg.nmax}_a_{cfg.plotting_a}_la_{cfg.l_a}_lb_{cfg.l_b}.png", format="PNG"
        )
