import mpmath
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import pandas as pd
import time
import logging
from mpmath import mp
from modules import CrossTerms, SingleTerms


class Config():
    dataset = 'dataset/'
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

    elif dataType == "versus_energy_gap":
        l_a = 0.75
        l_b = 0.25
        omega_values = np.arange(-100, 5+1, step=1)
        omega_values = omega_values/2
        a_vals = [0.0]

    elif dataType == "versus_acceleration":
        l_a = 0.75
        l_b = 0.25
        omega_vals = [-50]
        step_acceleration = 1
        a_temp_values = np.arange(
            0, 100+step_acceleration, step=step_acceleration)
        a_vals = np.around(a_temp_values, decimals=1)


if __name__ == '__main__':
    cfg = Config()
    nmax, nmin, mmax, mmin = cfg.nmax, cfg.nmin, cfg.mmax, cfg.mmin
    dataset = cfg.dataset

    if cfg.dataType == 'versus_length_ratio':
        # a_vals = np.arange(0, 0.2, step=0.2)
        # a = 0.0
        a_vals, omega, l_a, l_b_vals, gamma_vals, step_length_ratio = cfg.a_vals, cfg.omega, cfg.l_a, cfg.l_b_vals, cfg.gamma_vals, cfg.step_length_ratio
        sum = cfg.sum

        for a in a_vals:
            pa_values = []
            pb_values = []
            lab_values = []
            pe_plus_values = []
            pe_minus_values = []
            for l_b in l_b_vals:
                cross_term = CrossTerms(
                    a=a, la=l_a, lb=l_b, omega=omega, nmin=nmin, nmax=nmax, mmin=mmin, mmax=mmax)
                la_term = SingleTerms(a=a, l_values=l_a,
                                      omega=omega, kmin=nmin, kmax=nmax, space_type='A')
                lb_term = SingleTerms(
                    a=a, l_values=l_b, omega=omega, kmin=nmin, kmax=nmax, space_type='B')

                lab_val = cross_term.lab_values()/sum
                lab_values.append(lab_val)
                pa_val = la_term.pd_values()
                pa_values.append(pa_val)
                pb_val = lb_term.pd_values()
                pb_values.append(pb_val)
                pe_plus_values.append(
                    (pa_val + pb_val + 2*lab_val) / 4)
                pe_minus_values.append(
                    (pa_val + pb_val - 2*lab_val) / 4)

            plt.plot(gamma_vals, pe_plus_values)
            plt.plot(gamma_vals, pe_minus_values)
            plt.plot(gamma_vals, lab_values)
            plt.plot(gamma_vals, pa_values)
            plt.plot(gamma_vals, pb_values)
            plt.show()
            df = pd.DataFrame([gamma_vals, l_b_vals, pa_values, pb_values, lab_values,
                               pe_plus_values, pe_minus_values])
            df.to_excel(
                f"{dataset}{cfg.dataType}/n_{nmax}_omega_{omega}_step_{step_length_ratio}_a_{a}.xlsx")

    elif cfg.dataType == "versus_energy_gap":
        a_vals, omega_vals, l_a, l_b = cfg.a_vals, cfg.omega_values, cfg.l_a, cfg.l_b
        sum = cfg.sum
        for a in a_vals:
            lab_values = []
            pa_values = []
            pb_values = []
            pe_plus_values = []
            pe_minus_values = []
            for omega in omega_vals:
                cross_term = CrossTerms(
                    a, l_a, l_b, omega, nmin, nmax, mmin, mmax)
                la_term = SingleTerms(
                    a, l_a, omega, nmin, nmax, space_type='A')
                lb_term = SingleTerms(
                    a, l_b, omega, nmin, nmax, space_type='B')

                lab_val = cross_term.lab_values()/sum
                lab_values.append(lab_val)
                pa_val = la_term.pd_values()
                pa_values.append(pa_val)
                pb_val = lb_term.pd_values()
                pb_values.append(pb_val)
                pe_plus_values.append(
                    (pa_val + pb_val + 2*lab_val) / 4)
                pe_minus_values.append(
                    (pa_val + pb_val - 2*lab_val) / 4)

            plt.plot(omega_vals, lab_values)
            plt.plot(omega_vals, pa_values)
            plt.plot(omega_vals, pb_values)
            plt.plot(omega_vals, pe_plus_values)
            plt.plot(omega_vals, pe_minus_values)
            plt.show()
            df = pd.DataFrame([omega_vals, pa_values, pb_values,
                              lab_values, pe_plus_values, pe_minus_values])
            df.to_excel(
                f"{dataset}{cfg.dataType}/n_{nmax}_a_{a}_la_{l_a}_lb_{l_b}.xlsx")

    elif cfg.dataType == "versus_acceleration":
        # steps = 1
        a_vals, omega_vals, l_a, l_b, step_acceleration = cfg.a_vals, cfg.omega_vals, cfg.l_a, cfg.l_b, cfg.step_acceleration
        sum = cfg.sum
        for omega in omega_vals:
            print(omega)
            lab_values = []
            pa_values = []
            pb_values = []
            pe_plus_values = []
            pe_minus_values = []
            for a in a_vals:
                print(a)
                cross_term = CrossTerms(
                    a, l_a, l_b, omega, nmin, nmax, mmin, mmax)
                la_term = SingleTerms(
                    a, l_a, omega, nmin, nmax, space_type='A')
                lb_term = SingleTerms(
                    a, l_b, omega, nmin, nmax, space_type='B')

                lab_val = cross_term.lab_values()/sum
                lab_values.append(lab_val)
                pa_val = la_term.pd_values()
                pa_values.append(pa_val)
                pb_val = lb_term.pd_values()
                pb_values.append(pb_val)
                pe_plus_values.append(
                    (pa_val + pb_val + 2*lab_val) / 4)
                pe_minus_values.append(
                    (pa_val + pb_val - 2*lab_val) / 4)

            plt.plot(a_vals, lab_values)
            plt.plot(a_vals, pa_values)
            plt.plot(a_vals, pb_values)
            plt.plot(a_vals, pe_plus_values)
            plt.plot(a_vals, pe_minus_values)
            plt.show()
            df = pd.DataFrame([a_vals, pa_values, pb_values,
                              lab_values, pe_plus_values, pe_minus_values])
            df.to_excel(
                f"{dataset}{cfg.dataType}/n_{nmax}_amax_{a_vals[-1]}_omega_{omega}_step_{step_acceleration}.xlsx")
