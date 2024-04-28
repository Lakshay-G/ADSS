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

if os.path.exists("/Users/lakshaygoel/Documents/Fall 2023/Phys 437A/python_code/example.log"):
    os.remove(
        "/Users/lakshaygoel/Documents/Fall 2023/Phys 437A/python_code/example.log")

logging.basicConfig(filename='/Users/lakshaygoel/Documents/Fall 2023/Phys 437A/python_code/example.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def trapezoidal_integrator(integrand, a=0, b=1, num_points=100, args=None):
    # Generate equally spaced points
    integral_approx = mpmath.mpf(0)
    if args is None:
        args = ()
    x_values = np.linspace(a, b, num_points)

    # Evaluate the integrand at these points
    y_values = integrand(x_values, *args)

    # Calculate the width of each trapezoid
    dx = (b - a) / (num_points - 1)

    # Compute the integral using the trapezoidal rule
    integral_approx = dx * (np.sum(y_values) - 0.5 *
                            (y_values[0] + y_values[-1]))

    return integral_approx


class CrossTerms ():
    def __init__(self, a, la, lb_values, omega, nmin, nmax, mmin, mmax):
        self.a = a
        # self.sigma = sigma
        self.la = la
        self.lb_values = lb_values
        self.omega = omega
        self.nmin = nmin
        self.nmax = nmax
        self.mmin = mmin
        self.mmax = mmax

    def func_s_pm(self, s):
        if self.a == 0:
            return (np.exp(-s*s/(4)) * np.cos(self.omega*s) - 1) * 4/(s*s)
        else:
            return np.exp(-s*s/(4)) * np.cos(self.omega*s) * self.a*self.a/((np.sinh(self.a*s/2))**2) - 4/(s*s)

    def pm(self):
        # val = quad(self.func_s_pm,
        #            a=0, b=np.inf, limit=100)[0]
        val = trapezoidal_integrator(
            self.func_s_pm, a=0.0000001, b=200, num_points=30000)

        temp = - self.omega/(4*np.pi) - ((1/(8*np.pi*np.pi)) * val)
        return np.sqrt(np.pi) * temp

    def func_z_i2(self, z, lb, n, m):
        if self.a == 0:
            return (np.exp(-((self.omega-z))**2) + np.exp(-((self.omega+z))**2)) * np.sin(z*(self.la*n-lb*m))
        else:
            return np.exp(-((self.omega-z))**2) * np.sin(2*z*(np.arcsinh(a*(self.la*n-lb*m)/2))/self.a) / np.tanh(np.pi*z/self.a)

    def i2(self, lb, n, m):
        if self.a == 0:
            val = trapezoidal_integrator(
                self.func_z_i2, a=0, b=200, num_points=30000, args=(lb, n, m))
        else:
            val = trapezoidal_integrator(
                self.func_z_i2, a=-100, b=100, num_points=30000, args=(lb, n, m))

        val = val / np.sqrt(np.pi)
        return val

    def i1(self, lb, n, m):
        if self.a == 0:
            return - np.exp(-((self.la*n-lb*m)**2)/(4)) * np.sin(self.omega*(self.la*n-lb*m))
        else:
            return - np.exp(-((np.arcsinh(self.a*(self.la*n-lb*m)/2))**2)/(self.a*self.a)) * np.sin(2*self.omega*(np.arcsinh(self.a*(self.la*n-lb*m)/2))/self.a)

    def lab_values(self):
        lab_values = []
        for lb in self.lb_values:
            gamma = lb/self.la
            gamma = np.around(gamma, decimals=4)
            p_m = 0
            i_1 = 0
            i_2 = 0
            print(r"$\gamma = $", gamma)
            logging.info(f"Omega = {self.omega}$")

            for n in range(self.nmin, self.nmax+1):
                for m in range(self.mmin, self.mmax+1):
                    if self.la*n == lb*m:
                        p_m += self.pm()

                    elif self.la*n != lb*m:
                        prefactor = (1 / (4*np.sqrt(np.pi)*(self.la*n-lb*m))) * \
                            (1 / (np.sqrt(self.a*self.a*(self.la*n-lb*m)
                             * (self.la*n-lb*m)/4 + 1)))
                        i_1 += prefactor * self.i1(lb, n, m)
                        i_2 += prefactor * self.i2(lb, n, m)

            lab_values.append(p_m + i_1 + i_2)
            print(p_m, i_1, i_2)
            logging.info(f"P_M: {p_m}\t I_1 + I_2: {i_1+i_2}")

        return lab_values


class SingleTerms ():
    def __init__(self, a, l_values, omega, kmin, kmax, space_type):
        self.a = a
        # self.sigma = sigma
        self.l_values = l_values
        self.omega = omega
        self.kmin = kmin
        self.kmax = kmax
        self.space_type = space_type

    def func_s_pm(self, s):
        if self.a == 0:
            return (np.exp(-s*s/(4)) * np.cos(self.omega*s) - 1) * 4/(s*s)
        else:
            return np.exp(-s*s/(4)) * np.cos(self.omega*s) * self.a*self.a/((np.sinh(self.a*s/2))**2) - 4/(s*s)

    def pm(self):
        val = quad(self.func_s_pm,
                   a=0, b=np.inf, limit=100)[0]

        temp = - self.omega/(4*np.pi) - ((1/(8*np.pi*np.pi)) * val)
        return np.sqrt(np.pi) * temp

    def func_z_i2(self, z, l, k):
        if self.a == 0:
            return (np.exp(-((self.omega-z))**2) + np.exp(-((self.omega+z))**2)) * np.sin(z*l*k)
        else:
            return np.exp(-((self.omega-z))**2) * np.sin(2*z*(np.arcsinh(a*(l*k)/2))/self.a) / np.tanh(np.pi*z/self.a)

    def i2(self, l, k):
        if self.a == 0:
            val = trapezoidal_integrator(
                self.func_z_i2, a=0, b=200, num_points=30000, args=(l, k))
        else:
            val = trapezoidal_integrator(
                self.func_z_i2, a=-100, b=100, num_points=30000, args=(l, k))

        val = val / np.sqrt(np.pi)
        return val

    def i1(self, l, k):
        if self.a == 0:
            return - np.exp(-((l*k)**2)/(4)) * np.sin(self.omega*(l*k))
        else:
            return - np.exp(-((np.arcsinh(self.a*(l*k)/2))**2)/(self.a*self.a)) * np.sin(2*self.omega*(np.arcsinh(self.a*(l*k)/2))/self.a)

    def pd_values(self):
        pd_values = []
        # if type(self.l_values) == int:
        if self.space_type == 'A':
            l = self.l_values
            i_1 = 0
            i_2 = 0
            print(r"$Space_type = $", self.space_type)
            print(r"$L values = $", l)
            # logging.info(f"Omega = {self.omega}$")

            p_m = self.pm()
            for k in range(self.kmin, self.kmax+1):
                if k != 0:
                    prefactor = (1 / (4*np.sqrt(np.pi)*(l*k))) * (1 / (np.sqrt(self.a*self.a*(l*k)
                                                                               * (l*k)/4 + 1)))
                    i_1 += prefactor * self.i1(l, k)
                    # print(l)
                    i_2 += prefactor * self.i2(l, k)

            pd_values.append(p_m + i_1 + i_2)
            print(p_m, i_1, i_2)

        elif self.space_type == 'B':
            for l in self.l_values:
                i_1 = 0
                i_2 = 0
                print(r"$Space_type = $", self.space_type)
                print(r"$L values = $", l)
                # logging.info(f"Omega = {self.omega}$")

                p_m = self.pm()
                for k in range(self.kmin, self.kmax+1):
                    if k != 0:
                        prefactor = (1 / (4*np.sqrt(np.pi)*(l*k))) * (1 / (np.sqrt(self.a*self.a*(l*k)
                                                                                   * (l*k)/4 + 1)))
                        i_1 += prefactor * self.i1(l, k)
                        # print(l)
                        i_2 += prefactor * self.i2(l, k)

                pd_values.append(p_m + i_1 + i_2)
                print(p_m, i_1, i_2)
                logging.info(f"P_M: {p_m}\t I_1 + I_2: {i_1+i_2}")

        return pd_values


if __name__ == "__main__":
    nmax = 10
    nmin = -nmax
    mmax, mmin = nmax, nmin
    # sigma = 1/10
    a_vals = np.arange(0, 0.2, step=0.2)
    omega = 0.01
    step = 0.1

    # for fig. 1
    fig = 1
    l_b_values_rough = np.arange(5, 17.6, step=step)
    l_b_values = np.around(l_b_values_rough, decimals=2)
    l_a = 10

    # for fig. 2
    # fig = 2
    # l_b_values_rough = np.arange(15, 20.0, step=step)
    # l_a = 20
    # l_b_values = np.around(l_b_values_rough, decimals=2)

    for a in a_vals:
        print(f"The value of a now is: {a}")

        start_time = time.time()
        sum = 0
        for n in range(nmin, nmax+1):
            sum += (-1)**(2*n)

        cross_term = CrossTerms(
            a=a, la=l_a, lb_values=l_b_values, omega=omega, nmin=nmin, nmax=nmax, mmin=mmin, mmax=mmax)
        la_term = SingleTerms(a=a, l_values=l_a,
                              omega=omega, kmin=nmin, kmax=nmax, space_type='A')
        lb_term = SingleTerms(
            a=a, l_values=l_b_values, omega=omega, kmin=nmin, kmax=nmax, space_type='B')
        lab_values = np.array(cross_term.lab_values()) / sum
        print(lab_values)
        # pa_values = np.array(la_term.pd_values())
        pa_values = np.full((len(l_b_values),), la_term.pd_values())
        print(pa_values)
        pb_values = np.array(lb_term.pd_values())
        print(pb_values)
        gamma_vals = l_b_values/l_a
        gamma_vals = np.around(gamma_vals, decimals=4)
        # print(lab_values.shape, np.size(pa_values), np.size(pb_values))
        pe_plus_values = (pa_values + pb_values + 2*lab_values) / 4
        pe_minus_values = (pa_values + pb_values - 2*lab_values) / 4

        plt.figure(figsize=(8, 6))
        if fig == 1:
            plt.plot(gamma_vals, lab_values, '-g', label=r"$L_{AB}$", lw=1)
            plt.plot(gamma_vals, pa_values, '-y', label=r"$P_A$", lw=1)
            plt.plot(gamma_vals, pb_values, '-r', label=r"$P_B$", lw=1)
            plt.plot(gamma_vals, pe_plus_values, '-b',
                     label=r"$P_E ^{+}$", lw=1)
            # plt.plot(gamma_vals, pe_minus_values, '-k', label=r"$P_E ^{-}$")
        else:
            plt.plot(gamma_vals, pe_plus_values, '-b', label=r"$P_E ^{+}$")
        plt.grid()
        # plt.title(f"a = {a}, sigma = {sigma}")
        # plt.title(f"a = {a}, omega = {omega}")
        plt.xlabel(
            r"Length ratio, $\,\mathregular{\gamma = \frac{l_B}{l_A}}$", fontsize=14)
        plt.ylabel(
            r"Transition probability, $\,\mathregular{P_E/\,\lambda^2}$", fontsize=14)
        if fig == 2:
            plt.ylim(0.047, 0.054)
        plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
        plt.savefig(
            f"fig_{fig}_n_{nmax}_a_{a}_dash_step_{step}.png", format="PNG")
        plt.show()

        end_time = time.time()
        print("This process took :: ", end_time-start_time, " seconds")
        logging.info(f"This process took :: {end_time-start_time} seconds")
        df = pd.DataFrame([gamma_vals, l_b_values, pa_values, pb_values,
                          lab_values, pe_plus_values, pe_minus_values])
        df.to_excel(f"fig_{fig}_data_for_a_{a}_n_{nmax}_step_{step}.xlsx")
