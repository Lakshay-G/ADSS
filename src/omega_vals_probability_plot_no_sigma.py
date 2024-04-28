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
    def __init__(self, a, la, lb, omega_values, nmin, nmax, mmin, mmax):
        self.a = a
        # self.sigma = sigma
        self.la = la
        self.lb = lb
        self.omega_values = omega_values
        self.nmin = nmin
        self.nmax = nmax
        self.mmin = mmin
        self.mmax = mmax

    def func_s_pm(self, s, omega):
        if self.a == 0:
            return (np.exp(-s*s/(4)) * np.cos(omega*s) - 1) * 4/(s*s)
        else:
            return np.exp(-s*s/(4)) * np.cos(omega*s) * self.a*self.a/((np.sinh(self.a*s/2))**2) - 4/(s*s)

    def pm(self, omega):
        val = quad(self.func_s_pm, args=(omega),
                   a=0.000001, b=np.inf, limit=100000, epsrel=1e-6, epsabs=0)[0]
        rand = 1
        # val = trapezoidal_integrator(
        #     self.func_s_pm, a=0.001, b=200000, num_points=200000, args=(omega, rand))

        temp = - omega/(4*np.pi) - ((1/(8*np.pi*np.pi)) * val)
        return np.sqrt(np.pi) * temp

    def func_z_i2(self, z, omega, n, m):
        if self.a == 0:
            return (np.exp(-((omega-z))**2) + np.exp(-((omega+z))**2)) * np.sin(z*(self.la*n-self.lb*m))
        else:
            return np.exp(-((omega-z))**2) * np.sin(2*z*(np.arcsinh(a*(self.la*n-self.lb*m)/2))/self.a) / np.tanh(np.pi*z/self.a)

    def i2(self, omega, n, m):
        if self.a == 0:
            val = trapezoidal_integrator(
                self.func_z_i2, a=0, b=200000, num_points=200000, args=(omega, n, m))
        else:
            val = trapezoidal_integrator(
                self.func_z_i2, a=-100000, b=100000, num_points=200000, args=(omega, n, m))

        val = val / np.sqrt(np.pi)
        return val

    def i1(self, omega, n, m):
        if self.a == 0:
            return - np.exp(-((self.la*n-self.lb*m)**2)/(4)) * np.sin(omega*(self.la*n-self.lb*m))
        else:
            return - np.exp(-((np.arcsinh(self.a*(self.la*n-self.lb*m)/2))**2)/(self.a*self.a)) * np.sin(2*omega*(np.arcsinh(self.a*(self.la*n-self.lb*m)/2))/self.a)

    def lab_values(self):
        lab_values = []
        for omega in self.omega_values:
            p_m = 0
            i_1 = 0
            i_2 = 0
            print(r"$\Omega = $", omega)
            logging.info(f"Omega = {omega}$")

            for n in range(self.nmin, self.nmax+1):
                for m in range(self.mmin, self.mmax+1):
                    if self.la*n == self.lb*m:
                        p_m += self.pm(omega)

                    elif self.la*n != self.lb*m:
                        prefactor = (1 / (4*np.sqrt(np.pi)*(self.la*n-self.lb*m))) * \
                            (1 / (np.sqrt(self.a*self.a*(self.la*n-self.lb*m)
                             * (self.la*n-self.lb*m)/4 + 1)))
                        i_1 += prefactor * self.i1(omega, n, m)
                        i_2 += prefactor * self.i2(omega, n, m)

            lab_values.append(p_m + i_1 + i_2)
            print(p_m, i_1, i_2)
            logging.info(f"P_M: {p_m}\t I_1 + I_2: {i_1+i_2}")

        return lab_values


class SingleTerms ():
    def __init__(self, a, l, omega_values, kmin, kmax):
        self.a = a
        # self.sigma = sigma
        self.l = l
        self.omega_values = omega_values
        self.kmin = kmin
        self.kmax = kmax

    def func_s_pm(self, s, omega):
        if self.a == 0:
            return (np.exp(-s*s/(4)) * np.cos(omega*s) - 1) * 4/(s*s)
        else:
            return np.exp(-s*s/(4)) * np.cos(omega*s) * self.a*self.a/((np.sinh(self.a*s/2))**2) - 4/(s*s)

    def pm(self, omega):
        val = quad(self.func_s_pm, args=(omega),
                   a=0.000001, b=np.inf, limit=100000, epsrel=1e-6, epsabs=0)[0]
        rand = 1
        # print(omega)
        # val = trapezoidal_integrator(
        #     self.func_s_pm, a=0.001, b=200000, num_points=200000, args=(omega, rand))

        temp = - omega/(4*np.pi) - ((1/(8*np.pi*np.pi)) * val)
        print(np.sqrt(np.pi)*omega/(4*np.pi))
        return np.sqrt(np.pi) * temp

    def func_z_i2(self, z, omega, k):
        if self.a == 0:
            return (np.exp(-((omega-z))**2) + np.exp(-((omega+z))**2)) * np.sin(z*self.l*k)
        else:
            return np.exp(-((omega-z))**2) * np.sin(2*z*(np.arcsinh(a*(self.l*k)/2))/self.a) / np.tanh(np.pi*z/self.a)

    def i2(self, omega, k):
        if self.a == 0:
            val = trapezoidal_integrator(
                self.func_z_i2, a=0, b=200000, num_points=200000, args=(omega, k))
        else:
            val = trapezoidal_integrator(
                self.func_z_i2, a=-100000, b=100000, num_points=200000, args=(omega, k))

        val = val / np.sqrt(np.pi)
        return val

    def i1(self, omega, k):
        if self.a == 0:
            return - np.exp(-((self.l*k)**2)/(4)) * np.sin(omega*(self.l*k))
        else:
            return - np.exp(-((np.arcsinh(self.a*(self.l*k)/2))**2)/(self.a*self.a)) * np.sin(2*omega*(np.arcsinh(self.a*(self.l*k)/2))/self.a)

    def pd_values(self):
        pd_values = []
        for omega in self.omega_values:
            i_1 = 0
            i_2 = 0
            print(r"$\Omega = $", omega)
            logging.info(f"Omega = {omega}$")

            p_m = self.pm(omega)
            for k in range(self.kmin, self.kmax+1):
                if k != 0:
                    prefactor = (1 / (4*np.sqrt(np.pi)*(self.l*k))) * (1 / (np.sqrt(self.a*self.a*(self.l*k)
                                                                                    * (self.l*k)/4 + 1)))
                    i_1 += prefactor * self.i1(omega, k)
                    i_2 += prefactor * self.i2(omega, k)

            pd_values.append(p_m + i_1 + i_2)
            print(p_m, i_1, i_2)
            logging.info(f"P_M: {p_m}\t I_1 + I_2: {i_1+i_2}")

        return pd_values


if __name__ == "__main__":
    nmax = 5
    nmin = -nmax
    mmax, mmin = nmax, nmin
    a_vals = np.arange(0, 1, step=1)
    l_a = 0.75
    l_b = 0.25
    omega_values = np.arange(-100, 5+1, step=1)
    omega_values = omega_values/2
    # pa = []
    # pb = []

    for a in a_vals:
        print(f"The value of a now is: {a}")

        start_time = time.time()
        sum = 0
        for n in range(nmin, nmax+1):
            sum += (-1)**(2*n)

        # cross_term = CrossTerms(
        #     a, sigma, l_a, l_b, omega_values, nmin, nmax, mmin, mmax)
        # la_term = SingleTerms(a, sigma, l_a, omega_values, nmin, nmax)
        # lb_term = SingleTerms(a, sigma, l_b, omega_values, nmin, nmax)
        cross_term = CrossTerms(
            a, l_a, l_b, omega_values, nmin, nmax, mmin, mmax)
        la_term = SingleTerms(a, l_a, omega_values, nmin, nmax)
        lb_term = SingleTerms(a, l_b, omega_values, nmin, nmax)
        lab_values = np.array(cross_term.lab_values()) / sum
        pa_values = np.array(la_term.pd_values())
        # pa.append(la_term.pd_values())
        pb_values = np.array(lb_term.pd_values())
        # pb.append(lb_term.pd_values())
        print(lab_values.shape, np.size(pa_values), np.size(pb_values))
        pe_plus_values = (pa_values + pb_values + 2*lab_values) / 4
        pe_minus_values = (pa_values + pb_values - 2*lab_values) / 4

        # save data before crashing lol
        df = pd.DataFrame([omega_values, pa_values, pb_values,
                          lab_values, pe_plus_values, pe_minus_values])
        df.to_excel(f"omega_data_for_a_{a}_n_{nmax}_la_{l_a}_lb_{l_b}.xlsx")

        # plt.figure(figsize=(6, 3))
        plt.plot(omega_values, lab_values, '-g', label=r"$L_{AB}$")
        plt.plot(omega_values, pa_values, '-y', label=r"$P_A$")
        plt.plot(omega_values, pb_values, '-r', label=r"$P_B$")
        plt.plot(omega_values, pe_plus_values, '-b', label=r"$P_E ^{+}$")
        plt.plot(omega_values, pe_minus_values, '-k', label=r"$P_E ^{-}$")
        plt.grid()
        # plt.title(f"a = {a}, sigma = {sigma}, l_a = {l_a}, l_b = {l_b}")
        # plt.title(f"a = {a}, l_a = {l_a}, l_b = {l_b}")
        plt.xlabel(r"Energy gap, $\mathregular{\Omega \sigma}$", fontsize=14)
        plt.ylabel(
            r"Transition probability, $\mathregular{P_E/\,\lambda^2}$", fontsize=14)
        # plt.ylim(0, 14)
        plt.legend(loc='upper right', fontsize=12)
        plt.savefig(
            f"omega_no_quad_n_{nmax}_a_{a}_dot_la_{l_a}_lb_{l_b}.png", format="PNG")
        plt.close()

        end_time = time.time()
        print("This process took :: ", end_time-start_time, " seconds")
        logging.info(f"This process took :: {end_time-start_time} seconds")

    # plt.plot(pa, '.r')
    # plt.plot(pb, '.k')
    # plt.show()
