import mpmath
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import pandas as pd
import time
import logging
from mpmath import mp
# np.seterr(all='warn')


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
    def __init__(self, a, la, lb, omega, nmin, nmax, mmin, mmax):
        self.a = a
        # self.sigma = sigma
        self.la = la
        self.lb = lb
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
            return np.exp(-((self.omega-z))**2) * np.sin(2*z*(np.arcsinh(self.a*(self.la*n-lb*m)/2))/self.a) / np.tanh(np.pi*z/self.a)

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
        # lab_values = []
        # for lb in self.lb:
        lb = self.lb
        gamma = lb/self.la
        gamma = np.around(gamma, decimals=4)
        p_m = 0
        i_1 = 0
        i_2 = 0
        # print(r"$\gamma = $", gamma)
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

        lab_values = p_m + i_1 + i_2
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
            return np.exp(-((self.omega-z))**2) * np.sin(2*z*(np.arcsinh(self.a*(l*k)/2))/self.a) / np.tanh(np.pi*z/self.a)

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
        # pd_values = []
        # if type(self.l_values) == int:
        if self.space_type == 'A':
            l = self.l_values
            i_1 = 0
            i_2 = 0
            print(r"$Space_type = $", self.space_type)
            # print(r"$L values = $", l)
            # logging.info(f"Omega = {self.omega}$")

            p_m = self.pm()
            for k in range(self.kmin, self.kmax+1):
                if k != 0:
                    prefactor = (1 / (4*np.sqrt(np.pi)*(l*k))) * (1 / (np.sqrt(self.a*self.a*(l*k)
                                                                               * (l*k)/4 + 1)))
                    i_1 += prefactor * self.i1(l, k)
                    # print(l)
                    i_2 += prefactor * self.i2(l, k)

            pd_values = p_m + i_1 + i_2
            print(p_m, i_1, i_2)

        elif self.space_type == 'B':
            # for l in self.l_values:
            l = self.l_values
            i_1 = 0
            i_2 = 0
            print(r"$Space_type = $", self.space_type)
            # print(r"$L values = $", l)
            # logging.info(f"Omega = {self.omega}$")

            p_m = self.pm()
            for k in range(self.kmin, self.kmax+1):
                if k != 0:
                    prefactor = (1 / (4*np.sqrt(np.pi)*(l*k))) * (1 / (np.sqrt(self.a*self.a*(l*k)
                                                                               * (l*k)/4 + 1)))
                    i_1 += prefactor * self.i1(l, k)
                    # print(l)
                    i_2 += prefactor * self.i2(l, k)

            pd_values = p_m + i_1 + i_2
            print(p_m, i_1, i_2)
            logging.info(f"P_M: {p_m}\t I_1 + I_2: {i_1+i_2}")

        return pd_values
