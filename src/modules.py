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


class AreaSingleTerms ():
    def __init__(self, a, ld, omega, kmin, kmax, kpmin, kpmax):
        self.a = a
        self.ld = ld
        self.omega = omega
        self.kmin = kmin
        self.kmax = kmax
        self.kpmin = kpmin
        self.kpmax = kpmax

    def func_s_pm(self, s):
        return np.exp(-s*s/4) * np.cos(self.omega*s) / ((np.sinh(self.a*s/2))**2) - 4/(self.a*self.a*s*s)

    def pm(self):
        term1 = -self.omega/(4*np.pi)
        term2 = -self.a*self.a/(8*np.pi*np.pi) * quad(self.func_s_pm,
                                                      a=0, b=np.inf, limit=100)[0]
        # term2 = -self.a*self.a/(8*np.pi*np.pi) * trapezoidal_integrator(self.func_s_pm,
        #                                                                 a=0.000001, b=1000, num_points=30000)
        return term1 + term2

    def i1_single(self, k):
        kd = self.a*self.ld*k/2
        # factor = (1/(2*np.pi*self.ld * k)) * (1/np.sqrt(1 + kd**2))
        # factor = (1/(4*np.pi*self.ld * k)) * (1/np.sqrt(1 + kd**2))
        mainterm = np.exp(-(np.arcsinh(kd)/self.a)**2) * \
            np.sin(2*self.omega*np.arcsinh(kd)/self.a)
        # return factor*mainterm
        return - mainterm

    def func_z_i2_single(self, z, kd, rand):
        return np.exp(-(self.omega-z)**2) * np.sin(2*z*np.arcsinh(kd)/self.a) / np.tanh(np.pi*z/self.a)

    def i2_single(self, k):
        # rand = 1
        kd = self.a*self.ld*k/2
        # factor = (1 / (4*np.pi*np.sqrt(np.pi)*self.ld*k)) * \
        #     (1/np.sqrt(1 + kd**2))
        integral_term = trapezoidal_integrator(
            self.func_z_i2_single, a=-100, b=100, num_points=30000, args=(kd, 0))
        # return factor * integral_term
        return integral_term/np.sqrt(np.pi)

    def i1_double(self, k, kp):
        kd = self.a*self.ld*np.sqrt(k*k + kp*kp) / 2
        # mainterm = np.exp(-(np.arcsinh(kd) / self.a)**2) * \
        #     np.sin(2*self.omega*np.arcsinh(kd)/self.a)
        mainterm = np.exp(-(np.arcsinh(kd)/self.a)**2) * \
            np.sin(2*self.omega*np.arcsinh(kd)/self.a)

        return - mainterm

    def func_z_i2_double(self, z, kd, rand):
        # return np.exp(-(self.omega-z)**2) * np.sin(2*z*np.arcsinh(kd)/self.a) / np.tanh(np.pi*z/self.a)
        return np.exp(-(self.omega-z)**2) * np.sin(2*z*np.arcsinh(kd)/self.a) / np.tanh(np.pi*z/self.a)

    def i2_double(self, k, kp):
        kd = self.a*self.ld*np.sqrt(k*k + kp*kp) / 2
        # factor = (1/(4*np.pi*np.sqrt(np.pi)*self.ld *
        #           np.sqrt(k*k + kp*kp))) * (1/(np.sqrt(1 + kd**2)))
        # integral_term = trapezoidal_integrator(
        #     self.func_z_i2_double, a=-100, b=100, num_points=30000, args=(kd, 0))
        integral_term = trapezoidal_integrator(
            self.func_z_i2_single, a=-100, b=100, num_points=30000, args=(kd, 0))

        # return factor*integral_term
        return integral_term/np.sqrt(np.pi)

    def pd_values(self):
        val = 0
        pm_val = self.pm()
        i1_val = 0
        i2_val = 0
        i1_val2 = 0
        i2_val2 = 0
        i1_double_val = 0
        i2_double_val = 0
        i1_double_val2 = 0
        i2_double_val2 = 0

        for k in range(self.kmin, self.kmax+1):
            kd = self.a*self.ld*k/2
            if k != 0:
                prefactor = (1 / (4*np.pi*self.ld*k * np.sqrt(1 + kd*kd)))
                i1_val += prefactor*self.i1_single(k)
                i2_val += prefactor*self.i2_single(k)

        # for kp in range(self.kpmin, self.kpmax+1):
        #     kd = self.a*self.ld*kp/2
        #     if kp != 0:
        #         prefactor = (1 / (4*np.pi*self.ld*kp * np.sqrt(1 + kd*kd)))
        #         i1_val2 += prefactor*self.i1_single(kp)
        #         i2_val2 += prefactor*self.i2_single(kp)

        for k in range(self.kmin, self.kmax+1):
            for kp in range(self.kpmin, self.kpmax+1):
                kd = self.a * self.ld * np.sqrt(k*k + kp*kp)/2
                if k != 0 and kp != 0:
                    prefactor = (
                        1 / (4*np.pi*self.ld*np.sqrt(k*k + kp*kp) * np.sqrt(1 + kd*kd)))
                    i1_double_val += prefactor*self.i1_double(k, kp)
                    i2_double_val += prefactor*self.i2_double(k, kp)

        # for k in range(self.kmin, self.kmax+1):
        #     for kp in range(self.kpmin, self.kpmax+1):
        #         if k != 0 and kp != 0:
        #             kd = self.a * self.ld * np.sqrt(k*k + kp*kp)
        #             prefactor = (
        #                 1 / (4*np.pi*self.ld*np.sqrt(k*k + kp*kp) * np.sqrt(1 + kd*kd)))
        #             i1_double_val2 += prefactor*self.i1_double(k, kp)
        #             i2_double_val2 += prefactor*self.i2_double(k, kp)

        val = (pm_val + 2*i1_val + 2*i2_val + 2*i1_double_val +
               2*i2_double_val) * np.sqrt(np.pi)
        print(pm_val, i1_val, i2_val, i1_val2, i2_val2, i1_double_val,
              i2_double_val, '\n', val)

        return val


class AreaCrossTerms ():
    def __init__(self, a: float, la: float, lb: float, omega: float, nmin: float, nmax: float, mmin: float, mmax: float, npmin: float, npmax: float, mpmin: float, mpmax: float):
        self.a = a
        self.la = la
        self.lb = lb
        self.omega = omega
        self.nmin = nmin
        self.nmax = nmax
        self.mmin = mmin
        self.mmax = mmax
        self.npmin = npmin
        self.npmax = npmax
        self.mpmin = mpmin
        self.mpmax = mpmax

    def func_s_pm(self, s: float) -> float:
        return np.exp(-s*s/4) * np.cos(self.omega*s) / ((np.sinh(self.a*s/2))**2) - 4/(self.a*self.a*s*s)

    def pm(self) -> float:
        term1 = -self.omega/(4*np.pi)
        term2 = -self.a*self.a/(8*np.pi*np.pi) * quad(self.func_s_pm,
                                                      a=0, b=np.inf, limit=100)[0]
        return term1 + term2

    def i1_single(self, n: int, m: int) -> float:
        kd = self.a * (self.la*n - self.lb*m) / 2
        mainterm = np.exp(-(np.arcsinh(kd)/self.a)**2) * \
            np.sin(2*self.omega*np.arcsinh(kd)/self.a)
        return - mainterm

    def func_z_i2_single(self, z: float, kd: float, rand) -> float:
        return np.exp(-(self.omega-z)**2) * np.sin(2*z*np.arcsinh(kd)/self.a) / np.tanh(np.pi*z/self.a)

    def i2_single(self, n: int, m: int) -> float:
        kd = self.a * (self.la*n - self.lb*m) / 2
        integral_term = trapezoidal_integrator(
            self.func_z_i2_single, a=-100, b=100, num_points=30000, args=(kd, 0))
        return integral_term/np.sqrt(np.pi)

    def i1_double(self, n: int, m: int, nn: int, mm: int) -> float:
        kd = self.a * np.sqrt((self.la*n - self.lb*m) **
                              2 + (self.la*nn - self.lb*mm)**2) / 2
        mainterm = np.exp(-(np.arcsinh(kd)/self.a)**2) * \
            np.sin(2*self.omega*np.arcsinh(kd)/self.a)
        return - mainterm

    def func_z_i2_double(self, z: float, kd: float, rand) -> float:
        return np.exp(-(self.omega-z)**2) * np.sin(2*z*np.arcsinh(kd)/self.a) / np.tanh(np.pi*z/self.a)

    def i2_double(self, n: int, m: int, nn: int, mm: int) -> float:
        kd = self.a * np.sqrt((self.la*n - self.lb*m) **
                              2 + (self.la*nn - self.lb*mm)**2) / 2
        integral_term = trapezoidal_integrator(
            self.func_z_i2_single, a=-100, b=100, num_points=30000, args=(kd, 0))
        return integral_term/np.sqrt(np.pi)

    def pd_values(self):
        val = 0
        pm_val = 0
        i1_val = 0
        i2_val = 0
        i1_val2 = 0
        i2_val2 = 0
        i1_double_val = 0
        i2_double_val = 0
        i1_double_val2 = 0
        i2_double_val2 = 0

        for n in range(self.nmin, self.nmax+1):
            for m in range(self.mmin, self.mmax+1):
                for nn in range(self.npmin, self.npmax+1):
                    for mm in range(self.mpmin, self.mpmax+1):
                        # print('working brooo!')
                        kd = self.a * \
                            np.sqrt((self.la*n - self.lb*m)**2 +
                                    (self.la*nn - self.lb*mm)**2) / 2
                        if self.la*n == self.lb*m and self.la*nn == self.lb*mm:
                            pm_val += self.pm()
                        if self.la*n == self.lb*m and self.la*nn != self.lb*mm:
                            prefactor = (
                                1 / (4*np.pi*(self.la*nn - self.lb*mm) * np.sqrt(1 + kd*kd)))
                            i1_val += prefactor*self.i1_single(nn, mm)
                            i2_val += prefactor*self.i2_single(nn, mm)
                        if self.la*n != self.lb*m and self.la*nn == self.lb*mm:
                            prefactor = (
                                1 / (4*np.pi*(self.la*n - self.lb*m) * np.sqrt(1 + kd*kd)))
                            i1_val2 += prefactor*self.i1_single(n, m)
                            i2_val2 += prefactor*self.i2_single(n, m)
                        if self.la*n != self.lb*m and self.la*nn != self.lb*mm:
                            prefactor = (1 / (4*np.pi*np.sqrt((self.la*n - self.lb*m)
                                         ** 2 + (self.la*nn - self.lb*mm)**2) * np.sqrt(1 + kd*kd)))
                            i1_double_val += prefactor * \
                                self.i1_double(n, m, nn, mm)
                            i1_double_val2 += prefactor * \
                                self.i1_double(n, m, nn, mm)
                            i2_double_val += prefactor * \
                                self.i2_double(n, m, nn, mm)
                            i2_double_val2 += prefactor * \
                                self.i2_double(n, m, nn, mm)

        sum = 0
        for n in range(self.nmin, self.nmax+1):
            for nn in range(self.npmin, self.mpmax+1):
                sum += 1

        val = (pm_val + i1_val + i2_val + i1_val2 + i2_val2 + i1_double_val +
               i2_double_val + i1_double_val2 + i2_double_val2) * np.sqrt(np.pi) / sum
        print(pm_val, i1_val, i2_val, i1_val2, i2_val2, i1_double_val,
              i2_double_val, i1_double_val2, i2_double_val2, '\n', val)

        return val


om_range = np.arange(-100, 5+1, step=1)/2
# om = 1
# acc_range = np.arange(1e-3, 100, step=1)
a = 0.001
nmax = 7


pa_vals = []
pb_vals = []
lab_vals = []
pe_plus_vals = []
pe_minus_vals = []
for om in om_range:
    start_time = time.time()

    print('omega is :: ', om)
    la_term = AreaSingleTerms(
        a=a, ld=0.25, omega=om, kmin=-nmax, kmax=nmax, kpmin=-nmax, kpmax=nmax)
    lb_term = AreaSingleTerms(
        a=a, ld=0.75, omega=om, kmin=-nmax, kmax=nmax, kpmin=-nmax, kpmax=nmax)
    lab_term = AreaCrossTerms(a=a, la=0.25, lb=0.75, omega=om, nmin=-nmax,
                              nmax=nmax, mmin=-nmax, mmax=nmax, npmin=-nmax, npmax=nmax, mpmin=-nmax, mpmax=nmax)
    pa_val = la_term.pd_values()
    pa_vals.append(pa_val)
    pb_val = lb_term.pd_values()
    pb_vals.append(pb_val)
    lab_val = lab_term.pd_values()
    lab_vals.append(lab_val)
    pe_plus_vals.append((pa_val+pb_val+2*lab_val)/4)
    pe_minus_vals.append((pa_val+pb_val-2*lab_val)/4)

    end_time = time.time()
    print("Time taken (for 1 iteration) in seconds :: ", end_time-start_time)


# plt.figure(figsize=(9, 6), dpi=300)

plt.plot(om_range, pa_vals, label=r"$P_A, l_A=0.25$", lw=1)
plt.plot(om_range, pb_vals, label=r"$P_B, l_B=0.75$", lw=1)
plt.plot(om_range, lab_vals, label=r"$L_{AB}$", lw=1)
plt.plot(om_range, pe_plus_vals, label=r"$P_E ^{+}$", lw=1)
plt.plot(om_range, pe_minus_vals, label=r"$P_E ^{-}$", lw=1)
plt.xlabel(
    r"Energy gap, $\mathregular{\Omega \sigma}$", fontsize=14)
plt.ylabel(
    r"Transition probability, $\mathregular{P_E/\,\lambda^2}$", fontsize=14)
plt.title("Acceleration: "+r"$a\sigma = " +
          str(a) + r"$", fontsize=16)

plt.legend()
plt.grid()
plt.tight_layout()
# plt.show()
plt.savefig(f'area_quantization_a_{a}.png', format="PNG")
plt.close()
