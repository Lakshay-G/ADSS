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
    def __init__(self, a, la, lb, omega_values, nmin, nmax, mmin, mmax):
        self.a = a
        self.la = la
        self.lb = lb
        self.omega_values = omega_values
        self.nmin = nmin
        self.nmax = nmax
        self.mmin = mmin
        self.mmax = mmax

    def func_s_pm(self, s, omega, rand):
        mp.dps = 50
        if self.a == 0:
            return (mp.exp(-s*s/(4)) * mp.cos(omega*s) - 1) * 4/(s*s)
        else:
            return mp.exp(-s*s/(4)) * mp.cos(omega*s) * self.a*self.a/((mp.sinh(self.a*s/2))**2) - 4/(s*s)

    def pm(self, omega):
        rand = 1
        val = quad(self.func_s_pm, args=(omega, rand),
                   a=0.000001, b=np.inf, limit=100000, epsrel=1e-6, epsabs=0)[0]
        # val = trapezoidal_integrator(
        #     self.func_s_pm, a=0.0001, b=100000, num_points=4000000, args=(omega, rand))

        temp = - omega/(4*np.pi) - ((1/(8*np.pi*np.pi)) * val)
        return np.sqrt(np.pi) * temp

    def func_z_i2(self, z, omega, n, m):
        mp.dps = 50
        if self.a == 0:
            return (np.exp(-((omega-z))**2) + np.exp(-((omega+z))**2)) * np.sin(z*(self.la*n-self.lb*m))
        else:
            return np.exp((-((omega-z))**2)) * np.sin(2*z*(np.arcsinh(a*(self.la*n-self.lb*m)/2))/self.a) / np.tanh(np.pi*z/self.a)

    def i2(self, omega, n, m):
        if self.a == 0:
            val = trapezoidal_integrator(
                self.func_z_i2, a=0, b=200, num_points=20000, args=(omega, n, m))
        else:
            val = trapezoidal_integrator(
                self.func_z_i2, a=-100, b=100, num_points=20000, args=(omega, n, m))

        val = val / np.sqrt(np.pi)
        return val

    def i1(self, omega, n, m):
        mp.dps = 50
        if self.a == 0:
            return - mp.exp(-((self.la*n-self.lb*m)**2)/(4)) * mp.sin(omega*(self.la*n-self.lb*m))
        else:
            return - mp.exp(-((mp.asinh(self.a*(self.la*n-self.lb*m)/2))**2)/(self.a*self.a)) * mp.sin(2*omega*(mp.asinh(self.a*(self.la*n-self.lb*m)/2))/self.a)

    def lab_values(self):
        lab_values = []
        for omega in self.omega_values:
            p_m = 0
            i_1 = 0
            i_2 = 0
            print(r"$a = $", self.a)
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
        self.l = l
        self.omega_values = omega_values
        self.kmin = kmin
        self.kmax = kmax

    def func_s_pm(self, s, omega, rand):
        mp.dps = 50
        if self.a == 0:
            return (mp.exp(-s*s/(4)) * mp.cos(omega*s) - 1) * 4/(s*s)
        else:
            return (mp.exp(-s*s/(4)) * mp.cos(omega*s) * self.a*self.a/((mp.sinh(self.a*s/2))**2) - 4/(s*s))

    def pm(self, omega):
        rand = 1
        val = quad(self.func_s_pm, args=(omega, rand),
                   a=0.000001, b=np.inf, limit=100000, epsrel=1e-6, epsabs=0)[0]

        # val = trapezoidal_integrator(
        #     self.func_s_pm, a=0.0001, b=100000, num_points=4000000, args=(omega, rand))

        temp = - omega/(4*np.pi) - ((1/(8*np.pi*np.pi)) * val)
        # print(np.sqrt(np.pi)*omega/(4*np.pi))
        return np.sqrt(np.pi) * temp

    def func_z_i2(self, z, omega, k):
        mp.dps = 50
        if self.a == 0:
            return (np.exp(-((omega-z))**2) + np.exp(-((omega+z))**2)) * np.sin(z*self.l*k)
        else:
            return np.exp(-((omega-z))**2) * np.sin(2*z*(np.arcsinh(a*(self.l*k)/2))/self.a) / np.tanh(np.pi*z/self.a)

    def i2(self, omega, k):
        if self.a == 0:
            val = trapezoidal_integrator(
                self.func_z_i2, a=0, b=200, num_points=20000, args=(omega, k))
        else:
            val = trapezoidal_integrator(
                self.func_z_i2, a=-100, b=100, num_points=20000, args=(omega, k))

        val = val / np.sqrt(np.pi)
        return val

    def i1(self, omega, k):
        mp.dps = 50
        if self.a == 0:
            return - mp.exp(-((self.l*k)**2)/(4)) * mp.sin(omega*(self.l*k))
        else:
            return - mp.exp(-((mp.asinh(self.a*(self.l*k)/2))**2)/(self.a*self.a)) * mp.sin(2*omega*(mp.asinh(self.a*(self.l*k)/2))/self.a)

    def pd_values(self):
        pd_values = []
        # pd_values = np.float128(pd_values)
        # pd_values = pd_values.tolist()
        for omega in self.omega_values:
            i_1 = 0
            i_2 = 0
            print(r"$a = $", self.a)
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
    nmax = 10
    nmin = -nmax
    mmax, mmin = nmax, nmin
    step = 1
    a_vals = np.arange(0, 100+step, step=step)
    a_vals = np.around(a_vals, decimals=1)
    l_a = 0.75
    l_b = 0.25
    omega = [50, -50]

    start_time = time.time()

    for i in range(len(omega)):
        print(r"$\Omega$"+f"value is {omega[i]}")
        pa = []
        pb = []
        lab = []
        pe_plus = []
        pe_minus = []
        for a in a_vals:
            # print(f"The value of a now is: {a}")

            sum = 0
            for n in range(nmin, nmax+1):
                sum += (-1)**(2*n)

            cross_term = CrossTerms(
                a, l_a, l_b, [omega[i]], nmin, nmax, mmin, mmax)
            la_term = SingleTerms(a, l_a, [omega[i]], nmin, nmax)
            lb_term = SingleTerms(a, l_b, [omega[i]], nmin, nmax)

            lab_val = np.array(cross_term.lab_values()) / sum
            lab.append(lab_val[0])

            pa_val = np.array(la_term.pd_values())
            pa.append(pa_val[0])

            pb_val = np.array(lb_term.pd_values())
            pb.append(pb_val[0])

            # print(lab_values.shape, np.size(pa_values), np.size(pb_values))
            pe_plus_val = (pa_val + pb_val + 2*lab_val) / 4
            pe_plus.append(pe_plus_val[0])

            pe_minus_val = (pa_val + pb_val - 2*lab_val) / 4
            pe_minus.append(pe_minus_val[0])

        end_time = time.time()
        print("This process took :: ", end_time-start_time, " seconds")
        logging.info(f"This process took :: {end_time-start_time} seconds")

        # scale = 'log'
        scale = 'none'

        plt.plot(a_vals, lab, '-g', label=r"$L_{AB}$")
        plt.plot(a_vals, pa, '-y', label=r"$P_A$")
        plt.plot(a_vals, pb, '-r', label=r"$P_B$")
        plt.plot(a_vals, pe_plus, '-b', label=r"$P_E ^{+}$")
        plt.plot(a_vals, pe_minus, '-k', label=r"$P_E ^{-}$")
        plt.grid()
        plt.title(r"$\Omega\sigma = " +
                  str(omega[i]) + r", \frac{l_a}{\sigma} = " + str(l_a) + r", \frac{l_b}{\sigma} = " + str(l_b) + r"$")
        plt.xlabel(r"Acceleration, $a$")
        plt.ylabel(r"Probability, $P_E$")
        if scale == 'log':
            plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f"new_fig_nmax_{nmax}_omega_{omega[i]}_step_{step}_amax_{a_vals[-1]}_scale_{scale}.png", format='PNG')

        plt.close()
        # print(pe_plus)
        df = pd.DataFrame([a_vals, pa, pb, lab,
                           pe_plus, pe_minus])
        df.to_excel(
            f"new_fig_data_for_n_{nmax}_omega_{omega[i]}_step_{step}_amax_{a_vals[-1]}.xlsx")
