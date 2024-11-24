# This code is based on Neuschafer et al. J.opt.Soc.Am.B/vol.11,No.4/April(1994)
import sys
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
import pandas as pd


# Define the parameters: a_w0, a_2w0, b, and any other constants
a_w0 = 0.00 # damping coeffiient of FW
a_2w0 = 0.48 # damping coeffiient of SH
U_w0_0 = 1.0  # initial field intensity


# propagation range
z_max = 25  # upper limit of z, in um
num_points = 1000  # number of steps in z
z_values = np.linspace(0, z_max, num_points)


# Initialize arrays to fill intensity distribution
U_w0_values = np.zeros(num_points)
U_2w0_values = np.zeros(num_points)
n = np.zeros(num_points)
n_g = np.zeros(num_points)
I_w = np.zeros(num_points)
I_2w = np.zeros(num_points)


# Define the integrands for the equations
def integrand_U_w0(z_prime, U_2w0_interp):
    return a_w0 + b * U_2w0_interp(z_prime)
def integrand_U_2w0(z_prime, U_w0_interp, z):
    return U_w0_interp(z_prime)**2 * np.exp(-0.5 * a_2w0 * (z - z_prime))




U_w0_values[0] = U_w0_0
for i, z in enumerate(z_values[1:], start=1):
    j = i + 1
    U_2w0_interp = interp1d(z_values[:j], U_2w0_values[:j], kind='linear', fill_value='extrapolate')
    (integral_value, e) = quad(integrand_U_w0, 0, z, args=(U_2w0_interp), epsabs=1e-3, epsrel=1e-3)
    U_w0_values[i] = U_w0_0 * np.exp(-0.5 * integral_value)
    U_w0_interp = interp1d(z_values[:j], U_w0_values[:j], kind='linear', fill_value='extrapolate')
    integral_value1, _ = quad(integrand_U_2w0, 0, z, args=(U_w0_interp, z), epsabs=1e-3, epsrel=1e-3)
    U_2w0_values[i] = (b / 2) * integral_value1


    # calculate SHG efficient
    n[i] =  (U_2w0_values[i]/(U_w0_values[i]))**2
    n_g[i] =  (U_2w0_values[i]/U_w0_values[i])
    I_w[i] = U_w0_values[i]**2
    I_2w[i] = U_2w0_values[i]**2
   
print(len(z_values))
print(len(U_2w0_values))
print(len(U_2w0_values))
