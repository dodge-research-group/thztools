import numpy as np

# SI Fundamental constants in SI units

# Fundamental constants (2019 SI)
nuCs = 9192631770                           # Caesium 133 freqency
c = 299792458                               # speed of light [m/s]
h = 6.62607015e-34                          # Planck's constant [J s]
qe = 1.602176634e-19                        # electron charge [C]
kB = 1.380649e-23                           # Boltzmann's constant [J/K]
NA = 6.02214076e23                          # Avagadro's constant [mol^-1]
Kcd = 683                                   # luminous efficacy of monochromatic radiation of frequency 540e12 [lm/W]

# electromagnetic constants
alpha = 1/137.035999139                     # fine structure constant
mu0 = 2*h*alpha/(c*qe**2)                   # vacuum permeability
eps0 = 1/(mu0*c**2)                         # vacuum permittivity
z0 = np.sqrt(mu0/eps0)                      # vacuum impedance [Ohm]

# Electron properties
me = 9.10938291e-31                         # electron mass [kg]

# Proton properties
mp = 1.672621898e-27                        # proton mass [kg]

# Neutron properties
mn = 1.674927471e-27                        # neutron mass [kg]

# Quantum constants
hbar = h/(2*np.pi)                          # hbar [J s]
Ry = me*qe**4/(8*eps0**(2*h**2))            # Rydberg [J]
a0 = 4*np.pi*eps0*hbar**2/(me*qe**2)        # Bohr [m]
