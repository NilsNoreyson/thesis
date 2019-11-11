# -*- coding: utf-8 -*-
from scipy import constants as sciConst
class Constant:
    def __init__(self):
        self.K0 = 273.15
        self.kB=1.3806488*10**(-23)
        self.EPSILON_0 = 8.854187817*10**(-12)
        self.E_CHARGE = 1.6021766208*10**(-19)
        self.h = 6.626070040*10**(-34)
        self.MASS_E = 9.10938291*10**(-31)
        self.NA= sciConst.constants.N_A
        self.VOL_mol = 22.4
        self.mole_per_l = self.NA/self.VOL_mol

if __name__ == '__main__':
    c = Constant()
    print(c.NA)