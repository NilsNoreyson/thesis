# -*- coding: utf-8 -*-

from Constants import *
from Material import *
from scipy.interpolate import interp1d
import scipy.integrate as integrate
import numpy as np

class Grain:
    def __init__(self,grainsize_radius,material,rPoints=100):
        self.R = grainsize_radius
        self.material = material
        self.charge_lookup = self.create_charge_lookup()
        self.rs = self.R*(1.0-np.logspace(0,3,num=rPoints)/1e3) #calcualtion points from surface to center (not lnear spaced)
    
    
    def create_charge_lookup(self):
        u_lookup_eV=np.linspace(-10,10,5000)
        u_lookup = self.material.eV_to_J(u_lookup_eV)
        f = np.vectorize(lambda x:self.material.n(x,))
        charges_lookup = f(u_lookup)
        charge_interp_f = interp1d(u_lookup,charges_lookup, kind='linear',bounds_error=False)
        return charge_interp_f
    
    def solve_with_values(self,E_init, E_dot_init):
        #v_init and v_dot_init in units of kT
        r = self.rs/self.material.LD
        U_init = self.material.J_to_kT(E_init)
        U_dot_init = self.material.J_to_kT(E_dot_init)
        #data, info_dict= integrate.odeint(self.deriv_E_E_dot, [U_init,U_dot_init], r,full_output=1,printmessg=False)
        data= integrate.odeint(self.deriv_E_E_dot, [U_init,U_dot_init], r,full_output=0,printmessg=False)
        v=data.transpose()[0]
        v_dot=data.transpose()[1]
        return r,v, v_dot


    def deriv_E_E_dot(self,U_U_dot,r_):
                
        U = U_U_dot[0]
        U_dot = U_U_dot[1]
        E = self.material.kT_to_J(U)
        U_dot_dot = 1-self.charge_lookup(E)/self.material.nb -2/r_*U_dot
        return [U_dot, U_dot_dot]





if __name__ == '__main__':
    from pylab import *
    ND = 9e21 #michi ND=9e15 cm^3
    #ND=1# michi
    material = Material(300,ND)
    grainsize_radius = 750e-9
    grain = Grain(grainsize_radius,material)

    print(grain.material.Diff_EF_EC_evolt*1000)
    
    print(grain.material.nb/1e6,'nb [cm**3]')
    print(grain.material.ND/1e6,'ND [cm**3]')
    
    kT_eV = grain.material.kT/grain.material.CONSTANTS.E_CHARGE
    print(grain.material.NC/1e6,'NC')
    print(kT_eV,'kT [eV]')
    
    
    #totoal chareg by lookup table    
    charges = grain.charge_lookup
    print(grain.material.NC*np.exp(-grain.material.Diff_EF_EC_evolt/kT_eV), 'nb_re')
    
    close()
    fig,axe = subplots(3,3)
    E_init = grain.material.kT_to_J(0.2)
    for v_i, v_dot_init in enumerate(np.linspace(-0.0,0.1,9)):
        axe = fig.axes[v_i]
        axe.set_title(v_dot_init)
        E_dot_init = grain.material.kT_to_J(v_dot_init)
        rs,vs,v_dot = grain.solve_with_values(E_init, E_dot_init)

        axe.plot(rs,vs,'ro-')
        axe.set_ylim(-0.1,0.3)
    #v_eV = 0
    #v = grain.material.eV_to_SI(v_eV)
    #print(charges(v)/grain.material.nb)
    print(4.0/3.0*np.pi * grain.R**3.0* grain.material.nb, 'total')
    print(grain.material.LD/1e-9, 'LD[nm]')