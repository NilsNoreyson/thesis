
from functools import lru_cache

import numpy as np
import pandas as pd

import scipy
from scipy import constants as sciConst
from scipy.integrate import solve_ivp
from scipy.integrate import quad
from scipy.interpolate import interp1d



class Constant:
    def __init__(self):
        self.K0 = sciConst.convert_temperature(0,'C','K')
        self.kB=sciConst.k
        self.EPSILON_0 = sciConst.epsilon_0
        self.E_CHARGE = sciConst.elementary_charge
        self.h = sciConst.h
        self.MASS_E = sciConst.electron_mass
        self.NA= sciConst.N_A
        self.VOL_mol = 22.4
        self.mole_per_l = self.NA/self.VOL_mol
        
    def K_to_C(self, K):
        return sciConst.convert_temperature(K, 'K', 'C')

    def C_to_K(self, C):
        return sciConst.convert_temperature(C, 'C', 'K')
    
    def eV_to_J(self,eV):
        return eV*self.E_CHARGE
    
    def J_to_eV(self,J):
        return J/self.E_CHARGE
    
    



def calc_kT(T_C):
    """
    Calculate the kT value for a temp. in °C
    T_C = Temp in °C
    """
    
    kT = CONST.kB*(CONST.C_to_K(T_C))
    return kT

def calc_eff_density_of_states(T_C,mass_e_eff_factor):
    """
    Calculate the eff. densitiy of states in the conduction band
    T_C = Temp in °C
    mass_e_eff_factor = material specific factor to calculate the effective mass
                        from the electron mass
    """
    
    kT = calc_kT(T_C)
    MASS_E_EFF = mass_e_eff_factor*CONST.MASS_E
    NC = 2*(2*np.pi*MASS_E_EFF*kT/(CONST.h**2))**(3.0/2.0)
    return NC

def calc_EDCF_by_temp(T_C, ND,mass_e_eff_factor):
    """
    T_C = Temperature in °C
    
    ND = number of donors per m³
    ND = 9e21 # 9*10**15 cm**3 Mich Thesis Seite 50
    
    mass_e_eff_factor = material specific factor to calculate the effective mass
                        from the electron mass
    """
    
        
    kT = calc_kT(T_C)
    
    NC = calc_eff_density_of_states(T_C,mass_e_eff_factor)
    
    ED1C_eV = 0.034
    ED2C_eV = 0.140
    
    a = np.exp(CONST.eV_to_J(ED1C_eV)/kT)
    b = np.exp(CONST.eV_to_J(ED2C_eV)/kT)
    t3 = 1.0
    t2 = (1.0/b-0.5*NC/ND)
    t1 = -1.0/b*NC/ND
    c = -1.0/(2*a*b)*NC/ND

    poly_params = (c,t1, t2, t3)


    solutions=np.roots(poly_params)
    EDCFs = []
    for sol in solutions:
        if sol.imag == 0:
            EDCF = np.log(sol.real)
            EDCFs.append(-EDCF*kT/CONST.E_CHARGE)
    if len(EDCFs)>1:
        raise Exception('Should not be...')
    else:
        return EDCFs[0]









class Material:
    def __init__(self,T_C,ND,
                  mass_e_eff_factor = 0.3, EPSILON = 9.86, DIFF_EF_EC_evolt = None):
        '''
        T_C = Temperature of the material
        ND = number of donors per m³
        DIFF_EF_EC_evolt = E_condution - E_Fermi 
        '''
        self.EPSILON = EPSILON
        self.ND = ND
        self.MASS_E_EFF = mass_e_eff_factor*CONST.MASS_E
        self.T_C = T_C
        self.kT = calc_kT(self.T_C)
        self.NC = calc_eff_density_of_states(T_C,mass_e_eff_factor)
        

        if DIFF_EF_EC_evolt:
            self.Diff_EF_EC_evolt = DIFF_EF_EC_evolt
        else:
            self.Diff_EF_EC_evolt = calc_EDCF_by_temp(T_C, ND, mass_e_eff_factor)
        self.Diff_EF_EC = CONST.eV_to_J(self.Diff_EF_EC_evolt)

        self.nb, self.nb_err = self.n(0)
        self.LD = np.sqrt((self.EPSILON*CONST.EPSILON_0*self.kT)
                          /(self.nb*(CONST.E_CHARGE**2)))
    
    def J_to_kT(self,J):
        return J/self.kT
    
    def kT_to_J(self,E_kT):
        return E_kT*self.kT
    
    def densitiy_of_states(self,E, E_c):
        return 4*np.pi*(2*self.MASS_E_EFF)**(3.0/2.0)/CONST.h**3*(E-E_c)**0.5
    
    def fermic_dirac(self,E_c):
        '''
        Calculate the value for the Fermi-Dirac distribution for an energetic
        position relative to the material specific conduction band E_c
        E = E_c+Diff_EF_EC+E_Fermi
        So the term in the Fermi-Dirac distribution E-E_Fermi will become
        E_c+Diff_EF_EC+E_Fermi-E_Fermi = E_c+Diff_EF_EC
        TODO: THIS SHOULD BE IN THE TEXT ABOVE SOMEWHERE
        '''
        if (E_c+self.Diff_EF_EC)/self.kT>100:
            f = 0
        else:
            f=1.0/(1+np.exp((E_c+self.Diff_EF_EC)/self.kT))
        
        return f

    def n_E(self,E,E_c):
        if E<E_c:
            n = 0
        else:
            n = self.densitiy_of_states(E, E_c)*self.fermic_dirac(E)
        return n
                                   
    @lru_cache(maxsize=512*512*512)
    def n(self, E_c):
        '''
        Calculate the number of charges in the conduction band at the position E_C 
        E_C  = the postition of the conduction band in J
        '''
        n, n_err = quad(lambda E:self.n_E(E, E_c),E_c,E_c+self.kT*100)
        return n, n_err

    
def boltzmann_acc(material, E_c):
    return np.exp(-(E_c+material.Diff_EF_EC)/(material.kT*2))

def boltzmann(material,E_c):
    return np.exp(-(E_c+material.Diff_EF_EC)/material.kT)

def densitiy_of_states(material,E, E_c):

    return 4*np.pi*(2*material.MASS_E_EFF)**(3.0/2.0)/CONST.h**3*(E-E_c)**0.5

def n_boltzmann(material,E_c):

    return boltzmann(material,E_c)*material.NC

def n_boltzmann_acc(material,E_c):

    return boltzmann_acc(material,E_c)*material.NC
    

from scipy.integrate import solve_ivp
class Grain:
    def __init__(self,grainsize_radius,material,rPoints=1000):
        self.R = grainsize_radius
        self.material = material
        self.rs = np.linspace(self.R/1000, self.R, rPoints)
    
  
    def solve_with_values(self,E_init, E_dot_init):
        r_LD = self.rs/self.material.LD
        E_init_kT = self.material.J_to_kT(E_init)
        E_dot_init_kt = self.material.J_to_kT(E_dot_init)

        #the solver should stop, when the slope is zero.
        #This is reasonable since if the slope is zero, this should be the lowest
        #point of the graph so, when we "hit_ground" the solver should stop,
        #to save some computational time
        def hit_ground(t, y):
            #print(y)
            if y[0]:
                if E_init_kT<0:
                    if y[0]>0:
                        return 0
                    if y[0]<E_init_kT:
                        return 0
                else:
                    if y[0]<0:
                        return 0
                    if y[0]>E_init_kT:
                        return 0

            if y[1]:
                if abs(y[1])<0.0001:
                    return 0
            return y[1]
        hit_ground.terminal = True
        

        #see the docstring why I chose the metohd BDF
        data = solve_ivp(self.deriv_E_E_dot,(r_LD[-1],r_LD[0]),  [E_init_kT,E_dot_init_kt],
                         t_eval=r_LD[::-1], events=hit_ground, method = 'BDF')
        
        #since we start the iteration to solve the equation from the outside,
        #the results have to be revered
        
        r = data.t[::-1]
        v = data.y[0][::-1]
        v_dot = data.y[1][::-1]
        
        #sinde we stop the evaluation earlier, when v_dot = 0,
        #the missing elements are fileed up
        missing_elements_count = len(r_LD)-len(r)
        r = np.concatenate((r_LD[:missing_elements_count], r))
        v = np.concatenate((np.ones(missing_elements_count)*v[0],v))
        v_dot = np.concatenate((np.ones(missing_elements_count)*v_dot[0],v_dot))
        
        

        return r,v, v_dot, data


    def deriv_E_E_dot(self,r_, U_U_dot):
        U = U_U_dot[0]
        U_dot = U_U_dot[1]
        E = self.material.kT_to_J(U)
        n = self.material.n(E)
        U_dot_dot = 1-n[0]/self.material.nb -2/r_*U_dot
        return [U_dot, U_dot_dot]




def create_grain_from_data(dF):
    if type(dF)==pd.Series:
        dF = pd.DataFrame([dF])
        
    if len(dF['temp'].unique())==1:
        T_C = dF['temp'].unique()[0]
    else:
        raise Exception('Multiple paramters for one grain are invalid.')
    
    if len(dF['ND'].unique())==1:
        ND = dF['ND'].unique()[0]
    else:
        raise Exception('Multiple paramters for onehttps://duckduckgo.com/?t=ffsb&q=relative+square+error+weights&atb=v152-1&ia=web grain are invalid.')
    
    if len(dF['mass_eff'].unique())==1:
        mass_e_eff_factor = dF['mass_eff'].unique()[0]/CONST.MASS_E 
    else:
        raise Exception('Multiple paramters for one grain are invalid.')
    
    if len(dF['R'].unique())==1:
        grainsize_radius = dF['R'].unique()[0]
    else:
        raise Exception('Multiple paramters for one grain are invalid.')
        

    

    material = Material(T_C,ND)
    grain = Grain(grainsize_radius=grainsize_radius,material=material)
    
    return grain

CONST = Constant()
pd.set_option('display.notebook_repr_html', True)

def _repr_latex_(self):
    return r"""
    \begin{center}
    {%s}
    \end{center}
    """ % self.to_latex(escape=False)

pd.DataFrame._repr_latex_ = _repr_latex_  # monkey patch pandas DataFrame