import numpy as np
from modes import *
from scattermatrix import *

class Layer:
    def __init__(self,**kwargs):
        self.n=kwargs.get('n',1)
        self.er=kwargs.get('er',self.n**2)
        self.ur=kwargs.get('ur',1)
        self.t=kwargs.get('t',0)
        self.n=np.sqrt(self.er*self.ur)

        if not np.isscalar(self.er):
            if self.er.ndim==2 or self.er.ndim==3:
                if np.isscalar(self.ur):
                    self.ur=np.ones_like(self.er)*self.ur
            if self.er.ndim==4:
                if np.isscalar(self.ur):
                    self.ur=np.ones(self.er.shape[:2])[None,None]*np.eye(3)[:,:,None,None]*self.ur
    
    def build_scatter_side(self,modes:Modes,transmission_side=False):
        return build_scatter_side(self.er,self.ur,modes.kx,modes.ky,modes.W0,transmission_side=transmission_side)
    
    def build_scatter(self,modes:Modes):
        if np.isscalar(self.er):
            return build_scatter_from_homo(self.er, self.ur, modes.kx, modes.ky, modes.W0, modes.k0 * self.t)
        if self.er.ndim==2:
            P, Q = build_omega2_isotropic(self.er, self.ur, modes)
            return build_scatter_from_omega2(P, Q, modes.W0, modes.k0*self.t)
        if self.er.ndim==3:
            P, Q = build_omega2_diagonal(self.er, self.ur, modes)
            return build_scatter_from_omega2(P, Q, modes.W0, modes.k0*self.t)
        if self.er.ndim==4:
            omega = build_omega(self.er, self.ur, modes)
            return build_scatter_from_omega(omega, modes.W0, modes.k0*self.t)
