import numpy as np

class Layer:
    def __init__(self,**kwargs):
        self.n=kwargs.get('n',1)
        self.er=kwargs.get('er',self.n**2)
        self.ur=kwargs.get('ur',1)
        self.t=kwargs.get('t',0)
        self.n=np.sqrt(self.er*self.ur)
        self.fill_factor=kwargs.get('fill_factor',1)