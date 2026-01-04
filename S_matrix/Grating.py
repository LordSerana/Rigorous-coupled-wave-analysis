import numpy as np

class Triangular():
    def __init__(self,T,base_angle,fill_factor):
        '''
        T:周期,base_angle:三角光栅底角,fill_factor:占空比,amplitude:光栅槽深
        n1:入射区域折射率,n2:透射区域折射率
        '''
        self.name="Triangular"
        self.T=T
        self.base_angle=np.radians(base_angle)
        self.fill_factor=fill_factor
        self.depth=T*fill_factor/2*np.tan(self.base_angle)
    
    def profile(self):
        '''
        m:快速傅里叶变换阶数
        '''
        a_fun=lambda x:(0)*(x<((1-self.fill_factor)/2)*self.T)+(x*np.tan(self.base_angle))*((1-self.fill_factor)/2*self.T<x<=self.T/2)\
        +(self.depth-(x-self.T/2)*np.tan(self.base_angle))*(self.T/2<x<((1-self.fill_factor)/2+self.T/2))\
        +(0)*(x>((1-self.fill_factor)/2+self.T/2))
        # a_fun=lambda x:(x*np.tan(self.base_angle))*(x<=self.T/2)+(self.depth-(x-self.T/2)*np.tan(self.base_angle))*(x>self.T/2)
        # a_diff_fun=lambda x:(np.tan(self.base_angle))*(x<=self.T/2)+(-np.tan(self.base_angle))*(x>self.T/2)
        # plt.plot(x,y)
        # plt.xlabel("x")
        # plt.ylabel("Grating profile")
        # plt.show()
        return a_fun

class Rectangular():
    def __init__(self,T,fill_factor,depth):
        self.name="Rectangular"
        self.T=T
        self.fill_factor=fill_factor
        self.depth=depth
    
    def profile(self):
        a_fun=lambda x:(0)*(x<((1-self.fill_factor)/2)*self.T)+\
            self.depth*((1-self.fill_factor)/2*self.T<x<=(1+self.fill_factor)/2*self.T)+\
            (0)*(x>(1+self.fill_factor)/2*self.T)
        
        a_fun=lambda x:(self.depth)*(x<=self.T*self.fill_factor)+(0)*(x>self.T*self.fill_factor)
        return a_fun