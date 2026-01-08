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
        # a_fun=lambda x:(0)*(x<((1-self.fill_factor)/2)*self.T)+(x*np.tan(self.base_angle))*(((1-self.fill_factor)/2*self.T)<x<=self.T/2)\
        # +(self.depth-(x-self.T/2)*np.tan(self.base_angle))*(self.T/2<x<((1-self.fill_factor)/2+self.T/2))\
        # +(0)*(x>((1-self.fill_factor)/2+self.T/2))
        a_fun=lambda x:(x*np.tan(self.base_angle))*(x<=self.T/2)+(self.depth-(x-self.T/2)*np.tan(self.base_angle))*(x>self.T/2)
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
        def a_fun(x):
            '''
            光栅模型居中设置
            '''
            x=np.mod(x,self.T)
            x0=(1-self.fill_factor)*self.T/2
            x1=(1+self.fill_factor)*self.T/2
            return self.depth*((x>=x0)&(x<=x1))
        return a_fun

class Sinusoidal():
    def __init__(self,T,fill_factor,depth):
        self.name="Sinusoidal"
        self.T=T
        self.fill_factor=fill_factor
        self.depth=depth
    
    def profile(self):
        def a_fun(x):
            '''
            居中设置正弦光栅
            '''
            x=np.mod(x,self.T)
            return self.depth*(1+np.sin(2*np.pi*(x-self.T/2)/self.T))/2
        return a_fun