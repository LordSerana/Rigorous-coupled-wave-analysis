import numpy as np
from scipy.integrate import quad

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
        def a_fun(x):
            x=np.mod(x,self.T)
            x0=(1-self.fill_factor)*self.T/2
            x1=(1+self.fill_factor)*self.T/2
            xc=(x0+x1)/2
            w=(x1-x0)/2
            h=np.zeros_like(x)
            mask=(x>=x0)&(x<=x1)
            h[mask]=self.depth*(1-np.abs(x[mask]-xc)/w)
            return h
        # a_fun=lambda x:(x*np.tan(self.base_angle))*(x<=self.T/2)+(self.depth-(x-self.T/2)*np.tan(self.base_angle))*(x>self.T/2)
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
        '''
        T:光栅周期;
        fill_factor:占空比;
        depth:深度
        '''
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
            return self.depth/2*(1+np.sin(2*np.pi/self.T*x))
        return a_fun
    
    def a_diff(self):
        def temp(x):
            x=np.mod(x,self.T)
            return self.depth*np.pi/self.T*np.cos(2*np.pi/self.T*x)
        return temp
    
    def Volume(self,z_min,z_max):
        '''
        计算在深度区间[z_min,z_max]内的材料体积
        '''
        def width_at_z(z):
            '''
            在深度z处的材料宽度
            '''
            s=2*z/self.depth-1
            return self.T*(0.5-np.arcsin(s)/np.pi)
        V,_=quad(width_at_z,z_min,z_max)
        return V

class Blazed():
    def __init__(self,T,angle,fill_factor,n):
        '''
        n:闪耀光栅的反射面在周期内的比例
        '''
        self.name="Blazed"
        self.T=T
        self.angle=np.radians(angle)
        self.fill_factor=fill_factor
        self.n=n
        self.depth=self.T*self.fill_factor*self.n*np.tan(self.angle)
    
    def profile(self):
        def a_fun(x):
            x=np.mod(x,self.T)
            x1=self.T*self.fill_factor*self.n
            x2=self.T*self.fill_factor
            temp=np.tan(self.angle)
            return temp*x*(x<x1)+((x2-x)*self.depth/(x2-x1)*((x>=x1)&(x<x2)))
        return a_fun