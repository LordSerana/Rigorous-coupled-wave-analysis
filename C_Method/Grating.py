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
        self.amplitude=T*fill_factor/2*np.tan(self.base_angle)
    
    def profile(self):
        '''
        m:快速傅里叶变换阶数
        '''
        a_fun=lambda x:(x*np.tan(self.base_angle))*(x<=self.T/2)+(self.amplitude-(x-self.T/2)*np.tan(self.base_angle))*(x>self.T/2)
        a_diff_fun=lambda x:(np.tan(self.base_angle))*(x<=self.T/2)+(-np.tan(self.base_angle))*(x>self.T/2)
        # plt.plot(x,y)
        # plt.xlabel("x")
        # plt.ylabel("Grating profile")
        # plt.show()
        return a_fun,a_diff_fun

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
