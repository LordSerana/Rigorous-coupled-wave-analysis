import numpy as np

def Generate_height_array(grating, T, Nx):
    """
    T:光栅周期
    返回两个数组：
    x:一个周期内的离散x坐标(长度为Nx)
    s:每个x处材料-空气界面的z坐标(从光栅顶部算起，向下为正)
    """
    x=np.linspace(0, T, Nx, endpoint=False)  # 周期内均匀采样
    
    if grating.name == "Triangular":
        # 以三角槽为例，材料存在于 s(x) 以下
        # 假设三角形顶点在中心，底边占满整个周期
        # 则边界高度 s(x) = depth * 2 * |x - T/2| / T  （形成倒V形槽）
        half=T/2
        s=grating.depth*2*np.abs(x-half)/T
    elif grating.name == "Blazed":
        # 闪耀光栅：锯齿形轮廓，左低右高（或左高右低）
        s=grating.depth*(x/T)#0到depth线性变化，形成锯齿
    elif grating.name == "Sinusoidal":
        s=grating.depth*0.5*(1+np.sin(2*np.pi*x/T-np.pi/2))#调整相位使顶部/底部合理
    elif grating.name == "Rectangular":
        # 矩形光栅，直接按占空比生成0和depth的阶跃
        s = np.where(np.mod(x,T)<grating.fill_factor*T,0.0,grating.depth)
    else:
        raise ValueError("Unknown grating type")
    return x, s