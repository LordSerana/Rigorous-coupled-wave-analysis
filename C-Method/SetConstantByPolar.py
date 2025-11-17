import numpy as np

def setConstanByPola(n1,n2,polar,Constant):
    '''
    d:光栅周期,h:槽深,profile:光栅槽型,n1、n2:入射、出射区域折射率,
    thetai:入射角,lam:波长,polar:极化状态,numTr:截断阶数,tol:误差等级
    '''
    eps0=8.8541878171e-12
    mu0=12.5663706141e-7
    mu1=1
    mu2=1
    eps1=n1**2/mu1
    eps2=n2**2/mu2
    if polar=='TE':
        mu0,eps0=-eps0,-mu0
        mu1,eps1=eps1,-mu1
        mu2,eps2=eps2,-mu2
    Z0=np.sqrt(mu0/eps0)
    Constant['n1']=n1
    Constant['n2']=n2
    Constant['eps0']=eps0
    Constant['eps1']=eps1
    Constant['eps2']=eps2
    Constant['mu0']=mu0
    Constant['mu1']=mu1
    Constant['mu2']=mu2
    Constant['Z0']=Z0
    return Constant