import numpy as np

def SetConstant(n1,n2,polar,period,n_Tr,lam,thetai):
    '''
    d:光栅周期,h:槽深,profile:光栅槽型,n1、n2:入射、出射区域折射率,
    thetai:入射角,lam:波长,polar:极化状态,numTr:截断阶数,tol:误差等级
    '''
    eps0=8.8541878171*1e-12
    mu0=12.5663706141*1e-7
    mu1=1
    mu2=1
    eps1=n1**2/mu1
    eps2=n2**2/mu2
    if polar=='TE':
        mu0,eps0=-eps0,-mu0
        mu1,eps1=eps1,-mu1
        mu2,eps2=eps2,-mu2
    Z0=np.sqrt(mu0/eps0)
    Constant={}
    Constant['period']=period
    Constant['n1']=n1
    Constant['n2']=n2
    Constant['eps0']=eps0
    Constant['eps1']=eps1
    Constant['eps2']=eps2
    Constant['mu0']=mu0
    Constant['mu1']=mu1
    Constant['mu2']=mu2
    Constant['Z0']=Z0
    Constant['k0']=2*np.pi/lam
    Constant['K']=2*np.pi/period
    Constant['n_Tr']=n_Tr
    m1=int(-((n_Tr-1)/2))
    m2=int(-m1)
    m_set=np.linspace(m1,m2,m2-m1+1,dtype=int)
    n_set=np.linspace(1-n_Tr,n_Tr-1,2*n_Tr-1,dtype=int)
    Constant['m_set']=m_set
    Constant['n_set']=n_set
    alpha_m=n1*Constant['k0']*np.sin(thetai)+Constant['K']*m_set
    beta1_m=np.sqrt(np.diag(np.eye(n_Tr)*(n1*Constant['k0'])**2-np.diag(alpha_m**2)),dtype=np.complex128)
    beta2_m=np.sqrt(np.diag(np.eye(n_Tr)*(n2*Constant['k0'])**2-np.diag(alpha_m**2)),dtype=np.complex128)
    Constant['alpha_m']=alpha_m
    Constant['beta1_m']=beta1_m
    Constant['beta2_m']=beta2_m

    idx=np.where(np.abs(np.imag(beta1_m))<1e-10)
    n1_set=m_set[idx]
    n1_set_ind=idx
    Constant['n1_set']=n1_set
    Constant['n1_set_ind']=n1_set_ind[0]

    idx=np.where(np.abs(np.imag(beta2_m))<1e-10)
    n2_set=m_set[idx]
    n2_set_ind=idx
    Constant['n2_set']=n2_set
    Constant['n2_set_ind']=n2_set_ind[0]
    return Constant