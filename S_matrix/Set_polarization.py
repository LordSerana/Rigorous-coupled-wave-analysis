import numpy as np

def Set_Polarization(thetai,phi,wavelength,pTM,pTE,Constant):
    Constant['thetai']=thetai
    Constant['phi']=phi
    Constant['wavelength']=wavelength
    Constant['pTM']=pTM
    Constant['pTE']=pTE
    k0=2*np.pi/wavelength
    Constant['k0']=k0
    n=[0,0,1]
    kinc=Constant['n1']*np.array([np.sin(thetai)*np.cos(phi),np.sin(thetai)*np.sin(phi),np.cos(thetai)])
    Constant['kinc']=kinc
    if thetai==0:
        aTE=[0,1,0]
    else:
        aTE=np.cross(n,kinc)/np.linalg.norm(np.cross(n,kinc))
    aTM=np.cross(kinc,aTE)/np.linalg.norm(np.cross(kinc,aTE))
    p=np.dot(pTE,aTE)+np.dot(pTM,aTM)
    p=p/np.linalg.norm(p)
    Constant['p']=p
    return Constant