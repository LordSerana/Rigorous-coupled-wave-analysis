from Compute import Compute

R_effi=[]
Abs_error=[]
Rela_error=[]

def Error(array1,array2):
    abs_error=abs(array2-array1)
    rela_error=abs(abs_error/array1)
    return max(abs_error),max(rela_error)

def Validate_Convergence(Constant):
    a=1#TM模式的分量
    b=0#TE模式的分量
    n=0
    for m in range(39,41):
        Constant['n_Tr']=2*m+1#谐波截断阶数
        if a==1:
            Polarization='TM'
            etaR_TM,etaT_TM=Compute(Constant['n1'],Constant['n2'],Polarization,Constant)
        if b==1:
            Polarization='TE'
            etaR_TE,etaT_TE=Compute(Constant['n1'],Constant['n2'],Polarization,Constant)
        R_effi.append(etaR_TM)
        if n!=0:
            abs_error,rela_error=Error(R_effi[-2],R_effi[-1])
            Abs_error.append(abs_error)
            Rela_error.append(rela_error)
        n+=1
    print("最大绝对误差:"+str(Abs_error[-1]))
    print("最大相对误差:"+str(Rela_error[-1]))