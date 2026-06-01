import numpy as np
import sys
sys.path.append('E:/Project/python')
from S_matrix.Layer import Layer

#=======切片算法
def Slice(layers,grating,n,Constant):
    '''
    n:切片数
    layers:传进仿真层,对中间层进行切片,并返回新的layers函数
    '''
    if grating.name!="Rectangular":
        origin_FillFactor=layers[1].fill_factor
        offset=layers[1].offset
        depth=Constant['depth']/n#切片层的平均厚度
        layer0=layers[0]
        layer_last=layers[-1]
        layer_new=[]
        layer_new.append(layer0)
        if grating.name=="Blazed":
            #=============具体来说,实现效果为将堆叠结构重整为左边对齐的闪耀光栅结构
            for i in range(n):
                fill_factor=(2*i+1)/2/n*origin_FillFactor
                offset=fill_factor/2-origin_FillFactor/2
                layer=Layer(n=Constant['n2'],t=depth,fill_factor=fill_factor,offset=offset)
                layer_new.append(layer)
            layer_new.append(layer_last)
        elif grating.name=="Triangular":
            for i in range(n):
                fill_factor=(2*i+1)/2/n*origin_FillFactor
                offset=-0.5#取0/-0.5都行,即翻转结构
                layer=Layer(n=Constant['n2'],t=depth,fill_factor=fill_factor,offset=offset)
                layer_new.append(layer)
            layer_new.append(layer_last)
    return layer_new