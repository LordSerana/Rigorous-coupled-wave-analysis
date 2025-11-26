def Error(array1,array2):
    abs_error=abs(array2-array1)
    rela_error=abs(abs_error/array1)
    return max(abs_error),max(rela_error)