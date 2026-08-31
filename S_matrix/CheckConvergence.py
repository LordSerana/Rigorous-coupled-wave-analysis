import numpy as np

def CheckConvergence(Result_last,Result_new):
    Relavant_Error=(Result_new-Result_last)/Result_last
    Max_Error=max(Relavant_Error)
    return Max_Error