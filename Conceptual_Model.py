
"""
Conceptual Model

@author: Anthanasios Paschalis
"""

import numpy as np
from numba import jit

@jit(nopython=True)
def Model(Param,Ini,R,PET):
    
    (Zr,n1,Ks,Ksb,Zg,n2,b,c,d,T1,T2,DT) = Param
    (S1_i,S2_i,V1_i,V2_i) = Ini
    
    n = len(R)
    
    S1_a = np.zeros((n,))
    S2_a = np.zeros((n,))
    V1_a = np.zeros((n,))
    V2_a = np.zeros((n,))
    Q1_a = np.zeros((n,))
    Q2_a = np.zeros((n,))
    
    S1_a[0] = S1_i
    S2_a[0] = S2_i
    V1_a[0] = V1_i
    V2_a[0] = V2_i
    
    S1 = S1_a[0]
    S2 = S2_a[0]
    V1 = V1_a[0]
    V2 = V2_a[0]

    for i in range(1,n):
        
        p_c   = R[i-1] #current step rainfall
        pet_c = PET[i-1] #current step PET

        AET = S1*pet_c
        
        if S1 == 1:
            I = 0
        else:
            I = min(p_c,Ks)
        
        u1 = p_c-I
        
        if (S2 == 1) or (S1 <= 0):
            Lk = 0
        else:
            Lk = Ksb*(S1**b)
        
        if (S1*(n1*Zr)) < ((Lk+AET)*DT):
            Lk = (S1*n1*Zr-AET)/DT
            
        u2 = c*(S2**d)
        
        if (S2*(n2*Zg)) < ((Lk-u2)*(DT)):
            u2 = S2*n2*Zg/DT

        
        S1 = S1 + DT*(I-AET-Lk)/(n1*Zr) # [-/h]
        S2 = S2 + DT*(Lk-u2)/(n2*Zg)    # [-/h]
        V1 = V1 + DT*(u1-(V1/T1))         # [mm/h]
        V2 = V2 + DT*(u2-(V2/T2))         # [mm/h]
        #print(S1)
        
        # Remove excess water if saturated
        if S1>1:
            V1 = V1+(S1-1)*(n1*Zr)# [mm]
            S1 = 1;
            
        if S2>1:
            V2 = V2+(S2-1)*(n2*Zg)# [mm]
            S2 = 1;
        
        Q1 = (DT)*V1/T1/DT
        Q2 = (DT)*V2/T2/DT
        
        S1_a[i] = S1
        S2_a[i] = S2
        V1_a[i] = V1
        V2_a[i] = V2
        Q1_a[i] = Q1
        Q2_a[i] = Q2
    
    Qt = Q1_a + Q2_a
    
    #print(p_c)
    
    return Qt,Q1_a,Q2_a,S1_a,S2_a,V1_a,V2_a
    