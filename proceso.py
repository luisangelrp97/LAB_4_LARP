# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 16:25:17 2020

@author: Usuario
"""
aba=[]
for i in range (0,len(grupos)):
    l_ct=grupos[i]["Close"].values.tolist()
    l_ot=grupos[i]["Open"].values.tolist()
    
    ct=l_ct[-1]
    ot=l_ot[0]
    aba.append(ot-ct)
    print(aba)
    