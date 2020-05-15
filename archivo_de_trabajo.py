# -*- coding:g utf-8 -*-
"""
Created on Fri May  1 22:13:59 2020

@author: LARP
"""

import pandas as pd
import numpy as np
from datetime import datetime
import funciones as fn
import datos
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from numpy import inf


#%% Descargar índicoe

df_data = pd.read_csv('../PROYECTO_LARP/Indice.csv')
b=df_data["Actual"].values.tolist()




#%% Juntar fecha con horas

fechas= df_data['DateTime']
horas=df_data['Unnamed: 1']

a= fechas+" " + horas 
#%% convertir a DateTime
fidt=[]
for i in range (0,len(a)):
    
    datestr = a[i]
    datetime_object = datetime.strptime(datestr, '%m/%d/%Y %H:%M:%S')
    fidt.append(datetime_object)
fidt=pd.DataFrame(fidt)
fidt["fidt"]=fidt
fidt = fidt.drop([0], axis=1)

    


#%% Sumar 30 min a las fechas 
import datetime 
y = datetime.timedelta(minutes=30)
ffdt = fidt + y
ffdt["ffdt"]=ffdt
ffdt= ffdt.drop(["fidt"],axis=1)



#%% crear for para descargar por tiempo 
grupos=[]
for i in range (0,len(fidt)):
    OA_Ak = 'a3938e973b5c22fd03a439dd18c6c326-1493b93c8be63f133a45e8ae7b857823'
    OA_In = "USD_MXN"  
    OA_Gn = "M1"  
    fini = pd.to_datetime(fidt["fidt"][i]).tz_localize('GMT') 
    ffin = pd.to_datetime(ffdt["ffdt"][i]).tz_localize('GMT') 
    try:
        stock = fn.f_precios_masivos(p0_fini=fini, p1_ffin=ffin, p2_gran=OA_Gn,
                                     p3_inst=OA_In, p4_oatk=OA_Ak, p5_ginc=4900)
        data=pd.DataFrame(stock)
        grupos.append(data)
    except:
        pass
 #%% Clasificacion de esenarios 

escenarios=[]
for i in range (0,len(df_data)):
    
    if df_data["Actual"][i]>= df_data["Consensus"][i]>= df_data["Previous"][i]:
       
        escenarios.append("A")
    elif df_data["Actual"][i]>=  df_data["Consensus"][i]< df_data["Previous"][i]:
        escenarios.append("B")
    elif df_data["Actual"][i]<  df_data["Consensus"][i]>= df_data["Previous"][i]:
        escenarios.append("C")
    else:
        df_data["Actual"][i]>  df_data["Consensus"][i]< df_data["Previous"][i]
        escenarios.append("D")
    

        
#%% Metricas 

direccion=[]
pips_alcistas=[]
pips_bajistas=[]
volatilidad=[]
for i in range(0,len(grupos)):
    l_ct=grupos[i]["Close"].values.tolist()
    l_ot=grupos[i]["Open"].values.tolist()
    l_h=grupos[i]["High"].values.tolist()
    l_l=grupos[i]["Low"].values.tolist()
    max_h=max(l_h)
    min_l=min(l_l)
    ct=l_ct[-1]
    ot=l_ot[0]
    res=ot-ct
    if res >= 0:
        direccion.append(res*10000)
    else:
        direccion.append(res*10000)
    res2=max_h-ot
    pips_alcistas.append(res2*10000)
    res3=ot-min_l
    pips_bajistas.append(res3*10000)
    res4 = max_h-min_l
    volatilidad.append(res4*10000)

#%% Direccion 
Dire=[]
for i in range (0,len(direccion)):
    if direccion[i]>=0:
        Dire.append(1)
    else:
        Dire.append(-1)

    


#%% Df_escenario

df_escenario= pd.DataFrame(list(zip(a,escenarios,Dire,pips_alcistas,pips_bajistas,volatilidad)),columns=("Timestamp","Escenarios","Direccion","Pip_alcistas","Pip_bajistas","Volatilidad"))
             

#%% Df_decisiones

operacion=[]
sl=[]
tp=[]
volumen=[]
slp=[]
tpp=[]
valor_pip=.42/100000
for i in range(0,len(grupos)):
    sl.append(5000)
    tp.append(10)
    volumen.append(48000)
    
    if escenarios[i]== "A":
        operacion.append("Compra")
        tpp.append(grupos[i]["Open"][0]+(valor_pip*tp[i]))
        slp.append(grupos[i]["Open"][0]-(valor_pip*sl[i]))
    elif escenarios[i] == "B":
        operacion.append("Venta")
        tpp.append(grupos[i]["Open"][0]-(valor_pip*tp[i]))
        slp.append(grupos[i]["Open"][0]+(valor_pip*sl[i]))
    elif escenarios[i]== ("C"):
        operacion.append("Compra")
        tpp.append(grupos[i]["Open"][0]+(valor_pip*tp[i]))
        slp.append(grupos[i]["Open"][0]-(valor_pip*sl[i]))
    else:
        operacion.append("Venta")
        tpp.append(grupos[i]["Open"][0]-(valor_pip*tp[i]))
        slp.append(grupos[i]["Open"][0]+(valor_pip*sl[i]))
#%% df_deciciones
df_deciciones=pd.DataFrame(list(zip(escenarios,operacion,sl,tp,volumen)),columns=("Escenario","Operacion","Sl","Tp","Volumen"))


        
#%% df_backtest
la=[]
lo=[]
al=[]
co=[]
ca=[]
resultado=[]
pips=[]
capital=[]
capital_acm=[]
capital_sum=100000
for i  in range(0,len(grupos)):
    take_profit=tpp[i]
    take_profit=round(take_profit,4)
    stop_lost=slp[i]
    stop_lost=round(stop_lost,4)
    close=grupos[i]["Close"].values.tolist()
    
    if operacion[i]== "Compra":
        
        high= grupos[i]["High"].values.tolist()
        low= grupos[i]["Low"].values.tolist()
        min_high=round(min(high),4)
        la.append(min_high)
        max_high=round(max(high),4)
        lo.append(max_high)
        min_low=round(min(low),4)
        co.append(min_low)
        max_low=round(max(low),4)
        ca.append(max_low)
        
        if take_profit in np.arange(min_high,max_high,.0001):
            pips.append((grupos[i]["Open"][0]-take_profit)*-10000)
            capital.append((grupos[i]["Open"][0]-take_profit)*volumen[i]*-1)
            capital_sum= capital_sum + ((grupos[i]["Open"][0]-take_profit)*volumen[i]*-1)
            capital_acm.append(capital_sum)
            resultado.append("ganada")
            
        elif stop_lost in np.arange(min_low,max_low,.0001):
            pips.append((grupos[i]["Open"][0]-stop_lost)*10000)
            resultado.append("perdedora")
            capital.append((grupos[i]["Open"][0]-stop_lost)*volumen[i]*-1)
            capital_sum= capital_sum + ((grupos[i]["Open"][0]-stop_lost)*volumen[i]*-1)
            capital_acm.append(capital_sum)
            
        else:
            if grupos[i]["Open"][0]<close[-1]:
                pips.append((grupos[i]["Open"][0]-close[-1])*-10000)
                capital.append((grupos[i]["Open"][0]-close[-1])*volumen[i]*-1)
                resultado.append("ganadora")
                capital_sum=capital_sum+ ((grupos[i]["Open"][0]-close[-1])*volumen[i]*-1)
                capital_acm.append(capital_sum)
                
                
            
            else:
                pips.append((grupos[i]["Open"][0]-close[-1])*10000)
                resultado.append("perdedora")
                capital.append((grupos[i]["Open"][0]-close[-1])*volumen[i]*-1)
                capital_sum= capital_sum + ((grupos[i]["Open"][0]-close[-1])*volumen[i]*-1)
                capital_acm.append(capital_sum)
        
    else:
        
        high= grupos[i]["High"].values.tolist()
        low= grupos[i]["Low"].values.tolist()
        min_high=round(min(high),4)
        la.append(min_high)
        max_high=round(max(high),4)
        lo.append(max_high)
        min_low=round(min(low),4)
        co.append(min_low)
        max_low=round(max(low),4)
        ca.append(max_low)
        
        
        if take_profit in np.arange(min_low,max_low,.0001):
            al.append("tp venta")
            pips.append((grupos[i]["Open"][0]-take_profit)*10000)
            resultado.append("ganadora") 
            capital.append((grupos[i]["Open"][0]-take_profit)*volumen[i])
            capital_sum = capital_sum + ((grupos[i]["Open"][0]-take_profit)*volumen[i])
            capital_acm.append(capital_sum)
        elif stop_lost in np.arange(min_high,max_high,.0001):
           
            pips.append((grupos[i]["Open"][0]-stop_lost)*-10000)
            resultado.append("perdedora")
            capital.append((grupos[i]["Open"][0]-stop_lost)*volumen[i])
            capital_sum= capital_sum + ((grupos[i]["Open"][0]-stop_lost)*volumen[i])
            capital_acm.append(capital_sum)
        else:
            if grupos[i]["Open"][0]<close[-1]:
                pips.append((grupos[i]["Open"][0]-close[-1])*-10000)
                resultado.append("perdedora")
                capital.append((grupos[i]["Open"][0]-close[-1])*volumen[i])
                capital_sum= capital_sum + ((grupos[i]["Open"][0]-close[-1])*volumen[i])
                capital_acm.append(capital_sum)
            else:
                pips.append((grupos[i]["Open"][0]-close[-1])*10000)
                resultado.append("ganadora") 
                capital.append((grupos[i]["Open"][0]-close[-1])*volumen[i])
                capital_sum = capital_sum + ((grupos[i]["Open"][0]-close[-1])*volumen[i])
                capital_acm.append(capital_sum)
               
df_backtest=pd.DataFrame(list(zip(a,escenarios,operacion,volumen,resultado,pips,capital,capital_acm)),
                         columns=("Timestamp","Escenario","Operacion","Volumen","Resultado","Pips","Capital","Capital_acm"))

    
        
        
        
        

            
        
       
        
        



             
    
    
        
        













































        


    
       

