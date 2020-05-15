# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 16:26:15 2020

@author: Usuario
"""
import funciones as fn 
import numpy as np 
from statsmodels.graphics.tsaplots import plot_acf 
from statsmodels.graphics.tsaplots import plot_pacf 
import matplotlib.pyplot as plt
import statsmodels.stats.diagnostic as stm
import pandas as pd 
global lista
global Fechas
global df_decisiones
global Decisiones

#%% caracterizacion econometrica

df_data = pd.read_csv('../PROYECTO_LARP/Indice.csv')
plt.plot(df_data["Actual"]) ## serie valor 
plot_acf(df_data["Actual"]) ## autocorrelacion
plot_pacf(df_data["Actual"]) ## autocorrelacion parcial 

#%% prueba de hetorecedasticidad

datos = pd.DataFrame(df_data["Actual"])

    
    
    
datos2 = datos
    
arc = stm.het_arch(datos2)
prov = [[0],[0]]
df_arch = pd.DataFrame(prov, columns = ['Valor'])
    
df_co = pd.DataFrame(arc)
    
info = np.array([['Prueba de Lagrange (Score Test)'],['p-value']])
df_arch['Descripción'] = info
    
reorden = ['Descripción', 'Valor']
    
    
df_arch = df_arch.reindex(columns = reorden)
df_co = np.reshape(df_co, (4,1))
df_arch.iloc[0,1] = df_co.iloc[0,0]
df_arch.iloc[1,1] = df_co.iloc[1,0]

#%%  prueba de normalidad 
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df_data["Actual"], model='additive',freq=24)
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(15,8))

ax1.set_ylabel('Original Series')
ax2.set_ylabel('Trend')
ax3.set_ylabel('Seasonal')
ax4.set_ylabel('Residual')

result.observed.plot(ax=ax1,color='indigo')
result.trend.plot(ax=ax2,color='dodgerblue')
result.seasonal.plot(ax=ax3,color ='turquoise')
result.resid.plot(ax=ax4,color='darkorange')

#%% Estacionariedad
# ADF Test
result = adfuller(datos.values, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')
#%% normalidad
d=[]
e=[]
f=[]
g=[]
h=[]
for i in range(0,len(datos)):
    if i <= 128:
        d.append(datos["Actual"][i])
    elif 128 < i <= 256:
        e.append(datos["Actual"][i])
    elif 256 < i <=384:
        f.append(datos["Actual"][i])
    elif 384 < i <=512:
        g.append(datos["Actual"][i])
    else:
        h.append(datos["Actual"][i])

j=pd.DataFrame(list(zip(d,e,f,g,h)),columns=("1","2","3","4","5"))
        

print(j.mean())
print(j.var())






#%% df_escenario
df_escenarios= fn.metricas()
df_escenarios

#%% df_deciciones

df_deciciones = fn.deciciones()
df_deciciones

#%% df_backtest

df_backtest = fn.df_backtest()
df_backtest

