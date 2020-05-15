# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:10:28 2020

@author: LARP
"""
import pandas as pd                                       
from datetime import timedelta                           
from oandapyV20 import API                                
import oandapyV20.endpoints.instruments as instruments    
from statsmodels.graphics.tsaplots import plot_acf 
from statsmodels.graphics.tsaplots import plot_pacf 
import matplotlib.pyplot as plt
import statsmodels.stats.diagnostic as stm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose

global grupos
global escenarios
global tpp
global slp
global operacion
import numpy as np
global volumen
global a

# -- --------------------------------------------------------- FUNCION: Descargar precios -- #
# -- Descargar precios historicos con OANDA

def f_precios_masivos(p0_fini, p1_ffin, p2_gran, p3_inst, p4_oatk, p5_ginc):
    """
    Parameters
    ----------
    p0_fini
    p1_ffin
    p2_gran
    p3_inst
    p4_oatk
    p5_ginc
    Returns
    -------
    dc_precios
    Debugging
    ---------
    """

    def f_datetime_range_fx(p0_start, p1_end, p2_inc, p3_delta):
        """
        Parameters
        ----------
        p0_start
        p1_end
        p2_inc
        p3_delta
        Returns
        -------
        ls_resultado
        Debugging
        ---------
        """

        ls_result = []
        nxt = p0_start

        while nxt <= p1_end:
            ls_result.append(nxt)
            if p3_delta == 'minutes':
                nxt += timedelta(minutes=p2_inc)
            elif p3_delta == 'hours':
                nxt += timedelta(hours=p2_inc)
            elif p3_delta == 'days':
                nxt += timedelta(days=p2_inc)

        return ls_result

    # inicializar api de OANDA

    api = API(access_token=p4_oatk)

    gn = {'S30': 30, 'S10': 10, 'S5': 5, 'M1': 60, 'M5': 60 * 5, 'M15': 60 * 15,
          'M30': 60 * 30, 'H1': 60 * 60, 'H4': 60 * 60 * 4, 'H8': 60 * 60 * 8,
          'D': 60 * 60 * 24, 'W': 60 * 60 * 24 * 7, 'M': 60 * 60 * 24 * 7 * 4}

    # -- para el caso donde con 1 peticion se cubran las 2 fechas
    if int((p1_ffin - p0_fini).total_seconds() / gn[p2_gran]) < 4999:

        # Fecha inicial y fecha final
        f1 = p0_fini.strftime('%Y-%m-%dT%H:%M:%S')
        f2 = p1_ffin.strftime('%Y-%m-%dT%H:%M:%S')

        # Parametros pra la peticion de precios
        params = {"granularity": p2_gran, "price": "M", "dailyAlignment": 16, "from": f1,
                  "to": f2}

        # Ejecutar la peticion de precios
        a1_req1 = instruments.InstrumentsCandles(instrument=p3_inst, params=params)
        a1_hist = api.request(a1_req1)

        # Para debuging
        # print(f1 + ' y ' + f2)
        lista = list()

        # Acomodar las llaves
        for i in range(len(a1_hist['candles']) - 1):
            lista.append({'TimeStamp': a1_hist['candles'][i]['time'],
                          'Open': a1_hist['candles'][i]['mid']['o'],
                          'High': a1_hist['candles'][i]['mid']['h'],
                          'Low': a1_hist['candles'][i]['mid']['l'],
                          'Close': a1_hist['candles'][i]['mid']['c']})

        # Acomodar en un data frame
        r_df_final = pd.DataFrame(lista)
        r_df_final = r_df_final[['TimeStamp', 'Open', 'High', 'Low', 'Close']]
        r_df_final['TimeStamp'] = pd.to_datetime(r_df_final['TimeStamp'])
        r_df_final['Open'] = pd.to_numeric(r_df_final['Open'], errors='coerce')
        r_df_final['High'] = pd.to_numeric(r_df_final['High'], errors='coerce')
        r_df_final['Low'] = pd.to_numeric(r_df_final['Low'], errors='coerce')
        r_df_final['Close'] = pd.to_numeric(r_df_final['Close'], errors='coerce')

        return r_df_final

    # -- para el caso donde se construyen fechas secuenciales
    else:

        # hacer series de fechas e iteraciones para pedir todos los precios
        fechas = f_datetime_range_fx(p0_start=p0_fini, p1_end=p1_ffin, p2_inc=p5_ginc,
                                     p3_delta='minutes')

        # Lista para ir guardando los data frames
        lista_df = list()

        for n_fecha in range(0, len(fechas) - 1):

            # Fecha inicial y fecha final
            f1 = fechas[n_fecha].strftime('%Y-%m-%dT%H:%M:%S')
            f2 = fechas[n_fecha + 1].strftime('%Y-%m-%dT%H:%M:%S')

            # Parametros pra la peticion de precios
            params = {"granularity": p2_gran, "price": "M", "dailyAlignment": 16, "from": f1,
                      "to": f2}

            # Ejecutar la peticion de precios
            a1_req1 = instruments.InstrumentsCandles(instrument=p3_inst, params=params)
            a1_hist = api.request(a1_req1)

            # Para debuging
            print(f1 + ' y ' + f2)
            lista = list()

            # Acomodar las llaves
            for i in range(len(a1_hist['candles']) - 1):
                lista.append({'TimeStamp': a1_hist['candles'][i]['time'],
                              'Open': a1_hist['candles'][i]['mid']['o'],
                              'High': a1_hist['candles'][i]['mid']['h'],
                              'Low': a1_hist['candles'][i]['mid']['l'],
                              'Close': a1_hist['candles'][i]['mid']['c']})

            # Acomodar en un data frame
            pd_hist = pd.DataFrame(lista)
            pd_hist = pd_hist[['TimeStamp', 'Open', 'High', 'Low', 'Close']]
            pd_hist['TimeStamp'] = pd.to_datetime(pd_hist['TimeStamp'])

            # Ir guardando resultados en una lista
            lista_df.append(pd_hist)

        # Concatenar todas las listas
        r_df_final = pd.concat([lista_df[i] for i in range(0, len(lista_df))])

        # resetear index en dataframe resultante porque guarda los indices del dataframe pasado
        r_df_final = r_df_final.reset_index(drop=True)
        r_df_final['Open'] = pd.to_numeric(r_df_final['Open'], errors='coerce')
        r_df_final['High'] = pd.to_numeric(r_df_final['High'], errors='coerce')
        r_df_final['Low'] = pd.to_numeric(r_df_final['Low'], errors='coerce')
        r_df_final['Close'] = pd.to_numeric(r_df_final['Close'], errors='coerce')

        return r_df_final




  
#-------------------------------------------------------------------------------------------------------------------------------------------------
def metricas():
    global escenarios
    escenarios=[]  # crear lista para guardar escenarios por publicacion de indice 
  
    df_data = pd.read_csv('../PROYECTO_LARP/Indice.csv')
    for i in range (0,len(df_data)): # crear for para evaluar el tipo de escenario 
        
        if df_data["Actual"][i]>= df_data["Consensus"][i]>= df_data["Previous"][i]:
           
            escenarios.append("A") #agregar el tipo de escenario 
        elif df_data["Actual"][i]>=  df_data["Consensus"][i]< df_data["Previous"][i]:
            escenarios.append("B")#agregar el tipo de escenario 
        elif df_data["Actual"][i]<  df_data["Consensus"][i]>= df_data["Previous"][i]:
            escenarios.append("C") #agregar el tipo de escenario 
        else:
            df_data["Actual"][i]>  df_data["Consensus"][i]< df_data["Previous"][i]
            escenarios.append("D") #agregar el tipo de escenario 
    from datetime import datetime,timedelta
    from datetime import datetime
    
    
    fechas= df_data['DateTime'] # tomar la columna de fechas
    horas=df_data['Unnamed: 1'] # tomar la columna con goras 
    global a
    a= fechas+" " + horas # concatenar fechas y horas 
    fidt=[]
    for i in range (0,len(a)): # crear for para cambiar las fechas a formato datetime
        
        datestr = a[i]
        datetime_object = datetime.strptime(datestr, '%m/%d/%Y %H:%M:%S') #cambiar a datetime
        fidt.append(datetime_object) 
    fidt=pd.DataFrame(fidt) # fecha de inicio 
    fidt["fidt"]=fidt
    fidt = fidt.drop([0], axis=1) # DataFRame con fechas de inicio
    import datetime  
    y = datetime.timedelta(minutes=30) # sumar 30 min a la fecha inicial 
    ffdt = fidt + y
    ffdt["ffdt"]=ffdt
    ffdt= ffdt.drop(["fidt"],axis=1) # DataFrame con las fechas finales 
    
        
    global grupos   
    grupos=[]
    for i in range (0,len(fidt)): ## mandar llamar la funcion para descargar precios 
        OA_Ak = 'a3938e973b5c22fd03a439dd18c6c326-1493b93c8be63f133a45e8ae7b857823'
        OA_In = "USD_MXN"  
        OA_Gn = "M1"  
        fini = pd.to_datetime(fidt["fidt"][i]).tz_localize('GMT') 
        ffin = pd.to_datetime(ffdt["ffdt"][i]).tz_localize('GMT') 
        try:
            stock = f_precios_masivos(p0_fini=fini, p1_ffin=ffin, p2_gran=OA_Gn,
                                         p3_inst=OA_In, p4_oatk=OA_Ak, p5_ginc=4900)
            data=pd.DataFrame(stock)
            grupos.append(data) # lista con todos los DataFrames de los precios 
        except:
            pass
    direccion=[]
    pips_alcistas=[]
    pips_bajistas=[]
    volatilidad=[]
    
    for i in range(0,len(grupos)): ## calcular la direccion en pips de la ventana de precio 
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
        
    Dire=[]
    for i in range (0,len(direccion)): ## calcular la direccion de la ventana del precio 
        if direccion[i]>=0:
            Dire.append(1)
        else:
            Dire.append(-1)
    
    df_escenario= pd.DataFrame(list(zip(a,escenarios,Dire,pips_alcistas,pips_bajistas,volatilidad)),
                               columns=("Timestamp","Escenarios","Direccion","Pip_alcistas","Pip_bajistas","Volatilidad"))
    return df_escenario

 #-------------------------------------------------------------------------------------------------------------------------------------------------   
def deciciones(): # funcion para crear df_deciciones
    global operacion
    operacion=[]
    sl=[]
    tp=[]
    global volumen
    volumen=[]
    global slp
    slp=[]
    global tpp
    tpp=[]
    valor_pip=.42/100000 #valor de un pip
    for i in range(0,len(grupos)): # crear for para obtener tp y sl por operacion 
        sl.append(5000) # pips stop lost
        tp.append(10)  # pips take profit
        volumen.append(48000) # volumen
        
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
    df_deciciones=pd.DataFrame(list(zip(escenarios,operacion,sl,tp,volumen)),columns=("Escenario","Operacion","Sl","Tp","Volumen"))
        
    return df_deciciones


#-------------------------------------------------------------------------------------------------------------------------------------------------
def df_backtest(): # funcion para crear df_backtest
    resultado=[]
    pips=[]
    capital=[]
    capital_acm=[]
    capital_sum=100000
    for i  in range(0,len(grupos)): #for para evaluar si el precio toco tp, sl o cerro al final de los 30 min
        take_profit=tpp[i]
        take_profit=round(take_profit,4)
        stop_lost=slp[i]
        stop_lost=round(stop_lost,4)
        close=grupos[i]["Close"].values.tolist()
        
        if operacion[i]== "Compra": # si la operacion es de compra 
            
            high= grupos[i]["High"].values.tolist()
            low= grupos[i]["Low"].values.tolist()
            min_high=round(min(high),4) # obtener valores minimos de high, low y maximos de high, low 
           
            max_high=round(max(high),4)
           
            min_low=round(min(low),4)
            
            max_low=round(max(low),4)
            
            
            if take_profit in np.arange(min_high,max_high,.0001): # buscar si el tp esta en el rango del min-max high
                pips.append((grupos[i]["Open"][0]-take_profit)*-10000)
                capital.append((grupos[i]["Open"][0]-take_profit)*volumen[i]*-1)
                capital_sum= capital_sum + ((grupos[i]["Open"][0]-take_profit)*volumen[i]*-1)
                capital_acm.append(capital_sum)
                resultado.append("ganada")
                
            elif stop_lost in np.arange(min_low,max_low,.0001): # buscar si el sl esta en el rango min-max loww
                pips.append((grupos[i]["Open"][0]-stop_lost)*10000)
                resultado.append("perdedora")
                capital.append((grupos[i]["Open"][0]-stop_lost)*volumen[i]*-1)
                capital_sum= capital_sum + ((grupos[i]["Open"][0]-stop_lost)*volumen[i]*-1)
                capital_acm.append(capital_sum)
                
            else:
                if grupos[i]["Open"][0]<close[-1]: # cerrar la pocision al final con ganancia 
                    pips.append((grupos[i]["Open"][0]-close[-1])*-10000)
                    capital.append((grupos[i]["Open"][0]-close[-1])*volumen[i]*-1)
                    resultado.append("ganadora")
                    capital_sum=capital_sum+ ((grupos[i]["Open"][0]-close[-1])*volumen[i]*-1)
                    capital_acm.append(capital_sum)
                    
                    
                
                else: # cerrar la operacion al final con pérdida 
                    pips.append((grupos[i]["Open"][0]-close[-1])*10000)
                    resultado.append("perdedora")
                    capital.append((grupos[i]["Open"][0]-close[-1])*volumen[i]*-1)
                    capital_sum= capital_sum + ((grupos[i]["Open"][0]-close[-1])*volumen[i]*-1)
                    capital_acm.append(capital_sum)
            
        else: # si la operacionn es de venta 
            
            high= grupos[i]["High"].values.tolist()
            low= grupos[i]["Low"].values.tolist()
            min_high=round(min(high),4)
          
            max_high=round(max(high),4)
           
            min_low=round(min(low),4)
            
            max_low=round(max(low),4)
            
            
            
            if take_profit in np.arange(min_low,max_low,.0001): # buscar si el tp esta en el min-max low
                
                pips.append((grupos[i]["Open"][0]-take_profit)*10000)
                resultado.append("ganadora") 
                capital.append((grupos[i]["Open"][0]-take_profit)*volumen[i])
                capital_sum = capital_sum + ((grupos[i]["Open"][0]-take_profit)*volumen[i])
                capital_acm.append(capital_sum)
            elif stop_lost in np.arange(min_high,max_high,.0001): #buscar si el sl esta en el min-max high
               
                pips.append((grupos[i]["Open"][0]-stop_lost)*-10000)
                resultado.append("perdedora")
                capital.append((grupos[i]["Open"][0]-stop_lost)*volumen[i])
                capital_sum= capital_sum + ((grupos[i]["Open"][0]-stop_lost)*volumen[i])
                capital_acm.append(capital_sum)
            else:
                if grupos[i]["Open"][0]<close[-1]:  # cerrar la operacion al final  con ganancia 
                    pips.append((grupos[i]["Open"][0]-close[-1])*-10000)
                    resultado.append("perdedora")
                    capital.append((grupos[i]["Open"][0]-close[-1])*volumen[i])
                    capital_sum= capital_sum + ((grupos[i]["Open"][0]-close[-1])*volumen[i])
                    capital_acm.append(capital_sum)
                else: # cerrar la operacion al final con pérdida 
                    pips.append((grupos[i]["Open"][0]-close[-1])*10000)
                    resultado.append("ganadora") 
                    capital.append((grupos[i]["Open"][0]-close[-1])*volumen[i])
                    capital_sum = capital_sum + ((grupos[i]["Open"][0]-close[-1])*volumen[i])
                    capital_acm.append(capital_sum)
                   
    df_backtest=pd.DataFrame(list(zip(a,escenarios,operacion,volumen,resultado,pips,capital,capital_acm)),
                             columns=("Timestamp","Escenario","Operacion","Volumen","Resultado","Pips","Capital","Capital_acm"))
    return df_backtest 
#-------------------------------------------------------------------------------------------------------------------------------------------------- 
## imprimir grafica de serie de tiempo, autocorrelacion y autocorrelacion parcial  
def autoc_autocp():
    df_data = pd.read_csv('../PROYECTO_LARP/Indice.csv')
    serie=  plt.plot(df_data["Actual"]) ## serie valor 
    autocorrelacion = plot_acf(df_data["Actual"]) ## autocorrelacion
    autocorrelacionp =  plot_pacf(df_data["Actual"]) ## autocorrelacion parcial 
    
    return serie, autocorrelacion, autocorrelacionp
    
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
   ## prueba de heterocedasticidad 
def het():
    df_data = pd.read_csv('../PROYECTO_LARP/Indice.csv')
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
    
    return df_arch

#--------------------------------------------------------------------------------------------------------------------------------------------------   
# Estacionariedad

def esta_ridad():
    df_data = pd.read_csv('../PROYECTO_LARP/Indice.csv')
    datos = pd.DataFrame(df_data["Actual"]) 
    # ADF Test
    result = adfuller(datos.values, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print('Critial Values:')
        print(f'   {key}, {value}')

#--------------------------------------------------------------------------------------------------------------------------------------------------
# Normalidad
def norm():
    df_data = pd.read_csv('../PROYECTO_LARP/Indice.csv')
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

#---------------------------------------------------------------------------------------------------------------------------------------------------

# Estacionalidad
    
def esta_lidad():    
    d=[]
    e=[]
    f=[]
    g=[]
    h=[]
    df_data = pd.read_csv('../PROYECTO_LARP/Indice.csv')
    datos = pd.DataFrame(df_data["Actual"])
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
            
    
    print("Media",j.mean())
    print("Varianza",j.var())

#----------------------------------------------------------------------------------------------------------------------------------------------------
# optimizacion 

    "Parameters"
    "----------"
    "stoploss"
   " takeprofit"
   " volume"




def optimi(stlp,tppl,volu):
    sl=[]
    tp=[]
    volumen=[]
    slp=[]
    tpp=[]
    valor_pip=.42/100000
    for i in range(0,len(grupos)):
        sl.append(stlp)
        tp.append(tppl)
        volumen.append(volu)
        
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
           
            
    grafica_ca=plt.plot(df_backtest["Capital"])
    
    return df_backtest , grafica_ca , capital_sum
    
            









































    


