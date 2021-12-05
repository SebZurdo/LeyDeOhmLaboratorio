# Importamos los módulos necesarios, pandas y matplotlib para las gráficas, numpy para arreglos y sklearn para la herramienta de regresión linear
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression

#Arreglos con los datos obtenidos en la práctica de laboratorio

datosVoltaje = [3.55, 2.38, 2.16, 1.66, 1.35]
datosResistencia = [2.36, 1.62, 1.49, 1.19, 0.85]
datosCorriente = [1.5, 1.47, 1.45, 1.39, 1.58]

#Creación de los DataFrame de pandas para poder realizar las gráficas

VyR=pd.DataFrame({'Voltaje': datosVoltaje,'Resistencia': datosResistencia})
VyC=pd.DataFrame({'Voltaje': datosVoltaje,'Corriente': datosCorriente})

#Conversión a arreglos de numpy para realizar la regresión linear

Voltaje = VyR['Voltaje'].values.reshape(-1,1)
Resistencia = VyR['Resistencia'].values.reshape(-1,1)

#Creación de un objeto LinearRegression para poder hacer la regresión linear

regresor = LinearRegression()
regresor.fit(Voltaje, Resistencia)

#Arreglo con los datos de la línea de tendencia calculada a partir de la regresión linear

regresionResistencia = regresor.predict(Voltaje)

#Elaboración de la primera gráfica con el DataFrame VyR y el arreglo con los datos de la línea de tendencia (Voltaje - Resistencia)

graph = plt.figure(figsize=(14,14))
plt.scatter(VyR['Voltaje'], VyR['Resistencia'])
plt.plot(VyR['Voltaje'], VyR['Resistencia'])
plt.plot(Voltaje , regresionResistencia, color = 'red')
plt.xlabel('Voltaje')
plt.ylabel('Resistencia')
plt.grid()
plt.show()

#Elaboración de la segunda gráfica con el DataFrame VyC (Voltaje - Corriente)

graph = plt.figure(figsize=(14,14))
plt.scatter(VyC['Voltaje'], VyC['Corriente'], color = 'sandybrown')
plt.plot(VyC['Voltaje'], VyC['Corriente'], color = 'sandybrown')
plt.xlabel('Voltaje')
plt.ylabel('Corriente')
plt.grid()
plt.show()





