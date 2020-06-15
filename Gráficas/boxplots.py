#!/usr/bin/env python
# coding: utf-8
# Se importan librerías:
import numpy as np # Vectores y matrices
import pandas as pd # Manipulación de datos
import matplotlib.pyplot as plt
import seaborn as sns # Gráficas estadísticas

# Abrir el excel de resultados:
excel = pd.read_excel('/home/user/T-F-M/boxplots/TFM_grafica.xlsx')
excel.head()

# Se crean los boxplots:
grafica_Val = pd.DataFrame(data = excel, columns = ['Valderas_CPU','Valderas_GPU','Valderas_RPi','Valderas_RPi_NCS'])
sns.boxplot(x="variable", y="value", width=0.7, palette="colorblind", data=pd.melt(grafica_Val))
plt.title('Clasificador de perros vs gatos (1.3M parámetros)')
plt.xlabel('Plataforma de ejecución')
plt.ylabel('Tiempo de respuesta (s)')
plt.show()

grafica_dn121 = pd.DataFrame(data = excel, columns = ['DenseNet121_CPU','DenseNet121_GPU','DenseNet121_RPi','DenseNet121_RPi_NCS'])
sns.boxplot(x="variable", y="value", width=0.7, palette="colorblind", data=pd.melt(grafica_dn121))
plt.title('DenseNet121 (8M parámetros)')
plt.xlabel('Plataforma de ejecución')
plt.ylabel('Tiempo de respuesta (s)')
plt.show()

grafica_dn169 = pd.DataFrame(data = excel, columns = ['DenseNet169_CPU','DenseNet169_GPU','DenseNet169_RPi','DenseNet169_RPi_NCS'])
sns.boxplot(x="variable", y="value", width=0.7, palette="colorblind", data=pd.melt(grafica_dn169))
plt.title('DenseNet169 (14.3 M parámetros)')
plt.xlabel('Plataforma de ejecución')
plt.ylabel('Tiempo de respuesta (s)')
plt.show()

grafica_rn50 = pd.DataFrame(data = excel, columns = ['ResNet50_CPU','ResNet50_GPU','ResNet50_RPi','ResNet50_RPi_NCS'])
sns.boxplot(x="variable", y="value", width=0.7, palette="colorblind", data=pd.melt(grafica_rn50))
plt.title('ResNet50 (25.6M parámetros)')
plt.xlabel('Plataforma de ejecución')
plt.ylabel('Tiempo de respuesta (s)')
plt.show()

