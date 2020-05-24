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
grafica_CPU = pd.DataFrame(data = excel, columns = ['Valderas_CPU','DenseNet121_CPU','DenseNet169_CPU','ResNet50_CPU'])
sns.boxplot(x="variable", y="value", width=0.7, palette="colorblind", data=pd.melt(grafica_CPU))
plt.xlabel('Redes')
plt.ylabel('Tiempo de respuesta (s)')
plt.show()

grafica_GPU = pd.DataFrame(data = excel, columns = ['Valderas_GPU','DenseNet121_GPU','DenseNet169_GPU','ResNet50_GPU'])
sns.boxplot(x="variable", y="value", width=0.7, palette="colorblind", data=pd.melt(grafica_GPU))
plt.xlabel('Redes')
plt.ylabel('Tiempo de respuesta (s)')
plt.show()

grafica_RPi = pd.DataFrame(data = excel, columns = ['Valderas_RPi','DenseNet121_RPi','DenseNet169_RPi','ResNet50_RPi'])
sns.boxplot(x="variable", y="value", width=0.7, palette="colorblind", data=pd.melt(grafica_RPi))
plt.xlabel('Redes')
plt.ylabel('Tiempo de respuesta (s)')
plt.show()

grafica_RPi_NCS = pd.DataFrame(data = excel, columns = ['Valderas_RPi_NCS','DenseNet121_RPi_NCS','DenseNet169_RPi_NCS','ResNet50_RPi_NCS'])
sns.boxplot(x="variable", y="value", width=0.7, palette="colorblind", data=pd.melt(grafica_RPi_NCS))
plt.xlabel('Redes')
plt.ylabel('Tiempo de respuesta (s)')
plt.show()

grafica_total = pd.DataFrame(data = excel, columns = ['Valderas_CPU','DenseNet121_CPU','DenseNet169_CPU','ResNet50_CPU',
'Valderas_GPU','DenseNet121_GPU','DenseNet169_GPU','ResNet50_GPU',
'Valderas_RPi','DenseNet121_RPi','DenseNet169_RPi','ResNet50_RPi',
'Valderas_RPi_NCS','DenseNet121_RPi_NCS','DenseNet169_RPi_NCS','ResNet50_RPi_NCS'])
sns.boxplot(x="variable", y="value", width=0.7, palette="colorblind", data=pd.melt(grafica_total))
plt.xlabel('Redes')
plt.ylabel('Tiempo de respuesta (s)')
plt.show()


