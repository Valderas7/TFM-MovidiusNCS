#!/usr/bin/env python
# coding: utf-8
# Se importan librerías:
import os # Listado archivos
import cv2 # OpenCV
import numpy as np # Manipular arrays
import time # Medir el tiempo de ejecución

# Se carga el documento de clases de Imagenet:
filas = open('/home/user/T-F-M/densenet169/modelos/labels.txt').read().strip().split("\n") # Divide en filas el documento
clases = [r[r.find(" ") + 1:].split(",")[0] for r in filas] # Recoge las clases por cada fila

# Se especifica el directorio principal y se crea un array para recopilar las imágenes:
directorio = "/home/user/T-F-M/densenet169/test"
imagenes = []

# Se carga la red preentrenada:
red = cv2.dnn.readNet('/home/user/T-F-M/densenet169/modelos/densenet-169.bin', '/home/user/T-F-M/densenet169/modelos/densenet-169.xml')

# Se especifica el dispositivo objetivo (MYRIAD):
red.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# Se imprime la lista de imágenes, se leen cada una de ellas y se van añadiendo a la lista:
for dirPath, dirNames, fileNames in os.walk(directorio): # Genera los nombres de los archivos
    print('Lista de archivos:') 
    for f in fileNames:
        print(os.path.join(dirPath, f))        
        imagen = cv2.imread(os.path.join(dirPath, f)) # Se lee una imagen dentro del directorio establecido
 # Transformar imagen a formato float32 y se escalan los píxeles con valores entre (0-1)
        imagenes.append(imagen) # La imagen se añade al array           
print('[Info] Todas las imágenes han sido leídas\n')

# La red requiere dimensiones fijadas para la imagen de entrada.
# Se realiza "mean subtraction" (103.94,116.78,123.68) para normalizar las imágenes de entrada
# Después de esto el BLOB de entrada tiene forma (1, 3, dim, dim) donde la dimensión tiene que ser de 224.

# Para todas las imágenes del array:
for j in range(len(imagenes)):
    # Se especifica el BLOB como entrada de la red y se escala los valores BGR multiplicando por 0.017:
    blob = cv2.dnn.blobFromImage(imagenes[j], 1, (224, 224), (103.94,116.78,123.68))
    blob = blob * 0.017
    red.setInput(blob)
    
    # Se obtiene la clasificación de salida y se mide el tiempo de ejecución:
    inicio = time.time()
    resultado = red.forward()
    final = time.time()
    print("[Info] El tiempo de ejecución fue de {:.3} segundos".format(final - inicio))

    # Ordena los índices de probabilidades en orden descendente (Top1 predicciones):
    resultado = resultado.reshape((1, 1000))
    indice = np.argsort(resultado[0])[::-1][:1]

    # Bucle sobre las predicciones:
    for (i, indice) in enumerate(indice):
        # Dibuja la predicción principal en la imagen:
        if i == 0:
            # Muestra la clase y la probabilidad en la consola:	
            print("[Info] Clase: {}. Probabilidad: {:.4}%\n".format(clases[indice+1], 100-(resultado[0][indice])/10))

            # Muestra la imagen que se le ha realizado la inferencia:
            texto = "Clase: {}, {:.2f}%".format(clases[indice+1],100-(resultado[0][indice])/10)
            cv2.putText(imagenes[j], texto, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Imagen", imagenes[j])
            cv2.waitKey(0)
