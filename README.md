# Movidius-NCS
Estudio comparativo del rendimiento de una red neuronal profunda en una Raspberry Pi con Movidius Neural Compute Stick y sin él. 

**IMPORTANTE:** 
- Model Optimizer v10 en PC.
- Model Optimizer v7 en RPi. (2019.3.334)


## Tutorial:
1. Seleccionar el dataset y entrenar la red neuronal, guardando el modelo y los pesos en un único archivo en formato '.h5'.
2. Realizar una conversión del formato '.h5' (Keras) a '.pb' (TensorFlow) con un script de Python.
3. Mediante un script leer el '.pb' creado y crear otro archivo '.pbtxt'. Se edita este archivo '.pbtxt' eliminando los nodos con los nombres "flatten/Shape", "flatten/strided_slice", "flatten/Prod" y "flatten/stack" y se sustituye la operación del nodo "flatten/Reshape" ("Flatten" en vez de "Reshape")
4. Mediante el Model Optimizer de OpenVINO (mo_tf.py) transformar este archivo '.pb' a dos archivos: un archivo '.xml' (que contiene el modelo de la red) y un archivo '.bin' (que contiene los pesos de la red neuronal). Hay que especificar la dimensión del tamaño del lote al principio. (sudo python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model /xxx/xxx/modelo.pb --input_shape [1,180,180,3]  --generate_deprecated_IR_V7). Los archivos se generarán en el directorio donde se encuentre el terminal. (**'--generate_deprecated_IR_V7' crea los archivos '.xml' y '.bin' mediante la v7 del Optimizador de Modelos. Se ejecuta este comando debido a que la v10 (2020.1) instalada no permite la inferencia en la Raspberry debido a un bug general**)
5. Ejecutar el script de Python que realiza la inferencia en el NCS utilizando los archivos '.bin' y '.xml'; o el script de Python que realiza la inferencia en la CPU utilizando los archivos '.pb' y '.pbtxt, ya sea en PC o en la Raspberry Pi.

