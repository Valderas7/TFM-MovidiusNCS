# Movidius-NCS
Estudio comparativo del rendimiento de una red neuronal profunda en una Raspberry Pi con Movidius Neural Compute Stick y sin él. Para ello se tienen tres máquinas virtuales en VMWare:
- Ubuntu 64-bit: Tiene OpenVINO instalado pero no se entrena la red ya que lo hace lentamente. Tiene instalado TF 1.2.
- MovidiusNCS: Realiza el entreno ya que lo hace de forma muy rápida y tiene OpenVINO. Tiene instalado TF 2.0 (desventaja para sesiones)
- NCS: Tiene OpenVINO instalado y no se entrena (lento) Tiene instalado TF 1.2.

**IMPORTANTE:** 
- Model Optimizer v10 en PC.
- Model Optimizer v7 en RPi.


## Pasos NCS:
1. Seleccionar el dataset y entrenar la red neuronal, guardando el modelo y los pesos en un único archivo en formato '.h5'.
2. Realizar una conversión del formato '.h5' (Keras) a '.pb' (TensorFlow) con un script de Python.
3. Mediante el Model Optimizer de OpenVINO (mo_tf.py) transformar este archivo '.pb' a dos archivos: un archivo '.xml' (que contiene el modelo de la red) y un archivo '.bin' (que contiene los pesos de la red neuronal). Hay que especificar la dimensión del tamaño del lote al principio. (sudo python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model /xxx/xxx/modeloPB.pb --input_shape [1,180,180,3]  --generate_deprecated_IR_V7). Los archivos se generarán en el directorio donde se encuentre el terminal. (**'--generate_deprecated_IR_V7' crear los archivos '.xml' y '.bin' mediante la v7 del Optimizador de Modelos. Se ejecuta este comando debido a que la v10 (2020.1) instalada no permite la inferencia en la Raspberry debido a un bug general**)
4. Ejecutar el script de Python que realiza la inferencia en el NCS utilizando estos archivos '.bin' y '.xml'.

## Pasos CPU:
1. Seleccionar el dataset y entrenar la red neuronal, guardando el modelo y los pesos en un único archivo en formato '.h5'.
2. Realizar una conversión del formato '.h5' (Keras) a '.pb' (TensorFlow) con un script de Python.
3. Mediante un script leer el '.pb' creado y crear otro archivo '.pbtxt'. Se edita este archivo '.pbtxt' eliminando los nodos con los nombres "flatten/Shape", "flatten/strided_slice", "flatten/Prod" y "flatten/stack" y se sustituye la operación del nodo "flatten/Reshape" ("Flatten" en vez de "Reshape")
4. Ejecutar el script de Python que realiza la inferencia en el NCS utilizando esos archivos '.pb' y '.pbtxt.
