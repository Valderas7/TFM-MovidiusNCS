# Movidius-NCS
Estudio comparativo de un clasificador de imágenes en Raspberry Pi, de forma que se compara el tiempo de la inferencia en la Raspberry Pi con y sin el Neural Compute Stick (NCS). También se estudia como la complejidad de una red neuronal repercute en el tiempo de inferencia y se analiza si los tiempos obtenidos con el NCS en la Raspberry Pi se igualan a los conseguidos por la CPU del portátil y a los de una GPU de Google Colab.  

**IMPORTANTE:** 
- Model Optimizer misma versión en PC y Raspberry Pi.


## Guías
**RED NEURONAL CREADA:** 
1. Seleccionar el dataset y entrenar la red neuronal, guardando el modelo y los pesos en un único archivo en formato '.h5'.
2. Realizar una conversión del formato '.h5' (Keras) a '.pb' (TensorFlow) con un script de Python.
3. Mediante un script leer el '.pb' creado y crear otro archivo '.pbtxt'. Se edita este archivo '.pbtxt' eliminando los nodos con los nombres "flatten/Shape", "flatten/strided_slice", "flatten/Prod" y "flatten/stack" y se sustituye la operación del nodo "flatten/Reshape" ("Flatten" en vez de "Reshape")
4. Mediante el Model Optimizer de OpenVINO (mo_tf.py) transformar este archivo '.pb' a dos archivos: un archivo '.xml' (que contiene el modelo de la red) y un archivo '.bin' (que contiene los pesos de la red neuronal). Hay que especificar la dimensión del tamaño del lote al principio. (sudo python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model /xxx/xxx/modelo.pb --input_shape [1,180,180,3]  --generate_deprecated_IR_V7). Los archivos se generarán en el directorio donde se encuentre el terminal. (**'--generate_deprecated_IR_V7' crea los archivos '.xml' y '.bin' mediante la v7 del Optimizador de Modelos. Se ejecuta este comando debido a que la v10 (2020.1) instalada no permite la inferencia en la Raspberry debido a un bug general**)
5. Ejecutar el script de Python que realiza la inferencia en el NCS utilizando los archivos '.bin' y '.xml'; o el script de Python que realiza la inferencia en la CPU utilizando los archivos '.pb' y '.pbtxt, ya sea en PC o en la Raspberry Pi.

**RED NEURONAL PRE-ENTRENADA:** 
1. Ejecutar el script 'downloader.py' de OpenVINO, el cual descarga las topologías de las red pre-entrenadas. Se descargarán así los archivos 'caffemodel' y '.prototxt' en caso del entorno Caffe; o el archivo 'pb' en caso de trabajar con el entorno Tensorflow.
2. Convertir estos archivos en dos archivos de representación intermedia ('.xml' y '.bin') mediante el Model Optimizer de OpenVINO (python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model <models_dir>/classification/xxxx/xxxx.caffemodel --data_type FP32 --output_dir <ir_dir> (hecho con Model Optimizer v7).
3. Copiar el archivo de texto '.txt' con todas las clases entrenadas en Imagenet al directorio del modelo.
4. Ejecutar el script de Python que realiza la inferencia en el NCS utilizando los archivos '.bin' y '.xml'; o el script de Python que realiza la inferencia en la CPU utilizando los archivos '.pb' y '.pbtxt (Tensorflow) o '.caffemodel' y '.prototxt' (Caffe), ya sea en PC o en la Raspberry Pi.
