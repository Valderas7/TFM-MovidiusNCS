# Se importan los paquetes necesarios:
from keras import backend as K
from keras.models import load_model

# Se pone la fase de aprendizaje a cero para quitar el modo de entrenamiento y se carga el modelo completo:
K.set_learning_phase(0)
modelo = load_model('/home/user/T-F-M/Valderas/modelos/modelo.h5', compile = False)

# Se importan los paquetes necesarios:
from keras import backend as K
import tensorflow as tf

# Se define una función para conseguir un grafo computacional:
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """ Congela el estado de una sesión y crea un modelo computacional.
    Se crea un nuevo modelo donde los nodos variables se reemplazan por
    constantes con su valor actual en la sesión. Este nuevo modelo será reducido, 
    de forma que la información no necesaria para la salida del modelo será eliminada.

    @param session: La sesión de TensorFlow que será exportada.
    @param keep_var_names: Una lista de nombres de variables que no serán exportadas;
                          o ninguno, para exportar todas las variables en el grafo.
    @param output_names: Nombres de las salidas del modelo.
    @param clear_devices: Elimina las directrices del dispositivo del modelo para una mejor portabilidad.
    @return: La definición del grafo exportado."""

    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    grafo = session.graph
    with grafo.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = grafo.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        grafo_exportado = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return grafo_exportado

# Se congela la sesión y se elige la salida del modelo creado, eliminando información innecesaria:
grafo_exportado = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in modelo.outputs])

# Se escribe el modelo de salida '.pb', guardándolo en el sistema de archivos del PC:
tf.train.write_graph(grafo_exportado, "modelo", "modelo.pb", as_text=False)
