# Se importan los paquetes necesarios:
from keras import backend as K
from keras.models import load_model

# Se pone la fase de aprendizaje a cero para ponerlo en modo test y se carga el modelo completo:
K.set_learning_phase(0)
modelo = load_model('/home/user/T-F-M/Valderas/modelos/modelo.h5', compile = False)

# Se importan los paquetes necesarios:
from keras import backend as K
import tensorflow as tf

# Se define una función para conseguir un grafo computacional:
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """ Exporta el estado de una sesión y crea un grafo computacional.
    Se crea un nuevo grafo computacional donde los nodos variables se reemplazan por
    constantes con su valor actual en la sesión. Este nuevo grafo será reducido, de forma 
    que los subgrafos no necesarios para la computación de la salida serán eliminados.

    @param session: La sesión de TensorFlow que será exportada.
    @param keep_var_names: Una lista de nombres de variables que no deberán ser exportadas,
                          o Ninguno para exportar todas las variables en el grafo.
    @param output_names: Nombres de las salidas del grafo relevantes.
    @param clear_devices: Elimina las directrices del dispositivo del grafo para una mejor portabilidad.
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

# Se exporta la sesión y se transforma el grafo a un archivo '.pb', eliminando subgrafos innecesarios:
grafo_exportado = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in modelo.outputs])

tf.train.write_graph(grafo_exportado, "modelo", "modelo.pb", as_text=False)
