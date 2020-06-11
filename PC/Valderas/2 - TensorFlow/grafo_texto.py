import tensorflow as tf

# Lee el modelo binario de TF:
with tf.gfile.FastGFile('/home/user/T-F-M/Valderas/modelos/modelo.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# Elimina nodos 'Const'.
for i in reversed(range(len(graph_def.node))):
    if graph_def.node[i].op == 'Const':
        del graph_def.node[i]
    for attr in ['T', 'data_format', 'Tshape', 'N', 'Tidx', 'Tdim',
                 'use_cudnn_on_gpu', 'Index', 'Tperm', 'is_training',
                 'Tpaddings']:
        if attr in graph_def.node[i].attr:
            del graph_def.node[i].attr[attr]

# Guarda como archivo de texto en el PC:
tf.train.write_graph(graph_def, "modelo", "modelo.pbtxt", as_text=True)

