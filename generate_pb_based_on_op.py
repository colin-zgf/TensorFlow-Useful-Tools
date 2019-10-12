import tensorflow as tf
from tensorflow.python.framework import graph_util

# Generate pb
v1 = tf.placeholder(shape=[None], dtype=tf.int32, name='v1')
v2 = tf.placeholder(shape=[None], dtype=tf.int32, name='v2')
so_file = 'code/user_ops/my_add.so'
my_add_module = tf.load_op_library(so_file)
c = my_add_module.my_add(v1, v2)
c = tf.identity(c, name='sum')
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
graph_def = tf.get_default_graph().as_graph_def()
output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['sum'])
model_f = tf.gfile.GFile('model.pb', 'wb')
model_f.write(output_graph_def.SerializeToString())