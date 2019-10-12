import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np

# Load pb
a = np.array([1, 2, 6], dtype=np.int32)
b = np.array([5, 4, 0], dtype=np.int32)
so_file = 'code/user_ops/my_add.so'
my_add_module = tf.load_op_library(so_file)
with tf.Session() as sess:
    model_f = gfile.FastGFile('model.pb', 'rb')
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(model_f.read())
    _ = tf.import_graph_def(graph_def, name='')
    input1 = sess.graph.get_tensor_by_name('v1:0')
    input2 = sess.graph.get_tensor_by_name('v2:0')
    sum = sess.graph.get_tensor_by_name('sum:0')
    result = sess.run(sum, feed_dict={input1: a, input2: b})
    print(result)