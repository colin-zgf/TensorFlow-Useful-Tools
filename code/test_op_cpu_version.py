import tensorflow as tf

so_file = '/usr/local/lib/python3.5/dist-packages/tensorflow/core/user_ops/my_add.so'
op_module = tf.load_op_library(so_file)

with tf.Session(''):
    x = op_module.my_add([6, 4], [2, 4]).eval()
print(x)
