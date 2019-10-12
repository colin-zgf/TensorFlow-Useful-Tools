import tensorflow as tf

so_file = '/usr/local/lib/python3.5/dist-packages/tensorflow/core/user_ops/cuda_op_kernel.so'
cuda_op_module = tf.load_op_library(so_file)

with tf.Session(''):
    x = cuda_op_module.add_one([[6, 4], [2, 4]]).eval()
print(x)