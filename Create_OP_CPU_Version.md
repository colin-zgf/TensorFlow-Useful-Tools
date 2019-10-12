# Create OP (CPU Version)

In this secion, an CPU version user defined OP is created and tested. Please make sure that you have installed tensorflow successfully on your computer. The operation is implemented on Ubuntu system. The steps of generating CPU version OP are listed below:

## Step1: create **user_ops** folder under tensorflow "core" directory (e.g. /user/local/lib/python3.5/dist_packages/tensorflow/core)

~~~
sudo mkdir user_ops
~~~

## Step2: Create c++ file. 

~~~
sudo touch my_add.cc
~~~

In this case, the OP receives two "int32" tensors, add this two tensors, and finally set the first element of the sum to be zero. Thus, in the file "my_add.cc", one can write the code as follows:

~~~
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("MyAdd")
    .Input("x: int32")
    .Input("y: int32")
    .Output("z: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(0, c->input(1));
      return Status::OK();
    });

class MyAddOp : public OpKernel {
 public:
  explicit MyAddOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& a = context->input(0);
    const Tensor& b = context->input(1);
    auto A = a.flat<int32>();
    auto B = b.flat<int32>();
    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, a.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = A.size();

    for (int i = 1; i < N; i++) {
      output_flat(i) = A(i)+B(i);
      output_flat(0) = 0;
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("MyAdd").Device(DEVICE_CPU), MyAddOp);
~~~

## Step3: Compile the code

* In python2
~~~
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
g++ -std=c++11 -shared my_add.cc -o my_add.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2
~~~

* In python3

~~~
TF_INC=$(python3.5 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python3.5 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
g++ -std=c++11 -shared my_add.cc -o my_add.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2
~~~

## Step4: Test the code

After Step3, now we can test the code.

~~~
import tensorflow as tf

so_file = '/usr/local/lib/python3.5/dist-packages/tensorflow/core/user_ops/my_add.so'
op_module = tf.load_op_library(so_file)

with tf.Session(''):
    x = op_module.my_add([6, 4], [2, 4]).eval()
print(x)
~~~

The output should be
[0, 6]
