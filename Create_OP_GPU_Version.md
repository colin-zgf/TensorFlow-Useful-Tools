# Create OP (GPU Version)

In this secion, an GPU version user defined OP is created and tested. Please make sure that you have installed tensorflow successfully on your computer. The operation is implemented on Ubuntu system. The steps of generating GPU version OP are listed below:

## Step1: create **user_ops** folder under tensorflow "core" directory (e.g. /user/local/lib/python3.5/dist_packages/tensorflow/core)

~~~
sudo mkdir user_ops
~~~

## Step2: Create c++ file. 

~~~
sudo touch cuda_op_kernel.cu.cc cuda_op_kernel.cc
~~~

In this case, the OP receives one "int32" tensors, and add one to this tensor. One needs to create the two files since one relates cuda and the other relates c++. For example, in the file "cuda_op_kernel.cu.cc", one can write the code as follows:

~~~
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <cuda.h>
#include <stdio.h>

__global__ void AddOneKernel(const int* in, const int N, int* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    out[i] = in[i] + 1;
  }
}

void AddOneKernelLauncher(const int* in, const int N, int* out) {
  AddOneKernel<<<32, 256>>>(in, N, out);
}

#endif
~~~

Here **Eigen** is an advanced C++ application which could support linear algebre, matrix and vector calculation, numerical analysis and other related algorithms.

In cuda_op_kernel.cu.cc, the algorithm is written as

~~~
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("AddOne")
    .Input("input: int32")
    .Output("output: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

void AddOneKernelLauncher(const int* in, const int N, int* out);

class AddOneOp : public OpKernel {
 public:
  explicit AddOneOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    // Call the cuda kernel launcher
    AddOneKernelLauncher(input.data(), N, output.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("AddOne").Device(DEVICE_GPU), AddOneOp);
~~~

## Step3: Compile the code

* In python3

~~~
TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
nvcc -std=c++11 -c -o cuda_op_kernel.cu.o cuda_op_kernel.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 -shared -o cuda_op_kernel.so cuda_op_kernel.cc cuda_op_kernel.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda-9.0/lib64/
~~~

**Note, the last part '-L /usr/local/cuda-9.0/lib64/' is to call the library of cuda. Sometimes, it will raise some loading error for TensorFlow if one does not write it.**

## Step4: Test the code

After Step3, now we can test the code.

~~~
import tensorflow as tf

so_file = '/usr/local/lib/python3.5/dist-packages/tensorflow/core/user_ops/cuda_op_kernel.so'
cuda_op_module = tf.load_op_library(so_file)

with tf.Session(''):
    x = cuda_op_module.add_one([[6, 4], [2, 4]]).eval()
print(x)
~~~

The output should be
[[7, 5]
 [3, 5]]
