#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("MyAdd")
    .Input("x: int32")
    .Input("y: int32")
    .Output("z: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
      c->set_output(0, c->input(0));
      c->set_output(0, c->input(1));
      return Status::OK();
    });


#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class MyAddOp : public OpKernel {

 public:
  explicit MyAddOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //Grab the input tensor
    const Tensor& a = context->input(0);
    const Tensor& b = context->input(1);
    auto A = a.flat<int32>();
    auto B = b.flat<int32>();
    //Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, a.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    //Set all but the first element of the output tensor to 0
    const int N = A.size();

    for (int i = 1; i < N; i++) {
       output_flat(i) = A(i) + B(i);
    }
    output_flat(0) = 0;
  }
};


REGISTER_KERNEL_BUILDER(Name("MyAdd").Device(DEVICE_CPU), MyAddOp);
