from memory import UnsafePointer

# ANCHOR: softmax_gpu_kernel
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from math import exp
from utils.numerics import max_finite, min_finite


alias SIZE = 128
alias TPB = 128
alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (TPB, 1)
alias layout = Layout.row_major(SIZE)


fn softmax_gpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=True, dtype, layout],
    input: LayoutTensor[mut=False, dtype, layout],
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    reduce = tb[dtype]().row_major[SIZE]().shared().alloc()
    if global_i < SIZE:
        shared[local_i] = input[local_i]
        reduce[local_i] = input[local_i]
    barrier()
    stride = TPB // 2
    while stride > 0:
        var s: output.element_type = 0
        if local_i < stride:
            s = reduce[local_i + stride]
        barrier()
        if local_i < stride:
            reduce[local_i] = max(s, reduce[local_i])
        barrier()
        stride //= 2

    shared = tb[dtype]().row_major[SIZE]().shared().alloc()
    var max_val: output.element_type = reduce[0] 
    if global_i < SIZE:
        exp = exp(shared[local_i] - max_val)
        shared[local_i] = exp
        reduce[local_i] = exp
    barrier()
    stride = TPB // 2
    while stride > 0:
        var s: output.element_type = 0
        if local_i < stride:
            s = reduce[local_i + stride]
        barrier()
        if local_i < stride:
            reduce[local_i] += s
        barrier()
        stride //= 2
    var sum_exp: output.element_type = reduce[0]  
    if global_i < SIZE:
        output[global_i] = shared[local_i] / sum_exp


# ANCHOR_END: softmax_gpu_kernel


# ANCHOR: softmax_cpu_kernel
fn softmax_cpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout, MutableAnyOrigin],
    input: LayoutTensor[dtype, layout, MutableAnyOrigin],
):
    var max_val: input.element_type = Scalar[dtype].MIN_FINITE
    for i in range(input_size):
        max_val = max(max_val, input[i])

    var sum_exp: input.element_type = 0.0
    for i in range(input_size):
        var exp_val = exp(input[i] - max_val)
        output[i] = exp_val
        sum_exp += exp_val

    for i in range(input_size):
        output[i] = output[i] / sum_exp


# ANCHOR_END: softmax_cpu_kernel

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor


@compiler.register("softmax")
struct SoftmaxCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,  # "cpu" or "gpu"
        input_size: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[rank=1],
        input: InputTensor[rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        # Note: rebind is necessary now but it shouldn't be!
        var output_tensor = rebind[
            LayoutTensor[dtype, layout, MutableAnyOrigin]
        ](output.to_layout_tensor())
        var input_tensor = rebind[
            LayoutTensor[dtype, layout, MutableAnyOrigin]
        ](input.to_layout_tensor())
        alias layout = input_tensor.layout

        @parameter
        if target == "gpu":
            gpu_ctx = ctx.get_device_context()
            # making sure the output tensor is zeroed out before the kernel is called
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output_tensor.dtype](
                    gpu_ctx,
                    rebind[UnsafePointer[Scalar[output_tensor.dtype]]](
                        output_tensor.ptr
                    ),
                    input_size,
                    owning=False,
                ),
                0,
            )

            gpu_ctx.enqueue_function[
                softmax_gpu_kernel[layout, input_size, dtype]
            ](
                output_tensor,
                input_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=(TPB, 1),
            )

        elif target == "cpu":
            softmax_cpu_kernel[layout, input_size, dtype](
                output_tensor, input_tensor
            )
        else:
            raise Error("Unsupported target: " + target)
