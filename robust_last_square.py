import numpy as numpy
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

# mod = SourceModule("""
# __global__ void fit(float *dest, float *a, float *b)
# {
#   const int i = threadIdx.x;
#   dest[i] = a[i] * b[i];
# }
# """)

# def test_pycuda():
#     multiply_them = mod.get_function("multiply_them")
#     a = numpy.random.randn(400).astype(numpy.float32)
#     b = numpy.random.randn(400).astype(numpy.float32)

#     dest = numpy.zeros_like(a)
#     multiply_them(
#             drv.Out(dest), drv.In(a), drv.In(b),
#             block=(400,1,1), grid=(1,1))

#     print(dest-a*b)

mod =SourceModule("""
//cuda
    #include <stdio.h>

    __global__ void fit(float *current_integrated_sample, int current_scale, int grade) 
    {
        float a1 = 0.8, a2 = 0.14, a3 = 0.42;
        int array_length = current_scale;
        float *nx, *ny;
        cudaMallocManaged(&nx, array_length*sizeof(float));
        cudaMallocManaged(&ny, array_length*sizeof(float));

  ...
        // Free memory
        cudaFree(nx);
        cudaFree(ny);
    }

//!cuda
""")

def fit_2D(current_integrated_sample, current_scale, grade):
    # coefficients of the model
    a1, a2, a3 = 0.8, 0.14, 0.42
    # create a coordinate matrix
    nx = numpy.arange(1,current_scale+1)
    ny = numpy.arange(1,current_scale+1)
    x, y = numpy.meshgrid(nx, ny)

    # make the estimation
    if grade == 1:
        z = a1 * x + a2 * y + a3
    if grade == 2:
        z = a1 * x * x + a2 * y * y + a3
        x = numpy.power(x, 2)
        y = numpy.power(y, 2)

    # non-robust least squares estimation

    x_fl = x.flatten()
    y_fl = y.flatten()
    z_ones = numpy.ones([x.size, 1])
    X = numpy.hstack((numpy.reshape(x_fl, ([len(x_fl), 1])), numpy.reshape(y_fl, ([len(y_fl), 1])), z_ones))
    Z_fl = current_integrated_sample.flatten()
    Z = numpy.reshape(Z_fl, ([len(Z_fl), 1]))
    A_lsq = numpy.linalg.lstsq(X, Z)[0]

    # robust least sqaures (starting with the least squares solution)
    A_robust = A_lsq

    resid = numpy.dot(X, A_robust) - Z
    resid = numpy.power(resid,2)
    t = numpy.sum(resid)/numpy.power(current_scale,2)
    return t