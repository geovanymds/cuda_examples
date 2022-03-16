from pycuda import driver, compiler, gpuarray, tools
import math
import numpy as numpy

# -- initialize the device
import pycuda.autoinit

MAX_THREADS_PER_BLOCK = 1024
MAX_DIM_X_BLOCKS = int(math.pow(2,31) - 1)
MAX_DIM_Y_BLOCKS = 65535

kernel_code_template = """
//cuda
  __global__ void x_axis_cumsum(float *matrix_input, float *matrix_output)
    {
      const uint bx = blockIdx.x;
      const uint by = blockIdx.y;

      const uint ty = threadIdx.y;

      const uint begin = (bx * %(MAX_DIM_X_BLOCKS)s + by * %(MAX_THREADS_PER_BLOCK)s + ty) * %(MATRIX_WIDTH)s;

      matrix_output[begin] = matrix_input[begin];

      for(uint index = 1; index < %(MATRIX_WIDTH)s; index++) {
        matrix_output[begin+index] = matrix_input[begin+index] + matrix_output[begin+index - 1];
        // printf("%%d\\n",begin+index);
      }
    }
//!cuda
"""

def get_formated_kernel_function(formatter_options: dict) -> str:
    return (kernel_code_template%formatter_options)

def read_input(file_name: str) -> list[list[numpy.float32]]:
    with open(file_name, 'r') as file:
        matrix = [[numpy.float32(number) for number in line.split()] for line in file]
    return matrix

def get_block_params(matrix:numpy.ndarray):
    return {
      'x_axis_threads': 1,
      'y_axis_threads': int(matrix.shape[0]) if int(matrix.shape[0]) <= MAX_THREADS_PER_BLOCK else MAX_THREADS_PER_BLOCK,
      'number_of_blocks': int(math.ceil(matrix.shape[0]/MAX_THREADS_PER_BLOCK))
    }

def main():
    matrix_input = numpy.matrix(read_input('matrix_1000x1000_1.txt')).astype(numpy.float32)
    
    matrix_input_gpu = gpuarray.to_gpu(matrix_input)
    #ALOCATE A EMPTY MATRIX TO THE RESULT
    matrix_output = gpuarray.empty((matrix_input.shape[0],matrix_input.shape[1]), numpy.float32);
    kernel_code_formatter_dict = {
            'MAX_DIM_X_BLOCKS': MAX_DIM_X_BLOCKS,
            'MAX_THREADS_PER_BLOCK': MAX_THREADS_PER_BLOCK,
            'MATRIX_WIDTH': matrix_input.shape[0]
        }
    kernel_code = get_formated_kernel_function(kernel_code_formatter_dict)
    # #COMPILE KEWRNEL CODE
    mod = compiler.SourceModule(kernel_code)
    block_params = get_block_params(matrix_input)
    x_cumsum = mod.get_function("x_axis_cumsum")
    x_cumsum(
        #INPUTS
        matrix_input_gpu, 
        #OUTPUT
        matrix_output,
        # # grid of multiple blocks
        # grid = (matrix_size // tile_size, matrix_size // tile_size), 
        # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
        block = (block_params['x_axis_threads'], block_params['y_axis_threads'], block_params['number_of_blocks']),
    )
    print(numpy.cumsum(matrix_input, axis=1))
    print("\n")
    print(matrix_output)
    # print(block_dimensions)

main()