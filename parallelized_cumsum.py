from numpy import ndarray
from pycuda import compiler, gpuarray
import math
import numpy as numpy

MAX_THREADS_PER_BLOCK = 1024
MAX_DIM_X_BLOCKS = int(math.pow(2,31) - 1)
MAX_DIM_Y_BLOCKS = 65535

kernel_code_template_cumsum_x = """
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
      }
      __syncthreads();
    }
//!cuda
"""

kernel_code_template_cumsum_y = """
//cuda
  __global__ void y_axis_cumsum(float *matrix_input, float *matrix_output)
    {
      const uint bx = blockIdx.x;
      const uint by = blockIdx.y;

      const uint ty = threadIdx.y;

      const uint begin = bx * %(MAX_DIM_X_BLOCKS)s + by * %(MAX_THREADS_PER_BLOCK)s + ty;

      matrix_output[begin] = matrix_input[begin];

      for(uint index = %(MATRIX_WIDTH)s; index <= %(MATRIX_WIDTH)s * %(MATRIX_HEIGHT)s - %(MATRIX_WIDTH)s; index+=%(MATRIX_WIDTH)s) {
        matrix_output[begin+index] = matrix_input[begin+index] + matrix_output[begin+index - %(MATRIX_WIDTH)s];
      }
      __syncthreads();
    }
//!cuda
"""

def get_formated_kernel_function(formatter_options: dict, kernel_code_template: str) -> str:
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

def cumsum(kernel_code_template: str, kernel_function_name: str, formatter_options: dict, matrix_input: ndarray, axis: int):
    matrix_input_gpu = gpuarray.to_gpu(matrix_input)
    #ALOCATE A EMPTY MATRIX TO THE RESULT
    matrix_output = gpuarray.empty((matrix_input.shape[0],matrix_input.shape[1]),numpy.float32);
    kernel_code = get_formated_kernel_function(formatter_options,kernel_code_template)
    # #COMPILE KEWRNEL CODE
    mod = compiler.SourceModule(kernel_code)
    block_params = get_block_params(matrix_input)
    cumsum = mod.get_function(kernel_function_name)
    cumsum(
        #INPUTS
        matrix_input_gpu, 
        #OUTPUT
        matrix_output,
        block = (block_params['x_axis_threads'], block_params['y_axis_threads'], block_params['number_of_blocks']),
    )
    return matrix_output
    
def cumsum_x(matrix: ndarray):
    kernel_code_formatter_dict_x = {
        'MAX_DIM_X_BLOCKS': MAX_DIM_X_BLOCKS,
        'MAX_THREADS_PER_BLOCK': MAX_THREADS_PER_BLOCK,
        'MATRIX_WIDTH': matrix.shape[0]
    }
    return numpy.array(cumsum(kernel_code_template_cumsum_x,"x_axis_cumsum", kernel_code_formatter_dict_x, matrix,1).get())

def cumsum_y(matrix: ndarray):
    kernel_code_formatter_dict_y = {
        'MAX_DIM_X_BLOCKS': MAX_DIM_X_BLOCKS,
        'MAX_THREADS_PER_BLOCK': MAX_THREADS_PER_BLOCK,
        'MATRIX_WIDTH': matrix.shape[0],
        'MATRIX_HEIGHT': matrix.shape[1]
    }
    return numpy.array(cumsum(kernel_code_template_cumsum_y, "y_axis_cumsum", kernel_code_formatter_dict_y, matrix,0).get())