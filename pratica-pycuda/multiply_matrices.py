from pycuda import driver, compiler, gpuarray, tools
import numpy as numpy

# -- initialize the device
import pycuda.autoinit

kernel_code_template = """
//cuda
  __global__ void multiply_matrices(float *matrix_a, float *matrix_b, float *C)
    {

      const uint w_matrix_a = %(matrix_size)s;
      const uint w_matrix_b = %(matrix_size)s;

      const uint bx = blockIdx.x;
      const uint by = blockIdx.y;

      const uint tx = threadIdx.x;
      const uint ty = threadIdx.y;

      const uint a_begin = w_matrix_a * %(block_size)s * by;
      const uint a_end = a_begin + w_matrix_a - 1;
      const uint a_step = %(block_size)s;

      const uint b_begin = %(block_size)s * bx;
      const uint b_step = %(block_size)s * w_matrix_b;

      float Csub = 0;
      for (int a = a_begin, b = b_begin;
           a <= a_end;
           a += a_step, b += b_step)
        {
          __shared__ float matrix_a_s[%(block_size)s][%(block_size)s];
          __shared__ float matrix_b_s[%(block_size)s][%(block_size)s];

          matrix_a_s[ty][tx] = matrix_a[a + w_matrix_a * ty + tx];
          matrix_b_s[ty][tx] = matrix_b[b + w_matrix_b * ty + tx];

          __syncthreads();

          for (int k = 0; k < %(block_size)s; ++k)
            Csub += matrix_a_s[ty][k] * matrix_b_s[k][tx];

          __syncthreads();
        }

      const uint c = w_matrix_b * %(block_size)s * by + %(block_size)s * bx;
      C[c + w_matrix_b * ty + tx] = Csub;
    }
//!cuda
"""

def get_formated_kernel_function(formatter_options: dict) -> str:
    return (kernel_code_template%formatter_options)
    
def read_input(file_name: str) -> list[list[numpy.float32]]:
    with open(file_name, 'r') as file:
        matrix = [[numpy.float32(number) for number in line.split()] for line in file]
    return matrix

def compute_product_matrix_dimensions(matrix_a:numpy.ndarray, matrix_b:numpy.ndarray):
    rows_a = matrix_a.shape[0]
    columns_b = matrix_b.shape[1]
    return rows_a, columns_b

def is_square_matrix(matrix:numpy.ndarray):
    return matrix.shape[0] == matrix.shape[1]

# COMPUTE THE TILE SIZE TO AVOID OVERFLOW BLOCK LIMIT
def compute_tile_size(matrix_size: int) -> int:
    computed_tile_size = matrix_size//10 if matrix_size%10 == 0 else matrix_size//2
    while(computed_tile_size%10 == 0 and computed_tile_size > 60):
        computed_tile_size = computed_tile_size//10
    while(computed_tile_size%2 == 0 and computed_tile_size > 60):
        computed_tile_size = computed_tile_size//2
    return computed_tile_size

def main():
    try:
        matrix_a = numpy.matrix(read_input('/data/matrix_8x8_1.txt')).astype(numpy.float32)
        matrix_b = numpy.matrix(read_input('/data/matrix_8x8_2.txt')).astype(numpy.float32)

        if(not is_square_matrix(matrix_a) or not is_square_matrix(matrix_b)):
            raise Exception("Matriz não quadrada utilizada na entrada.") 

        matrix_size = int(compute_product_matrix_dimensions(matrix_a,matrix_b)[0])
        tile_size = compute_tile_size(matrix_size)
        block_size = tile_size

        #PRODUCCT USING CPU
        matrix_axb = numpy.dot(matrix_a,matrix_b)
        print(matrix_axb)

        #TRANSFERING THE MATRICES FROM HOST MEMORY TO GPU MEMORY 
        matrix_a_gpu = gpuarray.to_gpu(matrix_a) 
        matrix_b_gpu = gpuarray.to_gpu(matrix_b)
        matrix_axb_gpu = gpuarray.empty((matrix_size,matrix_size), numpy.float32)

        kernel_code_formatter_dict = {
            'matrix_size': matrix_size,
            'block_size': block_size,
        }
        kernel_code = get_formated_kernel_function(kernel_code_formatter_dict)
        print(matrix_size)
        print(tile_size)
        # #COMPILE KEWRNEL CODE
        mod = compiler.SourceModule(kernel_code)
        # #GET KERNEL FUNCTION
        multiply_function = mod.get_function("multiply_matrices")
        multiply_function(
            #INPUTS
            matrix_a_gpu, matrix_b_gpu, 
            #OUTPUT
            matrix_axb_gpu,
            # grid of multiple blocks
            grid = (matrix_size // tile_size, matrix_size // tile_size), 
            # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
            block = (tile_size, tile_size, 1),
        )
        print(matrix_axb_gpu)
        # print(matrix_axb)
    except Exception as error:
        print(f"Error: {error}")
    finally:
        print("Closing ...")        

main()