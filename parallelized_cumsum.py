from numpy import ndarray
from pycuda import compiler, gpuarray
import numpy as numpy


kernel_code_template = """
//cuda
  __global__ void cumsum(float *m_in, float *m_out)
    {
      const uint bx = blockIdx.x;
      const uint by = blockIdx.y;

      const uint begin = bx * %(SCALE)s * %(SCALE)s * (%(MATRIX_WIDTH)s/%(SCALE)s) + by * %(SCALE)s;
      uint j = begin;
      
      m_out[j] = m_in[j];
      
      uint k = 1; //MATRIX LENGHT COUNTER
      uint c = 1; //ROW COUNTER
      uint step = ((%(MATRIX_WIDTH)s/%(SCALE)s)-1)*%(SCALE)s;//STEP ON CHANGE ROW
      j+=1;

      while(k < %(SCALE)s * %(SCALE)s) {
        while(c < %(SCALE)s) {
          if(c == 0) {
            m_out[j] = m_in[j] + m_out[j - step - 1];
          } else {
            m_out[j] = m_in[j] + m_out[j - 1];
          }
          c+=1;
          j+=1;
          k+=1;
        }
        j+=step;
        c = 0;
      }
    }
//!cuda
"""

def get_formated_kernel_function(formatter_options: dict, kernel_code_template: str) -> str:
    return (kernel_code_template%formatter_options)

def get_grid_dimension(m: numpy.ndarray, scale: int):
    return int(m.shape[0]/scale), int(m.shape[1]/scale)

def cumsum(m: ndarray, scale:int):
    formatter_options = {
        'MATRIX_WIDTH': m.shape[1],
        'SCALE': scale
    }
    mg_in = gpuarray.to_gpu(m)
    #ALOCATE A EMPTY MATRIX TO THE RESULT
    mg_out = gpuarray.empty((m.shape[0], m.shape[1]), numpy.float32)
    kernel_code = get_formated_kernel_function(formatter_options,kernel_code_template)
    # #COMPILE KEWRNEL CODE
    mod = compiler.SourceModule(kernel_code)
    cumsum = mod.get_function("cumsum")
    x_blocks, y_blocks = get_grid_dimension(m,scale)
    cumsum(
        #INPUTS
        mg_in, 
        #OUTPUT
        mg_out,
        block = (1,1,1),
        grid = (x_blocks, y_blocks, 1)
    )
    return numpy.array(mg_out.get())