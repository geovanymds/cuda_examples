import numpy as numpy
import os
from pylab import *
import utils as utils
from scipy import stats
import time
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from robust_last_square import fit_2D

# mod =SourceModule("""
# //cuda
#     #include <stdio.h>

#     __global__ void fit(float *current_integrated_sample, int current_scale, int grade) 
#     {
#         // float a1 = 0.8, a2 = 0.14, a3 = 0.42;
#         int array_length = current_scale;
#         extern __shared__ int nx[];
#         extern __shared__ int ny[];
#         for (int index = 0; index < array_length; index++) {
#             nx[index] = index + 1;
#             ny[index] = index + 1;
#         }
#         __syncthreads();
#     }

# //!cuda
# """)

# local_fit = mod.get_function("fit")

# def fit_2D(current_integrated_sample, current_scale, grade):
#     block_size = (current_scale, 1, 1)
#     grid_size = numpy.floor((len(current_integrated_sample)/current_scale)).astype(int)
#     grid = (grid_size, grid_size, 1)
#     local_fit(
#         #inputs
#         current_integrated_sample, current_scale, grade,
#         #outputs
#         #kernel params
#         block=block_size,
#         grid = (grid_size,grid_size, 1)
#         )

def dfa_2d(mat, grade):
    begin_time = time.time()
    [width, height] = numpy.shape(mat)
    size = numpy.minimum(width, height)
    mat = numpy.reshape(mat, (size, size))
    number_of_scales = int(size/4)
    vetoutput = numpy.zeros(shape=(number_of_scales - 5, 2))
    k=0
    current_scale = 6
    while (current_scale < number_of_scales + 1):
        current_mat = mat
        current_size = size
        if numpy.mod(current_size, current_scale) != 0:
            current_size = current_scale * int(numpy.trunc(current_size / current_scale))
            current_mat = mat[0:current_size, 0:current_size]
        # Passo 1 : Subdivisão da Série Temporal
        qt = int(numpy.power((current_size / current_scale), 2)) #quantidade de sub-matrizes
        t = numpy.arange(current_scale, current_size, current_scale) #intervalos para o split da matriz em sub-matrizes
        aux = numpy.array(numpy.array_split(current_mat, t, axis=1))
        current_mat = numpy.reshape(aux, (qt, current_scale, current_scale))
        # Passo 2 : Integração e Remoção da Tendência
        vetvar = [fit_2D((numpy.cumsum(m).reshape(current_scale,current_scale)), current_scale, grade) for m in current_mat]
        # 4.Calcula-se a função de flutuação DFA como a média das variâncias de cada intervalo:
        fs = numpy.sqrt(numpy.mean(vetvar))
        vetoutput[k, 0] = current_scale
        vetoutput[k, 1] = fs
        k = k + 1
        current_scale = current_scale + 1
    #vetoutput = numpy.log10(vetoutput[1::1, :])
    vetoutput = numpy.log10(vetoutput)
    x = vetoutput[:, 0]
    y = vetoutput[:, 1]
    slope, _, _, _, _ = stats.linregress(x, y)
    end_time = time.time()
    print(slope)
    return (slope, end_time-begin_time, y.reshape(1,-1))

def main(input_path, output_path, delimiter, output_file):
    arqs = os.listdir(input_path)
    arqs = sorted(arqs, key=lambda x: int((x.split('_')[1].split('.')[0])))
    for file in arqs:
        if file.endswith('.txt'):
            print(file)
            m = utils.read_input(input_path + file,delimiter)
            alfa, tempo, F = dfa_2d(m, 1)
            dt = numpy.dtype(str, 10)
            b = numpy.array([alfa, tempo], dtype=dt)
            b = numpy.reshape(b, newshape=(1, 2))
            with open(output_path + output_file, 'ab') as f:
                numpy.savetxt(f, b, fmt='%10s')

main("./data/input/fGn/fGn09/","./data/output/",",","results_fGn09.txt")