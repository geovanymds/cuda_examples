import numpy as numpy
import os
from pylab import *
from scipy import stats
import time
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from robust_last_square import fit_2D
from parallelized_cumsum import cumsum_y, cumsum_x
import codecs
# -- initialize the device
import pycuda.autoinit

def read_input(input_path: str,delimiter: str) -> ndarray:
    with codecs.open(input_path, encoding='utf-8-sig') as file:
       matrix = numpy.loadtxt(file, dtype = numpy.float32, delimiter=delimiter)
    return matrix

def save_output(output_path: str, output_file: str, output_value: ndarray) -> void:
    with open(output_path + output_file, 'ab') as file:
        numpy.savetxt(file, output_value, fmt='%10s')

def reshape_to_square_matrix(matrix: ndarray):
    [width, height] = numpy.shape(matrix)
    size = numpy.minimum(width, height)
    reshaped_matrix = numpy.reshape(matrix, (size, size)) # VERIFICAR COM A VANESSA, POIS O DESLOCAMENTO DE VALORES DE UMA LINHA INFERIOR PRA UMA SUPERIOR PODERIAM DIMINUIR A PRECISÃO DO ALGORITMO
    return size, reshaped_matrix

def adjust_matrix_to_scale(size: int, scale: int, matrix: ndarray):
    new_size = scale * int(numpy.trunc(size/scale))
    new_mat = matrix[0: new_size, 0:new_size]
    return new_size, new_mat

def calculate_quantity_submatrices(size: int, scale: int) -> int:
    return int(numpy.power((size / scale), 2))

def dfa_2d(matrix_input: ndarray, grade: int):
    begin_time = time.time()
    size, reshaped_matrix = reshape_to_square_matrix(matrix_input)
    number_of_scales = int(size/4)
    vetoutput = numpy.zeros(shape=(number_of_scales - 5, 2))
    integrated_matrix = cumsum_y(reshaped_matrix)
    print(numpy.cumsum(reshaped_matrix,axis=0))
    print("\n")
    print(integrated_matrix)
    k=0
    current_scale = 6
    while (current_scale < number_of_scales + 1):
        current_mat = integrated_matrix
        current_size = size
        if numpy.mod(current_size, current_scale) != 0:
            current_size, current_mat = adjust_matrix_to_scale(current_size, current_scale, current_mat)
        # Passo 1 : Subdivisão da Série Temporal
        submatrices_quantity = calculate_quantity_submatrices(current_size, current_scale) #quantidade de sub-matrizes
        t = numpy.arange(current_scale, current_size, current_scale) #intervalos para o split da matriz em sub-matrizes
        aux = numpy.array(numpy.array_split(current_mat, t, axis=1))
        current_mat = numpy.reshape(aux, (submatrices_quantity, current_scale, current_scale))
        # #Passo 2 : Integração e Remoção da Tendência
        # vetvar = [fit_2D((m.reshape(current_scale,current_scale)), current_scale, grade) for m in current_mat]
        # # 4.Calcula-se a função de flutuação DFA como a média das variâncias de cada intervalo:
        # fs = numpy.sqrt(numpy.mean(vetvar))
        # vetoutput[k, 0] = current_scale
        # vetoutput[k, 1] = fs
        k = k + 1
        current_scale = current_scale + 1
    # vetoutput = numpy.log10(vetoutput)
    # x = vetoutput[:, 0]
    # y = vetoutput[:, 1]
    # slope, _, _, _, _ = stats.linregress(x, y)
    # end_time = time.time()
    # print(slope)
    # return (slope, end_time-begin_time, y.reshape(1,-1))

def main(input_path: str, output_path: str, delimiter: str, output_file: str):
    arqs = os.listdir(input_path)
    arqs = sorted(arqs, key=lambda x: int((x.split('_')[1].split('.')[0])))
    for file in arqs:
        if file.endswith('.txt'):
            # print(file)
            m = read_input(input_path + file,delimiter)
            dfa_2d(m,1)
            # alfa, tempo, F = dfa_2d(m, 1)
            # dt = numpy.dtype(str, 10)
            # b = numpy.array([alfa, tempo], dtype=dt)
            # b = numpy.reshape(b, newshape=(1, 2))
            # save_output(output_path, output_file,b)

main("./data/input/fGn/fGn09/","./data/output/",",","results_fGn09.txt")