import numpy as numpy
import os
from pylab import *
from scipy import stats
import time
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from parallelized_cumsum import cumsum
import utils
# -- initialize the device
import pycuda.autoinit


def cumsum_by_scale(matrix_input: numpy.ndarray):
    begin_time = time.time()
    size, reshaped_matrix = utils.reshape_to_square_matrix(matrix_input)
    number_of_scales = int(size/4)
    print(reshaped_matrix)
    k=0
    current_scale = 2
    while (current_scale < 3):
        current_mat = reshaped_matrix
        current_size = size
        if numpy.mod(current_size, current_scale) != 0:
            current_size, current_mat = utils.adjust_matrix_to_scale(current_size, current_scale, current_mat)
        current_mat = cumsum(current_mat,current_scale)
        numpy.savetxt(f"./data/output/output_scale_{current_scale}.txt", current_mat, fmt="%.2f")
        k = k + 1
        current_scale = current_scale + 1
    return

def main(input_path: str, output_path: str, delimiter: str, output_file: str):
    arqs = os.listdir(input_path)
    for file in arqs:
        if file.endswith('.txt'):
            m = utils.read_input(input_path + file,delimiter)
            cumsum_by_scale(m)

main("./data/input/","./data/output/"," ","results.txt")