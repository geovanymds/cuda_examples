import numpy as numpy
import codecs

def read_input(input_path: str,delimiter: str) -> numpy.ndarray:
    with codecs.open(input_path, encoding='utf-8-sig') as file:
       matrix = numpy.loadtxt(file, dtype = numpy.float32, delimiter=delimiter)
    return matrix

def save_output(output_path: str, output_file: str, output_value: numpy.ndarray):
    with open(output_path + output_file, 'ab') as file:
        numpy.savetxt(file, output_value, fmt='%10s')

def reshape_to_square_matrix(matrix: numpy.ndarray):
    [width, height] = numpy.shape(matrix)
    size = numpy.minimum(width, height)
    reshaped_matrix = numpy.reshape(matrix, (size, size))
    return size, reshaped_matrix

def adjust_matrix_to_scale(size: int, scale: int, matrix: numpy.ndarray):
    new_size = scale * int(numpy.trunc(size/scale))
    new_mat = matrix[0: new_size, 0:new_size]
    return new_size, new_mat

def calculate_quantity_submatrices(size: int, scale: int) -> int:
    return int(numpy.power((size / scale), 2))