import numpy as numpy
import os
import dfa_2d as dfa
from pylab import *
import utils as utils

def main(input_path, output_path, delimiter, output_file):
    arqs = os.listdir(input_path)
    arqs = sorted(arqs, key=lambda x: int((x.split('_')[1].split('.')[0])))
    for file in arqs:
        if file.endswith('.txt'):
            print(file)
            m = utils.readInput(input_path + file,delimiter)
            alfa, tempo, F = dfa.dfa_2d(m, 1)
            dt = numpy.dtype(str, 10)
            b = numpy.array([alfa, tempo], dtype=dt)
            b = numpy.reshape(b, newshape=(1, 2))
            with open(output_path + output_file, 'ab') as f:
                numpy.savetxt(f, b, fmt='%10s')

main("./data/input/fGn/fGn09/","./data/output/",",","results_fGn09.txt")