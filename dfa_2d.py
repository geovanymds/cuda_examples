import numpy as numpy
from scipy import stats
import robust_last_square as fit

import time

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
        vetvar = [fit.fit_2D((numpy.cumsum(m).reshape(current_scale,current_scale)), current_scale, grade) for m in current_mat]
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
