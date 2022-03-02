import numpy as numpy

#ORIGINAL NÃO PARALELISADO
def fit_2D(current_integrated_sample, current_scale, grade):
    # coefficients of the model
    a1, a2, a3 = 0.8, 0.14, 0.42
    # create a coordinate matrix
    nx = numpy.arange(1,current_scale+1)
    ny = numpy.arange(1,current_scale+1)
    x, y = numpy.meshgrid(nx, ny)

    # make the estimation
    if grade == 1:
        z = a1 * x + a2 * y + a3
    if grade == 2:
        z = a1 * x * x + a2 * y * y + a3
        x = numpy.power(x, 2)
        y = numpy.power(y, 2)

    # non-robust least squares estimation

    x_fl = x.flatten()
    y_fl = y.flatten()
    z_ones = numpy.ones([x.size, 1])
    X = numpy.hstack((numpy.reshape(x_fl, ([len(x_fl), 1])), numpy.reshape(y_fl, ([len(y_fl), 1])), z_ones))
    Z_fl = current_integrated_sample.flatten()
    Z = numpy.reshape(Z_fl, ([len(Z_fl), 1]))
    A_lsq = numpy.linalg.lstsq(X, Z)[0]

    # robust least sqaures (starting with the least squares solution)
    A_robust = A_lsq

    resid = numpy.dot(X, A_robust) - Z
    resid = numpy.power(resid,2)
    t = numpy.sum(resid)/numpy.power(current_scale,2)
    return t