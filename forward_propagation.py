import network_computation as Nc
import poly_linear as pl
import poly_relu as pr
import numpy as np
import poly_relu as pr
import cdd


def whole_computation(layer, K, W, U, B, poly_set):
    for l in range(0, layer):
        if (l < layer-1):
            for i in range(0, K):
                M, bias = Nc.con_matrix(i, K, W[l], U[l], B[l])
                Pweight = M
                Pbias = bias

                poly_set = pl.set_matrix_multiply (Pweight, poly_set)
                poly_set = pl.set_bias_addition(poly_set, Pbias)
                poly_set = pr.whole_relu(poly_set, i, np.shape(U[l])[0])

        else:
            for i in range(0, K):
                M, bias = Nc.con_matrix(i, K, W[l], U[l], B[l])
                Pweight = M
                Pbias = bias
                poly_set = pl.set_matrix_multiply (Pweight, poly_set)
                poly_set = pl.set_bias_addition(poly_set, Pbias)

            return poly_set

















