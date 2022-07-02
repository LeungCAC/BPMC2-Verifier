import network_computation as Nc
import poly_linear as pl
import poly_relu as pr
import numpy as np
import poly_relu as pr
import poly_convertion as pcon

def back_propagation(sample, order, size):
    for dim in range(order*size, (order+1)*size):
        sample = pr.backward_relu(sample, dim)
    bs = pr.backward_linear(sample)
    return bs

def mc_sample (poly):
    p_point = poly.point
    p_ray = poly.ray

    cp, _ = p_point.shape
    cr = np.shape(p_ray)[0]

    alpha = np.random.rand(cp)
    alpha = alpha / np.sum(alpha)
    gamma = np.random.rand(cr) * np.random.randint(1, 10)

    sample = np.matmul(alpha, p_point) + np.matmul(gamma, p_ray)
    return sample

"""
p = np.array([[0,0],[1,1],[1,0]])
r = np.array([[1,0]])
poly= pcon.Poly(p,r)
print(mc_sample(poly))
"""

def whole_back_computation(layer, K, W, U, B, point_set):
    for l in range(layer-1, -1, -1):
        if (l == layer-1):
            for i in range(K-1, -1, -1):
                M, bias = Nc.con_matrix(i, K, W[l], U[l], B[l])
                Pweight = M
                Pbias = bias
                point_set = pl.set_backward_linear(point_set, Pbias, Pweight)

        else:
            for i in range(K-1, -1, -1):
                M, bias = Nc.con_matrix(i, K, W[l], U[l], B[l])
                Pweight = M
                Pbias = bias
                point_set = pr.whole_backward_relu(point_set, i, np.shape(U[l])[0])
                point_set = pl.set_backward_linear(point_set, Pbias, Pweight)

            return point_set