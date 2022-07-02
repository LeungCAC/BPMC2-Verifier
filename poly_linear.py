import numpy as np
import poly_convertion as pcon

# linear transformatio: matrix multiply and bias addition

def matrix_multiply (weight, poly):
    w = weight
    p_point = poly.point
    p_ray = poly.ray
    if (np.shape(p_point)[0] == 0):
        point = p_point
    else:
        point = np.matmul (p_point, w.T)
        point = np.unique(point, axis=0)

    if (np.shape(p_ray)[0] == 0):
        ray = p_ray
    else:
        ray = np.matmul (p_ray, w.T)
        ray = np.unique(ray, axis=0)
    #print(point,"\n", ray)
    #print("Stop Here", np.shape(point), np.shape(ray))
    p = pcon.Poly(np.array(point), np.array(ray))
    #print("matrix_multi",p.point, p.ray)
    return p

"""
w = np.array([[1,3],[3,2],[1,-2],[-1,-2]])  # row:next layer dimension, col:current layer dimension
p = np.array([[0,0],[1,1],[1,0]])
r = np.array([[1,0]])
poly= pcon.Poly(p,r)
x = matrix_multiply(w, poly)
print(x.point,x.ray)
"""

def bias_addition (poly, bias):
    bias = bias.T
    #print("bias addition",np.shape(poly.point), np.shape(poly.ray), np.shape(bias))
    if ( np.shape(poly.point)[0] == 0 ):
        return poly
    else:
        p_point = poly.point + bias
        p_point = np.unique(p_point, axis=0)
    #print("stop", poly.point, bias, p_point)

    p_ray = poly.ray
    if ( np.shape(poly.ray)[0] == 0 ):
        pass
    else:
        p_ray = np.unique(p_ray, axis=0)
    p = pcon.Poly(np.array(p_point), np.array(p_ray))
    #print("bias addition",p_point, p_ray, np.shape(p_point), np.shape(p_ray))
    return p

"""
bias = np.array([0,2])
p = np.array([[0,0],[1,1],[1,0]])
r = np.array([[1,0]])
poly= pcon.Poly(p,r)
print(bias_addition(poly,bias).point, bias_addition(poly,bias).ray)
"""


def backward_linear(point, bias, M):
    bp = point
    bias = bias.reshape(1,-1)[0]
    bp = np.subtract(bp, bias)
    bp = np.matmul(bp, np.linalg.pinv(M.T))
    return bp

"""
point = [2, 2]
bias = [0.5, 0.5]
M = np.array([[1,2],[2,1]])
print(backward_linear(point, bias, M))
"""     

def set_matrix_multiply (weight, poly_set):
    ps = []
    for poly in poly_set:
        ps.append(matrix_multiply(weight, poly))
    return ps


def set_bias_addition(poly_set, bias):
    ps = []
    for poly in poly_set:
        ps.append(bias_addition(poly, bias))
    return ps


def set_backward_linear(point_set, bias, M):
    bp_set = []
    for point in point_set:
        bp = backward_linear(point, bias, M)
        bp_set.append(bp)
    return bp_set