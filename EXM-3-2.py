import forward_propagation as fp
import cdd
import numpy as np
import poly_convertion as pc
import qq_verify as qv
import poly_convertion as pcon

W = [np.array([0.35,-0.74]).reshape(-1,1), np.array([[ 0.63,  0.77],
       [-0.28,  -0.48]])]
U = [np.array([[  0.12,  0.45],
       [0.76,  -0.44]]), np.array([[ 0, 0],[ 0, 0]])]

B = [np.array([0.27, 0.63]), np.array([ -0.72, 0.42])]

layer = 2
K = 3

def relu(x):
    return np.maximum(0, x)

IA = np.array([[1, 0, 0], [0, 1, 0],[0, 0, -1]])
Ib = np.array([-2,  0, -5]).reshape(-1, 1)

polyInp = pc.to_V(np.hstack((Ib, -IA)))
print(polyInp.point, polyInp.ray)
poly_set = [polyInp]

# compute the output polytope.
polyOut = fp.whole_computation(layer, K, W, U, B, poly_set)
print(len(polyOut), polyOut[0].point)
for poly in polyOut:
    print(poly.point)
"""
poly_out = []
for p in polyOut:
    poly = pcon.Poly(p.point[:,-2:], [])
    poly_out.append(poly)
"""


# Define Property polytope of the labeled sample
PA = np.array([[-1, 0, 1, 0, 0, 0], [0, 0, -1, 0, 1, 0],[-1,0,0,0,0,0],[0,-1,0,0,0,0],[0,0,-1,0,0,0],[0,0,0,-1,0,0],[0,0,0,0,-1,0],[0,0,0,0,0,-1]])
Pb = np.array([0, 0,10,10,10,10,10,10]).reshape(-1, 1)
polyPro = pc.to_V(np.hstack((Pb, -PA)))
print(polyPro.point, polyPro.ray)

quality_result = qv.qualitive_verify(polyOut, polyPro)
print(quality_result)

for poly in polyOut:
    if (np.shape(poly.point)[0] <= 1):
        pass
    else:
        quantatitive_result = qv.quantatitive_verify_mc_nobp(poly, polyPro, 10000)
        print(quantatitive_result)