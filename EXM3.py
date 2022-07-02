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


IA = np.array([[1, -1, 0], [0, 1, -1],[-1, 0, 0],[0, 0, 1]])
Ib = np.array([0, 0, 1, 1]).reshape(-1, 1)


polyInp = pc.to_V(np.hstack((Ib, -IA)))
print(polyInp.point, polyInp.ray)
poly_set = [polyInp]

# compute the output polytope.
polyOut = fp.whole_computation(layer, K, W, U, B, poly_set)
print(len(polyOut), polyOut[0].point)

poly_out = []
for p in polyOut:
    poly = pcon.Poly(p.point[:,-2:], [])
    poly_out.append(poly)



# Define Property polytope of the labeled sample
label = 0
num = 2
A1 = - np.ones(num-1).reshape(-1,1)
A2 =  np.identity(num-1)
A = np.hstack((A2[:,0:label],A1,A2[:,label:]))
A_label = np.hstack((np.hstack((np.zeros(label),np.array([-1]))), np.zeros(num-label-1)))
A = np.vstack((A,A_label))
b = np.append(np.zeros(num-1),np.array(1)).reshape(-1,1)
polyPro = pcon.to_V(np.hstack((b, -A)))
#print(polyPro.point, polyPro.ray)

quality_result = qv.qualitive_verify(poly_out, polyPro)
print(quality_result)


polyPro = [[-2,-2], [-2,2], [2,2]]
quantatitive_result = qv.quantatitive_verify_vr(poly_out[0].point, polyPro)
print(quantatitive_result)