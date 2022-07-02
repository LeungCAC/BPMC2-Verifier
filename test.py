import numpy as np
import poly_convertion as pc
import numpy as np
import forward_propagation as fp



# Get the parameters layers, W, U and b of a trained RNN
# If the U cannot get from the no-rnn layer, make 0 instead


layer = 3
w0 = [[-0.1,0.2],[-0.3,0.4],[-0.5,-0.6]]
w1 = [[0.1,-0.2,0.1],[-0.3,0.4,-0.3],[0.5,-0.6,0.5]]
w2 = [[0.1,-0.2,0.1],[-0.3,0.4,-0.3],[0.5,-0.6,0.5],[0.1,-0.2,0.1]]

u0 = [[0.4,-0.2,-0.2],[-0.1,-0.2,-0.3],[0.3,-0.4,-0.5]] 
u1 = [[0.4,-0.2,-0.2],[-0.1,-0.2,-0.3],[0.3,-0.4,-0.5]]
u2 = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]

b0 = [0.1, 0.2, -0.3] 
b1 = [-0.2, -0.1, 0.1]
b2 = [-0.1, -0.1, 0.2, -0.1]

W = [w0, w1, w2]
U = [u0, u1, u2]
B = [b0, b1, b2]

size = 6
Ib_1 = np.array([0.1,0.2,0.1,0.1,0.1,0.1]).reshape(-1,1)
Ib_2 = np.array([0,0.1,0.1,0.1,0.1,0.1]).reshape(-1,1)
IA_1 = np.identity(size )
IA_2 = -1 * np.identity(size)
#print(Ib_1, Ib_2, Ib_3)


Ib = np.vstack((Ib_1, -Ib_2))
IA = np.vstack((IA_1, IA_2))


polyInp = pc.to_V(np.hstack((Ib, -IA)))
print(polyInp, polyInp.point, polyInp.ray)
K = 3
polyout = fp.whole_computation(layer, K, W, U, B, [polyInp])

print(len(polyout))