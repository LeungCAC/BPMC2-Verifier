import forward_propagation as fp
import cdd
import numpy as np
import poly_convertion as pc
import qq_verify as qv
import poly_convertion as pcon
import backward_propagation as bp
import fractions
#np.set_printoptions(formatter={'all':lambda x:str(fractions.Fraction(x).limit_denominator())})

# Get the parameters layers, W, U and b of a trained RNN
# If the U cannot get from the no-rnn layer, make 0 instead

layer = 3
K = 3
m = 3

"""
w0 =  np.random.randn(3,2)
w0 = np.around(w0, decimals = m)
w1 =  np.random.randn(3,3)
w1 = np.around(w1, decimals = m)
w2 =  np.random.randn(2,3)
w2 = np.around(w2, decimals = m)

u0 =  np.random.randn(3,3)
u0 = np.around(u0, decimals = m)
u1 =  np.random.randn(3,3)
u1 = np.around(u1, decimals = m)
u2 =  np.random.randn(2,2)
u2 = np.around(u2, decimals = m)

b0 = np.random.randn(3)
b0 = np.around(b0, decimals = m)
b1 = np.random.randn(3) 
b1 = np.around(b1, decimals = m)
b2 = np.random.randn(2)
b2 = np.around(b2, decimals = m) 

W = [w0, w1, w2]
U = [u0, u1, u2]
B = [b0, b1, b2]
"""

w0 = np.array([[ 0.7026 , -1.58164],
       [-0.16227,  0.23622],
       [ 0.90164, -0.54237]])

w1 = np.array([[-0.0373 ,  0.0247 , -0.63885],
       [-0.02889, 0.32743,  -1.51715],
       [-0.3848 , -0.1482 , -0.82503]])

w2 = np.array([[ 0.17581, 0.38454, -1.04902],
       [ 1.36189, -0.87928,  1.17416]])

u0 = np.array([[ -0.6802 ,  0.23458,  0.24678],
       [ 0.87543,  1.42248, -0.83356],
       [ 1.09755,  0.07983, -1.09887]])

u1 = np.array([[ -0.78253,  0.45749,  0.9418 ],
       [ 0.61339,  0.75726,  0.16477],
       [-0.8502 , 0.03387, 0.35951]])

u2 = np.array([[0,0],[0,0]])

b0 = [-0.20968,  0.78769, -1.42158]

b1 = [ 0.67246,  0.25835, -0.28039]

b2 = [-0.14129,  0.43276]

W = [w0, w1, w2]
U = [u0, u1, u2]
B = [b0, b1, b2]


def relu(x):
    return np.maximum(0, x)

#input = [[ 1.67,  1.36],[ 0.9  , 0.5 ],[-0.71 ,-1.35]] 
input = [[-0.36 , 0.5 ],[-0.71 , 1.38],[ 1.79 ,-2.33]] 
# the output of the network defined by the parameters

def net(input, K):
    out = {}
    for k in range(0, K): 
        hid = np.array(input[k])  
        if( k == 0 ): 
            for l in range(0, layer):
                if(l < layer-1):
                    hid = np.matmul(W[l],hid)
                    hid = relu(hid + B[l])
                    out[l] = hid
                else:
                    hid = np.matmul(W[l],hid)
                    hid = hid + B[l]
                    out[l] = hid


        else:
            for l in range(0, layer):
                if(l < layer-1):
                    hid = np.matmul(W[l],hid) + np.matmul(U[l],out[l])
                    hid = relu(hid + B[l])
                    out[l] = hid
                else:
                    hid = np.matmul(W[l],hid) + np.matmul(U[l],out[l])
                    hid = hid + B[l]
                    out[l] = hid
    return hid
print(input,net(input, 3))

size = 6

Ib_1 = np.array([ -0.36 , 0.5 ,-0.71 , 1.38 ,1.79 ,-2.33]).reshape(-1,1)
Ib_2 = np.array([-0.36 , 0.5 ,-0.71 , 1.38, 1.79, -2.39]).reshape(-1,1)

#Ib_1 = np.array([1.67,  1.36, 0.9 , 0.5 ,-0.71 ,-1.35]).reshape(-1,1)
#Ib_2 = np.array([1.67,  1.36, 0.8 , 0.4 ,-0.71 ,-1.35]).reshape(-1,1)
IA_1 = np.identity(size )
IA_2 = -1 * np.identity(size)
#print(Ib_1, Ib_2, Ib_3)

# define the input polytope.
Ib = np.vstack((Ib_1, -Ib_2))
IA = np.vstack((IA_1, IA_2))
print(np.shape(Ib), np.shape(IA))
polyInp = pc.to_V(np.hstack((Ib, -IA)))
poly_set = [polyInp]

# compute the output polytope.
polyOut = fp.whole_computation(layer, K, W, U, B, poly_set)

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

poly_out = []
for p in polyOut:
    print(p.point)
    poly = pcon.Poly(p.point[:,-2:], [])
    poly_out.append(poly)


quality_result = qv.qualitive_verify(poly_out, polyPro)
print(quality_result)

#If false, then sample from the input region to find the adversarial sample
flag = True
while( flag == True ):
    s = bp.mc_sample(polyInp)
    #print(s)
    output = net(s.reshape(3,2),3)
    if(np.abs(output[0]-output[1]) <= 0.1):
        print(s, output)
        flag = False
