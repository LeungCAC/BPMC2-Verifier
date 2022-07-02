import numpy as np

"""
i：本层的第i次运算，从0开始，至K-1结束
W矩阵：前一层（m维）和该层（n维）的连接矩阵，输入应该为（n*m）的形状
U矩阵：本层（n维）的连接矩阵，输入应该为（n*n）的形状
K：输入序列的长度
"""

def con_matrix (i, K, W, U, bs):
    if ( i == 0 ):
        W = W
        b = bs
        n = np.shape(W)[0]
        m = np.shape(W)[1]
        S = np.zeros((np.shape(W)[0],(K-1)*m))
        C = np.hstack((W,S))

        R = np.identity((K-1)*m)
        L = np.zeros(((K-1)*m, np.shape(W)[1]))
        F = np.hstack((L,R))
        M = np.vstack((C, F))

        bd = np.zeros((K-i-1)*m)
        bias = np.hstack((b, bd)).reshape(-1,1)

        return M, bias

    else:
        W = W
        U = U
        b = bs
        n = np.shape(W)[0]
        m = np.shape(W)[1]
        PL = np.identity(i*n)
        PR = np.zeros((i*n, (K-i)*m))
        P = np.hstack((PL,PR))
        #print(P,np.shape(P))

        CL = np.zeros ((n, (i-1)*n))
        CR = np.zeros((n, (K-i-1)*m))
        C = np.hstack((CL,U,W,CR))
        #print(C,np.shape(C))
        #print(m, (K-i-1))
        FL = np.zeros(((K-i-1)*m, i*n+np.shape(W)[1]))
        FR = np.identity((K-i-1)*m)
        #print(np.shape(FL),np.shape(FR))
        F = np.hstack((FL,FR))
        #print(F,np.shape(F))

        M = np.vstack((P,C,F))

        bu = np.zeros(i*n)
        bd = np.zeros((K-i-1)*m)
        bias = np.hstack((bu, bs, bd)).reshape(-1,1)
        return M, bias

"""
K = 4
W = np.array([[1,2,3,4],[2,1,1,2],[1,2,4,5]])
U = np.array([[10,20,30],[30,40,50],[40,50,60]])
bs = np.array([100,200,300])

for i in range(0, K):
    M,b = con_matrix (i, K, W, U, bs)
    print(np.shape(M), np.shape(b))
    #print(M, b)
"""




