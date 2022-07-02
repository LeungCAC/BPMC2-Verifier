import numpy as np
import itertools
from numpy.testing._private.utils import HAS_LAPACK64
import scipy

def initial ( pconvex ):
    dim = np.shape(pconvex)[1]
    normal = - np.identity(dim)
    inter_normal = np.ones(dim)
    s_normal = np.ones(dim)/np.sqrt(dim)
    out_normal = np.vstack((normal, s_normal))
    #print(out_normal)
    
    o = np.array(pconvex, dtype=float).min(0)
    v = np.tile(o, (dim+1, 1))
    #print(v)

    d = 0
    for i in pconvex:
        if (np.dot(inter_normal, i) > d):
            d = np.dot(inter_normal, i)
    d = d - np.dot(inter_normal, o)
    #print (d)

    for i in range(dim):
        v[i][i] += d
    v = np.flip(v, axis = 0)
    #print (v)

    n_list = np.empty([dim+1, dim])   #average normal
    h_list = list(itertools.combinations(out_normal, dim))
    for i in range(len(h_list)):
        n = np.sum(h_list[i], axis = 0)/dim
        n = 1 / np.linalg.norm(n) * n
        n_list[i] = n
    
    n_num = np.shape(n_list)[0]
    p_num = np.shape(pconvex)[0]
    hdist = np.zeros([n_num, p_num])
    for i in range(n_num):
        for j in range(p_num):
            hdist[i][j] = np.dot(n_list[i], pconvex[j])
    #print("hdist", hdist)
    hs = np.array(hdist).max(1).reshape(-1,1)
    #print("hs",hs)

    o_num = np.shape(out_normal)[0]
    ndist = np.zeros([o_num, p_num])
    for i in range(o_num):
        for j in range(p_num):
            ndist[i][j] = np.dot(out_normal[i], pconvex[j])
    #print("hdist", hdist)
    ns = np.array(ndist).max(1).reshape(-1,1)
    #print("hs",hs)
    hs_list = list(itertools.combinations(ns, dim))

    #print("c1",v)
    #print("c2",n_list)
    #print("c3",np.array(h_list))
    #print("c4", np.array(hs_list))
    #print("c5", out_normal)
    
    #v: the vertex
    #n_list: average unit normal  
    #out_normal: unit outward normal of hyperplane, must be unit
    #h_list, hs_list: the hyperplane and interepts related to every vertex
    return v, out_normal, h_list, hs_list, n_list, hs, ns

"""
p = [[1,1], [2,2], [3,3],[2,1]]
initial(p)

"""


def choice_hyperplane(v, n_list, hs):
    print("choice_hyperplane")
    n_num = len(n_list)
    v_num = np.shape(v)[0]
    #print(p_num, n_num, v_num)

    dist = np.zeros([n_num, v_num])
    for i in range(n_num):
        for j in range(v_num):
            dist[i][j] = np.dot(n_list[i], v[j])
    #print("dist", dist)

    #print("hs",hs)
    dist = (dist - hs).T
    #print("dddd", dist)

    max_min = np.max(dist)
    #print(max_min)
    pos = np.where (dist == max_min)
    #print(pos)
    choice_v = pos[0][0]
    d = max_min
    n = n_list[choice_v]
    #print(d, n)
    return d, n
"""
p = [[1,1], [2,2], [3,3],[2,1]]
v, out_normal, h_list, hs_list, n_list, hs = initial(p)
choice_hyperplane(v, n_list, hs)
"""

def cut_hyperplane(v, h_list, hs_list, n_list, n, pconvex, out_normal, ns):
    print("cut hyperplane")
    dim = np.shape(v)[1] #space dimension
    Ns = np.dot(n, pconvex[0])
    for p in pconvex:
        if( np.dot(n, p) > Ns):
            Ns = np.dot(n, p)
    
    out_normal = np.vstack((out_normal, n))  #all hyperplane
    ns = np.hstack((ns.flatten(),[Ns]))   #all intercept
    
    #divide the vertex into H-, H and H+, and store the hyperplane related
    vn = []
    vp = []
    hn = []
    hp = []
    hsn = []
    hsp = []

    for s in range(np.shape(v)[0]):
        if(np.dot(v[s], n) <= Ns):
            vn.append(v[s])
            hn.append(h_list[s])
            hsn.append(hs_list[s])
        else:
            vp.append(v[s])
            hp.append(h_list[s])
            hsp.append(hs_list[s])

    #compute the new vertex
    va = []
    ha = []
    hsa = []
    add_flag = False
    for h in range(len(hp)):
        Hp_list = list(itertools.combinations(hp[h], dim-1))
        Hsp_list = list(itertools.combinations(hsp[h], dim-1))

        for s in range(np.shape(Hp_list)[0]):
            inter = np.array(Hp_list[s]).flatten().reshape(dim-1, dim)
            a = np.vstack((inter, n))
            b = np.hstack((np.array(Hsp_list[s]).flatten(), [Ns]))
            if (np.linalg.matrix_rank(np.mat(a)) < dim):
                pass
            else:
                so = np.linalg.solve(a, b)
                flag = True
                for i in range(np.shape(out_normal)[0]):
                    #print(np.dot(so, out_normal[i]), ns[i])
                    if(np.dot(so, out_normal[i]) > ns[i]  and np.dot(so, out_normal[i]) - ns[i] >= 0.0000000001):
                        flag = False
                        pass
                so = np.around(so, decimals = 10)
                nm = np.array(np.vstack((v,so)))
                nv = np.unique(nm, axis = 0)
                if (np.shape(nv)[0] == np.shape(v)[0] + 1 and flag == True):
                    va.append(so)
                    ha.append(list(a))
                    hsa.append(list(b))
                    add_flag = True

    #sum up and ready for next loop
    if(add_flag == True):
        v = np.vstack((vn, va))
        h_list = np.concatenate((hn, ha), axis = 0)

        hsn = np.array(hsn).reshape(-1, dim)
        hs_list = np.concatenate((hsn, hsa), axis = 0)
    else:
        v = vn
        h_list = hn
        hs_list = hsn

    n_list = []
    for i in range(len(h_list)):
        n = np.sum(h_list[i], axis = 0)/dim
        n = 1 / np.linalg.norm(n) * n
        n_list.append(n)
    
    n_num = np.shape(n_list)[0]
    p_num = np.shape(pconvex)[0]
    hdist = np.zeros([n_num, p_num])
    for i in range(n_num):
        for j in range(p_num):
            hdist[i][j] = np.dot(n_list[i], pconvex[j])
    #print("hdist", hdist)
    hs = np.array(hdist).max(1).reshape(-1,1)

    #print(v, h_list, hs_list, n_list, out_normal, hs)
    return v, h_list, hs_list, n_list, out_normal, hs, ns

"""
p = [[1,1], [2,2], [3,3], [2,1], [1,2]]
v, out_normal, h_list, hs_list, n_list, hs = initial(p)
d, n = choice_hyperplane(v, n_list, hs)
cut_hyperplane(v, h_list, hs_list, n_list, n, hs, p, out_normal)
"""


def reduce_vertex(pconvex, m):
    v, out_normal, h_list, hs_list, n_list, hs, ns = initial(pconvex)
    while (np.shape(v)[0] < m):
        print("reducing", np.shape(v))
        d, n = choice_hyperplane(v, n_list, hs)
        #print(d,n)
        if (d <= 0.00001):
            print("No possible vertex to be reduced!")
            return v
        v, h_list, hs_list, n_list, out_normal, hs, ns = cut_hyperplane(v, h_list, hs_list, n_list, n, pconvex, out_normal, ns)

    return v

"""
pconvex = [[1,2], [1.5,2], [4,3], [3,2], [3,2.2]]
m = 10000
v = reduce_vertex(pconvex, m)
print(v)
"""
