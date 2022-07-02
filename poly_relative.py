import numpy as np
import cdd
from scipy import optimize as op
from scipy.spatial import ConvexHull
import poly_convertion as pcon
from itertools import permutations, combinations
import sympy as sy
import polytope as pc


epsilon = 1e-5 # please adjust this for finding dimensions

# Example of H-representation--->V-representation, Polytope X
mat = cdd.Matrix([[-2,1,1],[0,0,1],[0,1,0]], number_type='float')
mat.rep_type = cdd.RepType.INEQUALITY
mat.canonicalize() # y>0的约束是多余的，所以会被排除掉
poly = cdd.Polyhedron(mat)
ext = poly.get_generators()
#print(ext)

# Another example of H-representation--->V-representation, Polytope Y
mat1 = cdd.Matrix([[-3,1,1],[0,0,1],[0,1,0]], number_type='float')
mat1.rep_type = cdd.RepType.INEQUALITY
mat1.canonicalize() # y>0的约束是多余的，所以会被排除掉
poly1 = cdd.Polyhedron(mat1)
ext1 = poly1.get_generators()
#print(ext1)


#Judge whether a point belongs to a given polytope region.
def belong_judge (p, poly):
    size = (np.shape(poly.point)[0])
    size2 = (np.shape(poly.ray)[0])
    c = np.ones(size + size2)   # 目标函数系数
    A_ub = - np.identity(size + size2) # 不等式约束系数A
    B_ub = np.array(np.zeros( size + size2 )).reshape((-1,1))  # 等式约束系数b
    e1 = np.append (np.ones(size), np.zeros(size2))
    flag = 1
    if (np.shape(poly.point)[0] == 0):
        e2 = poly.ray.T
        flag = 0
    elif (np.shape(poly.ray)[0] == 0):
        e2 = poly.point.T
    else:
        e2 = np.concatenate ((poly.point, poly.ray), axis = 0).T
    A_eq = np.vstack((e2, e1)) # 等式约束系数Aeq
    B_eq = np.append(np.array(p), [flag]).reshape((-1,1))   # 等式约束系数beq
    res = op.linprog(c, A_ub, B_ub, A_eq, B_eq)
    #print(res)
    if (res.success==True):
        return True
    else:
        #print(p)
        return False

"""
p = [1.45,1.5]
poly = pcon.point_ray(ext)
print(belong_judge(p,poly))

p = [3,3]
poly = pcon.point_ray(ext1)
print(belong_judge(p,poly))
"""

#Judge whether a polytope includes another.
def include_judge (polyA, polyB):
    if(np.shape(polyA.point)[0] == 0):
        pass
    else:
        for s in polyA.point:
            if(belong_judge(s, polyB) == False):
                return False
    if (np.shape(polyA.ray)[0] == 0):
        pass
    else:
        rank = np.linalg.matrix_rank(polyB.ray)
        for r in polyA.ray:
            m = np.vstack((polyB.ray, r))
            if(np.linalg.matrix_rank(m)>rank):
                return False
    return True
    
"""
polyA = pcon.point_ray(ext)
polyB = pcon.point_ray(ext1)
print(include_judge(polyA, polyB))
print(include_judge(polyB, polyA))
"""

# Judege whether 2 polytopes with different representations refers to the same region.
def equal_judge (polyA, polyB):
    if(include_judge(polyA, polyB) and include_judge(polyB, polyA)):
        return True
    else:
        return False

"""
polyA = pcon.point_ray(ext)
polyB = pcon.point_ray(ext1)
print(equal_judge(polyA, polyB))
"""


def toHullIdxScipy(low_dim_pts_reduct):
    if low_dim_pts_reduct.shape[1] == 1:
        lb_idx = np.argmin(low_dim_pts_reduct)
        ub_idx = np.argmax(low_dim_pts_reduct)
        return [lb_idx, ub_idx]
    else:
        # convex hull will get you Ax + b <= 0
        try:
            hull = ConvexHull(points=low_dim_pts_reduct)
        except:
            print ('W: Qhull failed, turn on QJ (Joggle) option')
            hull = ConvexHull(points=low_dim_pts_reduct, qhull_options = 'QJ')
        return hull

# Todo: find convexhull of vertices of polytopes to remove rebundant points.
def convex_hull (point):
    _, ndim = point.shape
    dim = np.linalg.matrix_rank(point-point[0])

    # If it is a full-dimension polytope
    if (ndim == dim):
        hull = toHullIdxScipy(point)
        hull_point = []
        for idx in hull.vertices:
            hull_point.append(hull.points[idx])
        return hull_point
    # Else, it is a degenerated polytope
    else:
        U,S,D = np.linalg.svd(point)

        npoints, ndim = point.shape
        # epsilon as a threshold to control the float error.
        if not (np.abs(S) < epsilon).any():
            dim_to_remove = S.shape[0]
        else:
            dim_to_remove = np.argmax(np.abs(S) < epsilon)
        
        diag_l = S.shape[0]
        reconstruct_diag = np.zeros((npoints, ndim))
        reconstruct_diag[:diag_l,:diag_l] = np.diag(S)

        new_pts = np.matmul(U,reconstruct_diag)
        rec_point = np.matmul(new_pts,D)

        low_dim_pts_reduct = new_pts[:, :dim_to_remove]
        hull = toHullIdxScipy(low_dim_pts_reduct)

        hull_point = []
        for idx in hull.vertices:
            hull_point.append(rec_point[idx])
        return hull_point


"""
#p = np.array([[1,1,0,1],[1,0,0,1],[0,1,0,1],[0.5,0.5,0,1]])
p = np.array([[1,1,0],[1,0,0],[0,1,0],[0.5,0.5,0]])
points = convex_hull(p)
print(points)
"""

#Judge its dimension in the space.
def dimension_judge(poly):
    point = np.array( np.unique(poly.point, axis = 0))
    dim = np.linalg.matrix_rank(point-point[0])
    return dim


#Choice the polytopes whose dimension is in the set interval.
def poly_choice(poly_set, k):
    dim_set = []
    for poly in poly_set:
        dim_set.append(dimension_judge(poly))

    mdim = np.max(dim_set)

    npoly_set = []
    for i in range(len(dim_set)):
        if (dim_set[i] >= mdim - k):
            npoly_set.append(poly_set[i])

    return npoly_set


"""
poly_set = []
p = np.array([[1,1,0,1],[1,0,0,1],[0,1,0,1],[0.5,0.5,0,1]])
r = np.array([])
poly = pcon.Poly(p,r)
for i in range(5):
    poly_set.append(poly)
npoly_set = poly_choice(poly_set, 0)
print(npoly_set)
"""


#The function of volume of bounded polytope. This method does not give an exact value, more like simulation.
def conpoly_volume_V (poly):
    b,A = pcon.to_H(poly)
    #print (b, A)
    po = pc.Polytope(A, b)
    vol = po.volume
    return vol

def conpoly_volume_H (A, b):
    po = pc.Polytope(A, b)
    vol = po.volume
    return vol

"""
p = np.array([[2,0],[0,2],[3,0],[1,2]])
r = np.array([])
poly = pcon.Poly(p,r)
print(conpoly_volume(poly))
"""

#The function of volume of bounded polytope. This method gives an exact value, recommend!
def convexhull_volume (point):
    return ConvexHull(point).volume

"""
p = np.array([[2,0],[0,2],[3,0],[1,2]])
volume = convexhull_volume(p)
print (volume)
"""

def normal_intersect (polyA, polyB):
    pA_b, pA_A = pcon.to_H(polyA)
    pB_b, pB_A = pcon.to_H(polyB)
    b = np.vstack((pA_b, pB_b))
    A = np.vstack((-pA_A, -pB_A))
    p = pcon.to_V(np.hstack((b,A)))
    return p



#The function of volume of unbounded polytope
def nonconpoly_volume(poly, m, n):
    #print (np.shape(poly.point)[1])
    size = np.shape(poly.point)[1]
    m_A = np.identity(size)
    m_b = m * np.ones(size).reshape(-1,1)
    pb = pcon.to_V(np.hstack((m_b, -m_A)))
    p = normal_intersect(poly, pb)
    #计算凸包顶点
    p_index = ConvexHull(p.point).vertices
    #print(p.point, p_index)
    point = [ ]
    for i in p_index:
        point.append(p.point[i])
    
    n = n
    point = sy.Matrix(point)
    for i in range(sy.shape(point)[0]):
        for j in range(sy.shape(point)[1]):
            if (point[i,j] == m):
                point[i,j] = n
            else:
                pass
    point = np.array(point)
    tol_num = np.shape(point)[0]
    #print(point)

    f_point = np.array([point[0]])
    F = np.repeat(f_point,size,0)
    #print("f_point",f_point)
    l_point = point[1: ,:]
    #print(f_point, l_point)

    choice_point = combinations(l_point, size)
    leaving_point = combinations(l_point, tol_num-size-1 )
    choice_point = np.array(list(choice_point))
    leaving_point = np.array(list(reversed(list(leaving_point))))
    volume = 0
    global M
    for i in range(0,np.shape(choice_point)[0]):
        flag = True
        M = sy.Matrix(choice_point[i].tolist())
        #print("chioce and leaving" ,choice_point[i], leaving_point[i])
        for p in leaving_point[i]:
            r = np.vstack((p,choice_point[i]))
            l = np.ones(size+1).reshape(-1,1)
            W = sy.Matrix(np.hstack((l,r)).tolist())
            #print(W)
            #print(M)
            expr = W.det() * (M-F).det()
            if (expr.subs(n,10)<0):
                flag = False
                break

        if(flag == True):
            p_volume = 1/sy.factorial(size) * sy.Abs((M-F).det())
            volume = sy.Add(volume,p_volume)
            #print(i, choice_point[i], p_volume)
    return volume

"""
n = sy.Symbol('n')
p = np.array([[2,0],[0,2],[3,0],[1,2]])
r = np.array([])
poly = pcon.Poly(p,r)
b = nonconpoly_volume(poly, 100000, n)
print(b)
"""

"""
m = sy.Symbol('m')
p = np.array([[2,0],[0,2],[2,4]])
r = np.array([[0,1],[1,0]])
poly = pcon.Poly(p,r)

b = nonconpoly_volume(poly, 100000, m)
print(b)
"""