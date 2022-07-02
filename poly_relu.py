from asyncio.selector_events import BaseSelectorEventLoop
from asyncio.windows_events import NULL
from sunau import Au_read
from sys import base_prefix
import numpy as np
import cdd
from scipy import optimize as op
from scipy.spatial import ConvexHull
from sympy import hermite_poly
import poly_convertion as pcon
import poly_relative as pr
import random

def forward_relu(poly, d):
    p_point = poly.point
    p_ray = poly.ray

    Ap = [] # the point set of the above part
    Ar =[]  # the ray set of the above part
    Bp = [] # the point set of the below part
    Br =[]  # the ray set of the below part
    
    Aflag = 0
    Bflag = 0
    Iflag = 0

    interp = [] # the intersection point between the hyperplane and the polytope
    for v1 in range(p_point.shape[0]):
        for v2 in range(v1+1, p_point.shape[0]):
            if (p_point[v2][d] == p_point[v1][d]):
                if(p_point[v2][d] == 0 ):
                    interp.append(v1)
                    interp.append(v2)
            else:
                c = p_point[v2][d] / (p_point[v2][d] - p_point[v1][d])
                if( c >= 0 and c <= 1 ):
                    ip = c * p_point[v1] + (1-c) * p_point[v2]
                    interp.append(ip)
                    Iflag = 1
    
    for v in p_point:
        for r in p_ray:
            if(r[d] == 0):
                pass
            else:
                c = -v[d] / r[d]
                if ( c >= 0 ):
                    ip = v + c * r
                    interp.append(ip)
                    Iflag = 1
   
    for v in p_point:
        if ( v[d] > 0 ):
            Ap.append(v)
            Aflag = 1
        elif( v[d] < 0 ):
            relu_v = np.hstack((v[0:d], np.hstack((np.maximum(v[d:d+1], 0), v[d+1:]))))
            Bp.append(relu_v)
            Bflag = 1
    
    if (Iflag == 1):
        if(Aflag == 1):
            Ap = np.vstack((Ap, interp))
        else:
            Ap = interp
        if(Bflag == 1):
            Bp = np.vstack((Bp, interp))
        else:
            Bp = interp
    
    Ap = np.unique(Ap, axis=0)
    Bp = np.unique(Bp, axis=0)  

    for r in p_ray:
        relu_r = np.hstack((r[0:d], np.hstack((np.maximum(r[d:d+1], 0), r[d+1:]))))
        row, _ = p_ray.shape

        # decide whether relu_r is a feasible direction of the origin polytope.
        g = np.zeros((1,row))
        Au = - np.identity(row)
        bu = np.zeros((1,row))
        Ae = p_ray.T
        be = relu_r

        res = op.linprog(g, Au, bu, Ae, be)
        if (res.success == True):
            Ar.append(relu_r)
        else:
            Br.append(relu_r)
    
    Ar = np.unique(Ar, axis=0)
    Br = np.unique(Br, axis=0)

    """
    (rowA, colA) = Ap.shape
    (rowB, colB) = Bp.shape

    if(colA >= 2 and rowA >= colA + 1):
        Ap = pr.convex_hull(Ap)
    if(colB >= 2 and rowB >= colB + 1):
        Bp = pr.convex_hull(Bp)
    """

    Apoly = pcon.Poly(Ap, Ar)
    Bpoly = pcon.Poly(Bp, Br)

    return Apoly, Bpoly


"""
p = np.array([[2,-1]])
r = np.array([[-1,1],[0,1]])
poly = pcon.Poly(p,r)
Hp, Lp = forward_relu(poly, 0)
for p in [Hp,Lp]:
    [H, L] = forward_relu(p, 1)
    print("Example 1:",H.point, H.ray)
    print("Example 1:",L.point, L.ray)
"""

"""
p = np.array([[1,2]])
r = np.array([[-1,-1],[-1,1]])
poly = pcon.Poly(p,r)
Hp, Lp = forward_relu(poly, 0)
for p in [Hp,Lp]:
    [H, L] = forward_relu(p, 1)
    print("Example 1:",H.point, H.ray)
    print("Example 1:",L.point, L.ray)
"""

"""
p = np.array([[-1,1]])
r = np.array([[1,1],[0,1]])
poly = pcon.Poly(p,r)
Hp, Lp = forward_relu(poly, 0)
for p in [Hp,Lp]:
    [H, L] = forward_relu(p, 1)
    print("Example 1:",H.point, H.ray)
    print("Example 1:",L.point, L.ray)

"""

"""
p = np.array([[0,1],[2,-1],[3,-1],[3,1]])
r = np.array([[0,1]])
poly = pcon.Poly(p,r)
Hp, Lp = forward_relu(poly, 0)
for p in [Hp,Lp]:
    [H, L] = forward_relu(p, 1)
    print("Example 1:",H.point, H.ray)
    print("Example 1:",L.point, L.ray)
"""

def set_forward_relu(poly_set, d):
    ps = []
    for poly in poly_set:
        Apoly, Bpoly = forward_relu(poly, d)
        if (list(Apoly.point)):
            ps.append(Apoly)
        if (list(Bpoly.point)):
            ps.append(Bpoly)
    return ps

def whole_relu (poly_set, order, size):
    for d in range(order*size, (order+1)*size):
        ps = set_forward_relu(poly_set, d)
        poly_set = ps
    return poly_set

"""        
p = np.array([[0,1],[2,-1],[3,-1],[3,1]])
r = np.array([[0,1]])
poly = pcon.Poly(p,r)
poly_set = whole_relu([poly],0,2)
for p in poly_set:
    print(p, p.point, p.ray)
"""


def backward_relu(point, dim):
    br = []
    if (point[dim] > 0):
        br.append(point)
    elif(point[dim] == 0):
        for i in range(100):
            s = random.uniform(-50, 0)
            point[dim] = s
            br.append(point)
    else:
        br = []
    return br

"""
point = [1,0,-1]
print(backward_relu(point, 2))
"""

def set_backward_relu (point_set, dim):
    bp_set = []
    for point in point_set:
        br = backward_relu(point, dim)
        bp_set = bp_set + br
    return bp_set


def whole_backward_relu ( point_set, order, size ):
    for dim in range( (order+1)*size-1,  order*size-1, -1):
        bp_set = set_backward_relu( point_set, dim )
        point_set = bp_set
    return point_set
        
    

            