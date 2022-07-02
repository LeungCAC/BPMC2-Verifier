import numpy as np
from numpy.lib.polynomial import polyint
import cdd

#define Polytope Class composed of points and rays
class Poly:
    def __init__(self, point, ray):
        self.point = point
        self.ray = ray


#function to decompose the points and rays of a V-representation polytope.
def point_ray (poly):
    p = np.array(poly[:])
    point = [ ]
    ray = [ ]
    for s in p :
        if (s[0] == 1) :
            point.append(s[1:])
        else:
            ray.append(s[1:])
    if (len(point) == 0):
        pass
    else:
        point = np.unique(point, axis=0)
    if (len(ray) == 0):
        pass
    else:
        ray = np.unique(ray, axis=0)
    p = Poly(np.array(point), np.array(ray))
    return p
#print(point_ray(ext),point_ray(ext).point,point_ray(ext).ray)


#function to resturct a V-representation polytope, given points and rays.
def poray_poly (poly):
    idp = np.ones (np.shape(poly.point)[0])
    idr = np.zeros (np.shape(poly.ray)[0])
    id = np.hstack ((idp,idr)).reshape(-1,1)
    
    if (np.shape(poly.point)[0]==0):
        p = poly.ray
    elif (np.shape(poly.ray)[0]==0):
        p = poly.point
    else:
        p = np.vstack ((poly.point, poly.ray))

    # points and rays labeled with 1 and 0 repsepctively
    vp = np.hstack((id,p))
    vp = np.unique(vp, axis=0)
    return vp

"""
p = np.array([[1,0,1],[1,0,1],[3,4,5]])
r = np.array([[1,1,1],[2,1,2],[1,0,1]])
print(poray_poly(Poly(p,r)))
print(point_ray(poray_poly(Poly(p,r))).point, point_ray(poray_poly(Poly(p,r))).ray )
"""

# The below 2 functions are used to convert polytopes in different representation.

def to_H (poly):
    M = poray_poly(poly)
    mat = cdd.Matrix(M,number_type='float')
    mat.rep_type = cdd.RepType.GENERATOR
    poly = cdd.Polyhedron(mat)
    ext = poly.get_inequalities()
    #ext.canonicalize()
    ext = np.array(ext [:])
    p_b = ext[ :,0].reshape(-1,1)
    p_A = - ext[:,1:]
    return p_b, p_A       #return b, A

"""
p = np.array([[1,0,1],[2,1,2],[3,4,5]])
r = np.array([[1,1,1],[2,1,2],[2,1,2]])
poly = Poly(p,r)
print(to_H(poly))

p = np.array([[0,0],[1,1],[1,0]])
r = np.array([[1,0]])
poly = Poly(p,r)
print(to_H(poly))
"""

def to_V (H_matrix):
    M = H_matrix
    mat = cdd.Matrix(M,number_type='float')
    mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat)
    ext = poly.get_generators()
    if(np.shape(ext)[0] == 0):
        pass
    #else:
        #ext.canonicalize()
    p = point_ray(ext)
    return p
    
"""
p = np.array([[1,0],[2,1],[3,5]])
r = np.array([[1,1],[2,1],[2,2]])
poly = Poly(p,r)
b, A = to_H(poly)
p = to_V(np.hstack((b, -A)))   #记得是-A
print(p.point, p.ray)

p = np.array([[0,0],[1,1],[1,0]])
r = np.array([[1,0]])
poly = Poly(p,r)
b, A = to_H(poly)
p = to_V(np.hstack((b, -A)))   #记得是-A
print(p.point, p.ray)
"""