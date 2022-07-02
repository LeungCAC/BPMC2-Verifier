from sympy import Line, true
import network_computation as Nc
import poly_linear as pl
import poly_relu as pr
import numpy as np
import poly_relative as pre
import poly_convertion as pcon
import backward_propagation as bpg
from shapely.geometry import Polygon
from shapely.geometry import LineString

def quantatitive_verify_mc(polyOut, counts, polyInp, polyPro, layer, K, W, U,B):
    bp_set = [ ]
    valid = 0  # whether there exists the samples preimage.
    satisfy = 0  # it is a sample satisfies the property.
    for k in range(counts):
        sample = bpg.mc_sample(polyOut)
        bp_set = bpg.whole_back_computation(layer, K, W, U, B, [sample])
        # A sample may generate a set of backward_propagation points.
        for bp in bp_set:
            if (pre.belong_judge(bp, polyInp)):
                valid = valid + 1
                if(pre.belong_judge(sample, polyPro)):
                    satisfy = satisfy + 1
                    break
                else:
                    break
    
    ratio = (satisfy / valid) * 1.0
    return  ratio


def qualitive_verify(polyOut, polyPro):
    for poly in polyOut:
        if (not pre.include_judge(poly, polyPro)):
            return False
    return True 

"""
def quantatitive_verify_vr(polyOut, polyPro):
    bo, Ao = pcon.to_H(polyOut)
    bp, Ap = pcon.to_H(polyPro)
    A = np.vstack((Ao, Ap))
    b = np.vstack((bo, bp))
    M = np.hstack((b, -A))
    polyInter = pcon.to_V(M)
    print(polyInter.point, polyInter.ray)
    print(polyOut.point)
    Volint = pre.convexhull_volume(polyInter.point)
    Volout = pre.convexhull_volume(polyOut.point)
    print(Volint, Volout)
    return Volint / Volout
"""


def quantatitive_verify_vr(data1,data2):

    poly1 = Polygon(data1).convex_hull      # Polygon：多边形对象
    poly2 = Polygon(data2).convex_hull

    if not poly1.intersects(poly2):
        inter_area = 0  # 如果两四边形不相交
    else:
        inter_area = poly1.intersection(poly2).area  # 相交面积
    return inter_area/poly1.area


def quantatitive_verify_vr_2(data1,data2):

    line1 = LineString(data1)    
    line2 = LineString(data2)

    if not line1.intersects(line2):
        line3 = 0  # 如果两四边形不相交
    else:
        interp = line1.intersection(line2)
        if (data1[0][0] >= data1[0][1]):
            line3 = LineString([data1[0],interp])
            line3 = LineString([data1[0],interp]).length
        else:
            line3 = LineString([data1[1],interp]).length
    return line3 /line1.length


def quantatitive_verify_mc_nobp(polyOut, polyPro, counts):
    satisfy = 0  # it is a sample satisfies the property.
    for k in range(counts):
        sample = bpg.mc_sample(polyOut)
        #print(sample)
        if(pre.belong_judge(sample, polyPro)):
            satisfy = satisfy + 1
    
    print(satisfy)
    ratio = (satisfy / counts) * 1.0
    return  ratio


    