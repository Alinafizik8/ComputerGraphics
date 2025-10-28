from math import cos, sin
import math
import numpy as np
from PIL import Image,ImageOps

fin = open("model_1.obj")
ZBuff=np.full((2000,2000),np.inf)

def R_matrix(a,b,g):
    a = (a*math.pi)/180
    b = (b*math.pi)/180
    g = (g*math.pi)/180
    m1 = np.array([[1,0,0],[0,cos(a),sin(a)],[0,-sin(a),cos(a)]])
    m2 = np.array([[cos(b), 0, sin(b)], [0, 1, 0], [-sin(b), 0, cos(b)]])
    m3 = np.array([[cos(g),sin(g),0],[-sin(g),cos(g),0],[0,0,1]])
    return (m1 @ m2 @ m3)

def BaricentrCoord(x,y,x0,y0,x1,y1,x2,y2):
    demon = (x0-x2)*(y1-y2)-(x1-x2)*(y0-y2)
    if (demon != 0):
        lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / demon
        lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / demon
        lambda2 = 1.0 - lambda0 - lambda1
        return [lambda0,lambda1,lambda2]
    return [-1,-1,-1]

def Z_Bufferr(l0,l1,l2,z0,z1,z2,x,y):
    if (l0 >= 0.0) & (l2 >= 0.0) & (l1 >= 0.0):
        zz = l0 * z0 + l1 * z1 + l2 * z2
        if (zz < ZBuff[x][y]):
            ZBuff[x][y] = zz
            return True
    return False


def PrintTreangle(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    xs0 = ((x0*3400)/z0) + 1000
    ys0 = ((y0*3400)/z0) + 1000
    xs1 = ((x1*3400)/z1) + 1000
    ys1 = ((y1*3400)/z1) + 1000
    xs2 = ((x2*3400)/z2) + 1000
    ys2 = ((y2*3400)/z2) + 1000
    xmin = 0 if min(xs0,xs1,xs2)<0 else int(min(xs0,xs1,xs2))
    ymin = 0 if min(ys0,ys1,ys2)<0 else int(min(ys0,ys1,ys2))
    xmax = int(max(xs0,xs1,xs2))+1
    ymax = int(max(ys0,ys1,ys2))+1
    ALPHA = Normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    if (ALPHA < 0):
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                L = BaricentrCoord(x, y, xs0, ys0, xs1, ys1, xs2, ys2)
                l0 = L[0]
                l1 = L[1]
                l2 = L[2]
                if (Z_Bufferr(l0,l1,l2,z0,z1,z2,x,y)):
                    img_mat[y, x] = [255*(-ALPHA),105*(-ALPHA),180*(-ALPHA)]

def Normal(x0, y0, z0, x1, y1, z1, x2, y2,z2):
    N = np.cross([x1-x2,y1-y2,z1-z2],[x1-x0,y1-y0,z1-z0])
    alfa = np.dot(N, [0, 0, 1]) / (1 * math.sqrt(N[0] ** 2 + N[1] ** 2 + N[2] ** 2))
    return alfa

img_mat = np.zeros((2000, 2000, 3), dtype=np.uint8)
for i in range (2000):
   for j in range (2000):
        img_mat[i,j]=[255]

def FILE():
    v = []
    f = []
    for s in fin:
        sp=s.split()
        if sp[0]=='v':
            v.append([float(sp[1]),float(sp[2]),float(sp[3])])
            #img_mat[int(8000 * np.double(sp[2]) + 1000), int(8000 * np.double(sp[1]) + 1000)] = [248, 24, 148]
        elif sp[0] == 'f':
            spps = []
            for S in sp:
                spps.append(S.split('/'))
            f.append([int(spps[1][0]),int(spps[2][0]),int(spps[3][0])])
    r = R_matrix(0, 180, 0)
    arr = [0, -0.045,0.2]
    for i in range(len(f)):
        x0 = v[f[i][0] - 1][0]
        y0 = v[f[i][0] - 1][1]
        z0 = v[f[i][0] - 1][2]
        x1 = v[f[i][1] - 1][0]
        y1 = v[f[i][1] - 1][1]
        z1 = v[f[i][1] - 1][2]
        x2 = v[f[i][2] - 1][0]
        y2 = v[f[i][2] - 1][1]
        z2 = v[f[i][2] - 1][2]
        arr0 = r @ np.array([x0,y0,z0]) + np.array(arr)
        arr1 = r @ np.array([x1,y1,z1])  + np.array(arr)
        arr2 = r @ np.array([x2,y2,z2]) + np.array(arr)
        PrintTreangle(arr0[0], arr0[1], arr0[2], arr1.item(0), arr1.item(1), arr1.item(2), arr2.item(0), arr2.item(1), arr2.item(2))

for k in range (13):
    x0, y0 = 100, 100
    x1 = 100 + int(95 * cos((2 * math.pi / 13) * k))
    y1 = 100 + int(95 * sin((2 * math.pi / 13) * k))
    FILE()

img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('img3_5.png')
