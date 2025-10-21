from math import cos, sin
import math
import numpy as np
from PIL import Image,ImageOps

fin = open("model_1.obj")
ZBuff=np.full((2000,2000),np.inf)
def BaricentrCoord(x,y,x0,y0,z0,x1,y1,z1,x2,y2,z2):
    demon = (x0-x2)*(y1-y2)-(x1-x2)*(y0-y2)
    if (demon != 0):
        lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / demon
        lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / demon
        lambda2 = 1.0 - lambda0 - lambda1
        if (lambda0 >= 0.0) & (lambda2 >= 0.0) & (lambda1 >= 0.0):
            zz = lambda0 * z0 + lambda1*z1+lambda2*z2
            if (zz<ZBuff[x][y]):
                ZBuff[x][y]=zz
                return True
    return False

def PrintTreangle(x0, y0, z0, x1, y1, z1, x2, y2,z2):
    xmin = 0 if min(x0,x1,x2)<0 else min(x0,x1,x2)
    ymin = 0 if min(y0,y1,y2)<0 else min(y0,y1,y2)
    xmax = max(x0,x1,x2)+1
    ymax = max(y0,y1,y2)+1
    color = [255,0,0]
    img_mat[y0, x0] = color
    img_mat[y1, x1] = color
    img_mat[y2, x2] = color
    ALPHA = Normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    if (ALPHA < 0):
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                if (BaricentrCoord(x, y, x0, y0, z0, x1, y1,z1, x2, y2,z2)):
                    img_mat[y, x] = [175/ALPHA,191/ALPHA,25/ALPHA]

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
            v.append([sp[1],sp[2],sp[3]])
            img_mat[int(8000 * np.double(sp[2]) + 1000), int(8000 * np.double(sp[1]) + 1000)] = [248, 24, 148]
        elif sp[0] == 'f':
            spps = []
            for S in sp:
                spps.append(S.split('/'))
            f.append([spps[1][0],spps[2][0],spps[3][0]])
    for i in range (len(f)):
        x0 = int(8000 * np.double(v[int(f[i][0])-1][0]) + 1000)
        y0 = int(8000 * np.double(v[int(f[i][0])-1][1]) + 1000)
        z0 = int(8000 * np.double(v[int(f[i][0])-1][2]) + 1000)
        x1 = int(8000 * np.double(v[int(f[i][1])-1][0]) + 1000)
        y1 = int(8000 * np.double(v[int(f[i][1])-1][1]) + 1000)
        z1 = int(8000 * np.double(v[int(f[i][1])-1][2]) + 1000)
        x2 = int(8000 * np.double(v[int(f[i][2])-1][0]) + 1000)
        y2 = int(8000 * np.double(v[int(f[i][2])-1][1]) + 1000)
        z2 = int(8000 * np.double(v[int(f[i][2])-1][2]) + 1000)
        PrintTreangle(x0, y0, z0, x1,y1,z1, x2, y2, z2)
for k in range (13):
    x0, y0 = 100, 100
    x1 = 100 + int(95 * cos((2 * math.pi / 13) * k))
    y1 = 100 + int(95 * sin((2 * math.pi / 13) * k))
    FILE()

img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('img2.png')

