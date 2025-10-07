from math import cos, sin
import math
import numpy as np
from PIL import Image,ImageOps
# img_mat = np.zeros((200,200,3), dtype=np.uint8);
fin = open("model_1.obj")
# for i in range (200):
#     for j in range (200):
#         img_mat[i,j]=[0,0,0]
#
# def draw_line1(image,x0,y0,x1,y1,color):
#     count = 100
#     step = 1.0 / count
#     for t in np.arange(0, 1, step):
#         x = round((1.0 - t) * x0 + t * x1)
#         y = round((1.0 - t) * y0 + t * y1)
#         img_mat[y, x] = color
#
# def draw_line2(image,x0,y0,x1,y1,color):
#     count = math.sqrt((x0 - x1) ** 2 + (y0 -y1) ** 2)
#     step = 1.0 / count
#     for t in np.arange(0, 1, step):
#         x = round((1.0 - t) * x0 + t * x1)
#         y = round((1.0 - t) * y0 + t * y1)
#         image[y, x] = color
#
# def draw_line3(image, x0, y0, x1, y1, color):
#     for x in range(x0, x1):
#         t = (x - x0) / (x1 - x0)
#         y = round((1.0 - t) * y0 + t * y1)
#         image[y, x] = color
#
# def draw_line4(image, x0, y0, x1, y1, color):
#     if (x0 > x1):
#         x0, x1 = x1, x0
#         y0, y1 = y1, y0
#     for x in range(x0, x1):
#         t = (x - x0) / (x1 - x0)
#         y = round((1.0 - t) * y0 + t * y1)
#         image[y, x] = color
#
# def draw_line5(image, x0, y0, x1, y1, color):
#     xchange = False
#     if (abs(x0 - x1) < abs(y0 - y1)):
#         x0, y0 = y0, x0
#         x1, y1 = y1, x1
#         xchange = True
#     if (x0 > x1):
#         x0, x1 = x1, x0
#         y0, y1 = y1, y0
#     for x in range(x0, x1):
#         t = (x - x0) / (x1 - x0)
#         y = round((1.0 - t) * y0 + t * y1)
#         if (xchange):
#             image[x, y] = color
#         else:
#             image[y, x] = color
#
# def draw_line6(image, x0, y0, x1, y1, color):
#     xchange = False
#     if (abs(x0 - x1) < abs(y0 - y1)):
#         x0, y0 = y0, x0
#         x1, y1 = y1, x1
#         xchange = True
#     if (x0 > x1):
#         x0, x1 = x1, x0
#         y0, y1 = y1, y0
#     y = y0
#     dy = abs(y1 - y0)/(x1 - x0)
#     derror = 0
#     y_update = 1 if y1 > y0 else -1
#     for x in range(x0, x1):
#         if (xchange):
#             img_mat[x, y] = color
#         else:
#             img_mat[y, x] = color
#         derror += dy
#         if (derror > 0.5):
#             derror -= 1.0
#             y += y_update
#
# def draw_line7(image, x0, y0, x1, y1, color):
#     xchange = False
#     if (abs(x0 - x1) < abs(y0 - y1)):
#         x0, y0 = y0, x0
#         x1, y1 = y1, x1
#         xchange = True
#     if (x0 > x1):
#         x0, x1 = x1, x0
#         y0, y1 = y1, y0
#     y = y0
#     dy = 2.0 * (x1 - x0) * abs(y1 - y0)/(x1 - x0)
#     derror = 0.0
#     y_update = 1 if y1 > y0 else -1
#     for x in range(x0, x1):
#         if (xchange):
#             img_mat[x, y] = color
#         else:
#             img_mat[y, x] = color
#         derror += dy
#         if (derror > 2.0 * (x1 - x0) * 0.5):
#             derror -= 2.0 * (x1 - x0) * 1.0
#             y += y_update
#
def AlgoritmBrezenhema(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = 2 * abs(y1 - y0)
    derror = 0
    y_update = 1 if y1 > y0 else -1
    for x in range(x0, x1):
        if (xchange):
            img_mat[x, y] = color
        else:
            img_mat[y, x] = color
        derror += dy
        if (derror > (x1 - x0)):
            derror -= 2 * (x1 - x0)
            y += y_update
img_mat = np.zeros((2000, 2000, 3), dtype=np.uint8)
for i in range (2000):
    for j in range (2000):
        img_mat[i,j]=[255]
def FILE():
    v = []
    f = []
    spps = []
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
        x1 = int(8000 * np.double(v[int(f[i][1])-1][0]) + 1000)
        y1 = int(8000 * np.double(v[int(f[i][1])-1][1]) + 1000)
        x2 = int(8000 * np.double(v[int(f[i][2])-1][0]) + 1000)
        y2 = int(8000 * np.double(v[int(f[i][2])-1][1]) + 1000)
        AlgoritmBrezenhema(img_mat,x0,y0,x1,y1,[248, 24, 148])
        AlgoritmBrezenhema(img_mat,x2,y2,x1,y1,[248, 24, 148])
        AlgoritmBrezenhema(img_mat,x0,y0,x2,y2,[248, 24, 148])

for k in range (13):
    x0, y0 = 100, 100
    x1 = 100 + int(95 * cos((2 * math.pi / 13) * k))
    y1 = 100 + int(95 * sin((2 * math.pi / 13) * k))
    #draw_line1(img_mat,x0,y0,x1,y1,200)
    #draw_line2(img_mat, x0, y0, x1, y1, 200)
    #draw_line3(img_mat, x0, y0, x1, y1, 200)
    #draw_line4(img_mat, x0, y0, x1, y1, 200)
    #draw_line5(img_mat, x0, y0, x1, y1, 200)
    #draw_line6(img_mat, x0, y0, x1, y1, 200)
    #draw_line7(img_mat, x0, y0, x1, y1, 200)
    #AlgoritmBrezenhema(img_mat, x0, y0, x1, y1, 200)
    FILE()
img = Image.fromarray(img_mat,mode = 'RGB');
img = ImageOps.flip(img)
img.save('img.png')
