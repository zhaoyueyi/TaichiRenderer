# import taichi as ti
# import time
# import cv2
# import tkinter as tk
import numpy as np
from PIL import Image

from model import Model

TGA_white = (255, 255, 255, 255)
TGA_red   = (255,   0,   0, 255)
TGA_green = (  0, 128,   0, 255)
width  = 800
height = 800

# 浮点数range()
def frange(start, stop, step):
     x = start
     while x < stop:
         yield x
         x += step

# 直线光栅化
def line(x0: int, y0: int, x1: int, y1: int, image, color):
    # # ddl
    # for t in frange(0.0, 1.0, 0.01):
    #     x = x0*(1.-t) + x1*t;
    #     y = y0*(1.-t) + y1*t;
    #     image[int(x)][int(y)] = color

    # bresenham
    steep = False
    if abs(x0-x1) < abs(y0-y1):
        y0, x0 = x0, y0
        y1, x1 = x1, y1
        steep = True

    if x0 > x1:
        x1, x0 = x0, x1
        y1, y0 = y0, y1

    dx = x1 - x0
    dy = y1 - y0
    derror2 = abs(dy) * 2
    error2 = 0
    y = y0
    for x in frange(x0, x1, 1):
        if steep:
            x = x if x<width else width-1
            y = y if y<height else height-1
            image[x][y] = color
        else:
            # print(y)
            x = x if x < height else height - 1
            y = y if y < width else width - 1
            image[y][x] = color

        error2 += derror2
        if error2 > dx:
            y += (1 if y1>y0 else -1)
            error2 -= dx * 2

# 三角形光栅化
def triangle(t0, t1, t2, image, color):
    if t0[1]==t1[1] and t0[1]==t2[1]: return
    # bubble sort verts from low to high
    if t0[1]>t1[1]: t0, t1 = t1, t0
    if t0[1]>t2[1]: t0, t2 = t2, t0
    if t1[1]>t2[1]: t2, t1 = t1, t2

    # # border
    # line(t0[0], t0[1], t1[0], t1[1], image, color)
    # line(t1[0], t1[1], t2[0], t2[1], image, color)
    # line(t2[0], t2[1], t0[0], t0[1], image, color)

    # scanline
    total_height = t2[1] - t0[1]
    for i in range(0, total_height, 1):
        second_half = i>t1[1]-t0[1] or t1[1]==t0[1]
        segment_height = t2[1]-t1[1] if second_half else t1[1]-t0[1]
        alpha = float(i / total_height)
        beta  = float((i - (t1[1]-t0[1] if second_half else 0)) / segment_height)  # might div 0
        A = t0 + (np.subtract(t2, t0)) * alpha
        B = (t1 + (np.subtract(t2, t1)) * beta) if second_half else (t0 + (np.subtract(t1, t0)) * beta)
        if A[0]>B[0]: A, B = B, A
        for j in range(int(A[0]), int(B[0]), 1):
            image[j][t0[1]+i] = color


    # for y in range(t0[1], t1[1], 1):
    #     segment_height = t1[1] - t0[1] + 1
    #     alpha = float((y-t0[1]) / total_height)
    #     beta  = float((y-t0[1]) / segment_height)  # might div 0
    #     A = t0 + (np.subtract(t2, t0)) * alpha
    #     B = t0 + (np.subtract(t1, t0)) * beta
    #     if A[0]>B[0]: A, B = B, A
    #     for j in range(int(A[0]), int(B[0]), 1):
    #         image[j][y] = color
    #
    # for y in range(t1[1], t2[1], 1):
    #     segment_height = t2[1] - t1[1] + 1
    #     alpha = float((y - t0[1]) / total_height)
    #     beta  = float((y - t1[1]) / segment_height)  # might div 0
    #     A = t0 + (np.subtract(t2, t0)) * alpha
    #     B = t1 + (np.subtract(t2, t1)) * beta
    #     if A[0] > B[0]: A, B = B, A
    #     for j in range(int(A[0]), int(B[0]), 1):
    #         image[j][y] = color
    #     # image[int(A[0])][y] = TGA_red
    #     # image[int(B[0])][y] = TGA_green

def main():
    # init
    matrix_image = Image.new('RGBA', (width, height), (0, 0, 0, 255))
    matrix_image = np.asarray(matrix_image)
    matrix_image.flags.writeable = True

    # model read
    # model = Model('obj/african_head/african_head')
    # print(model.faces())

    # render
    # for face in model.faces():
    #     # print(face)
    #     for j in range(3):
    #         v0 = model.vert(face[j])
    #         v1 = model.vert(face[(j+1)%3])
    #         x0 = int((v0[0]+1.)*width/2.)
    #         y0 = int((v0[1]+1.)*height/2.)
    #         x1 = int((v1[0]+1.)*width/2.)
    #         y1 = int((v1[1]+1.)*height/2.)
    #         line(x0, y0, x1, y1, matrix_image, TGA_white)

    t0 = [[10, 70], [50, 160], [70, 80]]
    t1 = [[180, 50], [150, 1], [70, 180]]
    t2 = [[180, 150], [120, 160], [130, 180]]
    triangle(t0[0], t0[1], t0[2], matrix_image, TGA_red)
    triangle(t1[0], t1[1], t1[2], matrix_image, TGA_white)
    triangle(t2[0], t2[1], t2[2], matrix_image, TGA_green)

    # show&save
    matrix_image = Image.fromarray(matrix_image).transpose(Image.FLIP_TOP_BOTTOM)  # np矩阵转图像 PIL绘制以左上角为原点 所以上下翻转为以左下为原点
    matrix_image.show()
    matrix_image.save("output.tga")



if __name__ == '__main__':
    main()
