# @Time : 2021/4/21 12:05
# @Author : 赵曰艺
# @File : taichi_renderer.py
# @Software: PyCharm
# coding:utf-8
import taichi as ti
import taichi_glsl as tg
import numpy as np
from model import Model
from PIL import Image
from tr_model import TRModel

# RGB
white = (255, 255, 255)
red   = (255,   0,   0)
green = (  0, 128,   0)
# widget
width  = 800
height = 800
# env
light_dir = [1,  1, 1]  # light source
eye       = [-1, 1, 3]  # camera position
center    = [0,  0, 0]  # camera direction
up        = [0,  1, 0]  # camera up vector

@ti.func
def line():
    pass


@ti.func
def triangle(t0, t1, t2, color):
    if t0[1] == t1[1] and t0[1] == t2[1]: return
    # bubble sort verts from low to high
    if t0[1] > t1[1]: t0, t1 = t1, t0
    if t0[1] > t2[1]: t0, t2 = t2, t0
    if t1[1] > t2[1]: t2, t1 = t1, t2

    total_height = t2[1] - t0[1]
    for i in range(0, total_height, 1):
        second_half = i > t1[1] - t0[1] or t1[1] == t0[1]
        segment_height = t2[1] - t1[1] if second_half else t1[1] - t0[1]
        alpha = float(i / total_height)
        beta = float((i - (t1[1] - t0[1] if second_half else 0)) / segment_height)  # might div 0
        A = t0 + (np.subtract(t2, t0)) * alpha
        B = (t1 + (np.subtract(t2, t1)) * beta) if second_half else (t0 + (np.subtract(t1, t0)) * beta)
        if A[0] > B[0]: A, B = B, A
        for j in range(int(A[0]), int(B[0]), 1):
            matrix_pixels[j, t0[1] + i] = color


@ti.kernel
def render(model_faces: ti.template()):
    # ti.info('123')
    # face = ti.field(dtype=ti.int16, shape=3)
    for face in ti.grouped(model_faces):
        print(face)
    #     screen_coords = [None, None, None]
    #     world_coords = [None, None, None]
    #     for j in range(3):
    #         ti.debug("123")
    #         v = model.vert(face[j])
    #         screen_coords[j] = [(v[0] + 1.) * width / 2., (v[1] + 1.) * height / 2.]
    #         world_coords[j] = v
    #     triangle(screen_coords[0], screen_coords[1], screen_coords[2], white)

def main():
    # init
    ti.init(arch=ti.cpu)
    # matrix_pixels = ti.field(dtype=ti.uint8, shape=(width, height, 3))
    matrix_pixels = ti.Vector.field(3, float, (1024, 768))
    # model read
    model = Model('obj/african_head/african_head')
    # print(model.nfaces())
    model_faces = ti.field(dtype=ti.int16, shape=(int(model.nfaces()), 3))
    model_faces.from_numpy(model.faces())
    # print(model_faces)
    gui = ti.GUI('Taichi Renderer', (1024, 768), fast_gui=True)
    while gui.running:
    # render(model_faces)
        gui.set_image(matrix_pixels)
        gui.show()

if __name__ == '__main__':
    main()
