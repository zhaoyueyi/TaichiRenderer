# @Time : 2021/4/15 16:05

# @Author : 赵曰艺

# @File : main.py

# @Software: PyCharm

# coding:utf-8
import taichi as ti
import time
import math
import numpy as np
import pyTGA



def main():
    # # 在 GPU 上运行，自动选择后端
    # ti.init(arch=ti.gpu)
    #
    # # 在 GPU 上运行， 使用 NVIDIA CUDA 后端
    ti.init(arch=ti.cuda)
    # # 在 GPU 上运行， 使用 OpenGL 后端
    # ti.init(arch=ti.opengl)
    # # 在 GPU 上运行， 使用苹果 Metal 后端（仅对 OS X）有效
    # ti.init(arch=ti.metal)

    # 在 CPU 上运行 (默认)
    # ti.init(arch=ti.cpu)

    n = 320
    pixels = ti.var(dt=ti.f32, shape=(n * 2, n))  # 稀疏张量(640,320)float

    @ti.func
    def complex_sqr(z):
        return ti.Vector([z[0] ** 2 - z[1] ** 2, z[1] * z[0] * 2])

    @ti.kernel
    def paint(t: ti.f32):
        for i, j in pixels:  # 对于所有像素，并行执行
            c = ti.Vector([-0.8, ti.sin(t) * 0.2])
            z = ti.Vector([float(i) / n - 1, float(j) / n - 0.5]) * 2
            iterations = 0
            while z.norm() < 20 and iterations < 50:
                z = complex_sqr(z) + c
                iterations += 1
            pixels[i, j] = 1 - iterations * 0.02

    gui = ti.GUI("Fractal", (n*2, n))
    gui.button('hello')
    for i in range(1000000):
        paint(i * 0.03)
        gui.set_image(pixels)
        gui.show()
    # for i in range(1000000):
        # paint(i * 0.03)
        # gui.set_image(pixels)
        # gui.show()

if __name__ == '__main__':
    main()