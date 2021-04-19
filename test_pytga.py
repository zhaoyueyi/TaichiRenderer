import taichi as ti
import time
import math
import numpy as np
import pyTGA

TGA_white = (255, 255, 255, 255)
TGA_red   = (255,   0,   0, 255)

def frange(start, stop, step):
     x = start
     while x < stop:
         yield x
         x += step

def line(x0: int, y0: int, x1: int, y1: int, image, color):
    for t in frange(0.0, 1.0, 0.1):
        x = x0*(1.-t) + x1*t;
        y = y0*(1.-t) + y1*t;
        image[x][y] = color


# ti.init(arch=ti.gpu)
# n = 320
# pixels = ti.field(dtype=float , shape=(n * 2, n))
# @ti.func
# def complex_sqr(z):
#     return ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2])
# @ti.kernel
# def paint(t: float):
#     for i, j in pixels: # Parallized over all pixels
#         c = ti.Vector([-0.8, ti.cos(t) * 0.2])
#         z = ti.Vector([i / n - 1, j / n - 0.5]) * 2
#         iterations = 0
#         while z.norm() < 20 and iterations < 50:
#             z = complex_sqr(z) + c
#             iterations += 1
#         pixels[i, j] = 1 - iterations * 0.02
# gui = ti.GUI("Julia Set", res=(n * 2, n))
# for i in range(1000000):
#     paint(i * 0.03)
#     gui.set_image(pixels)
#     gui.show()

def main():
    TGA_image = [[(0,0,0,0)for i in range(100)] for i in range(100)]
    line(13, 20, 80, 40, TGA_image, TGA_white);
    # TGA_image[41][52] = TGA_red
    print(TGA_image)
    image = pyTGA.Image(data=TGA_image)
    image.save("output.tga")

    # data_bw = [
    #     [0, 255, 0, 0],
    #     [0, 0, 255, 0],
    #     [255, 255, 255, 0]
    # ]
    #
    # data_rgb = [
    #     [(0, 0, 0), (255, 0, 0), (0, 0, 0), (0, 0, 0)],
    #     [(0, 0, 0), (0, 0, 0), (255, 0, 0), (0, 0, 0)],
    #     [(255, 0, 0), (255, 0, 0), (255, 0, 0), (0, 0, 0)]
    # ]
    #
    # data_rgba = [
    #     [(0, 0, 0, 0), (255, 0, 0, 150), (0, 0, 0, 0), (0, 0, 0, 0)],
    #     [(0, 0, 0, 0), (0, 0, 0, 0), (255, 0, 0, 150), (0, 0, 0, 0)],
    #     [(255, 0, 0, 150), (255, 0, 0, 150), (255, 0, 0, 150), (0, 0, 0, 0)]
    # ]
    #
    # ##
    # # Create from grayscale data
    # image = pyTGA.Image(data=data_bw)
    # # Save as TGA
    # image.save("image_black_and_white")
    #
    # ##
    # # Create from RGB data
    # image = pyTGA.Image(data=data_rgb)
    # image.save("image_rgb")
    #
    # ##
    # # Create from RGBA data
    # image = pyTGA.Image(data=data_rgba)
    # image.save("image_rgba")
    #
    # ##
    # # Save with RLE compression
    # image = pyTGA.Image(data=data_rgba)
    # image.save("image_rgba_compressed", compress=True)
    #
    # ##
    # # Save in original format
    # image = pyTGA.Image(data=data_rgba)
    # image.save("image_rgba_original", original_format=True)
    #
    # ##
    # # Save with 16 bit depth
    # # You can start also from RGB, but you will lose data
    # image = pyTGA.Image(data=data_rgb)
    # image.save("test_16", force_16_bit=True)
    #
    # data_rgb_16 = [
    #     [(0, 0, 0), (31, 0, 0), (0, 0, 0), (0, 0, 0)],
    #     [(0, 0, 0), (0, 0, 0), (31, 0, 0), (0, 0, 0)],
    #     [(31, 0, 0), (31, 0, 0), (31, 0, 0), (0, 0, 0)]
    # ]
    #
    # image = pyTGA.Image(data=data_rgb_16)
    # image.save("image_16_bit", force_16_bit=True)
    #
    # ##
    # # Load and modify an image
    # image = pyTGA.Image()
    # image.load("image_black_and_white.tga").set_pixel(0, 3, 175)
    # image.save("image_black_and_white_mod.tga")
    #
    # # Get some data
    # print(image.get_pixel(0, 3))
    # print(image.get_pixels())

if __name__ == '__main__':
    main()
