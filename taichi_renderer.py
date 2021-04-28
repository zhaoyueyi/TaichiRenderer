# @Time : 2021/4/21 12:05
# @Author : 赵曰艺
# @File : taichi_renderer.py
# @Software: PyCharm
# coding:utf-8
import taichi as ti
# import taichi_glsl as tg
import numpy as np
import tr_utils as tu
from tr_model import TRModel
# from tr_object import Triangle
from tr_shader import PointShader
import tr_unit as tru


# # widget
# width  = 800
# height = 800
# # env
# light_dir = [1,  1, 1]  # light source
# eye       = [-1, 1, 3]  # camera position
# center    = [0,  0, 0]  # camera direction
# up        = [0,  1, 0]  # camera up vector

@ti.data_oriented
class TiRenderer:
    def __init__(self, res=(512, 512), **options):
        self.res = res
        self.image = ti.Vector.field(3, float, self.res)
        self.units = {}
        self.models = {}

        # self.options = options
        # self.light_dir = ti.Vector(3, ti.f32, ())

    def add_model(self, model):
        # tr_model = TRModel(self, model_name)
        shader = PointShader(self)
        unit = tru.ColorUnit(self.image)
        self.units[model] = unit
        self.models[model] = namespace(shader=shader)


    def render(self):
        # self.image.fill(self.bgcolor)
        for model, oinfo in self.models.items():
            unit = self.units[model]
            oinfo.shader.set_model(model)
            oinfo.shader.render(unit)


    def show(self, save_file=None):
        gui = ti.GUI('Taichi Renderer', self.res, fast_gui=True)
        while gui.running:
            self.render()
            gui.set_image(self.image)
            gui.show(save_file)

    # def add_triangle(self, a, b, c):
    #     tr = Triangle(a, b, c)
    #     self.models.append(tr)

    # @ti.func
    # def cook_coor(self, I):
    #     scale = ti.static(2 / min(*self.image.shape()))
    #     coor = (I - tg.vec2(*self.image.shape()) / 2) * scale
    #     return coor
    #
    # @ti.func
    # def uncook_coor(self, coor):
    #     coor_xy = tg.shuffle(coor, 0, 1)
    #     scale = ti.static(min(*self.image.shape()) / 2)
    #     I = coor_xy * scale + tg.vec2(*self.image.shape()) / 2
    #     return I
    #
    # # 直线光栅化
    # def line(self, x0: int, y0: int, x1: int, y1: int, image, color):
    #     # dda
    #     for t in tu.frange(0.0, 1.0, 0.01):
    #         x = x0*(1.-t) + x1*t;
    #         y = y0*(1.-t) + y1*t;
    #         image[int(x)][int(y)] = color

class namespace(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from None
