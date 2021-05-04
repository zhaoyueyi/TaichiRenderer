# @Time : 2021/4/21 12:05
# @Author : 赵曰艺
# @File : taichi_renderer.py
# @Software: PyCharm
# coding:utf-8

from common import *
import tr_utils as tu
from tr_model import TRModel
from tr_shader import *
import tr_unit as tru

@ti.data_oriented
class TiRenderer:
    def __init__(self, res=(1024, 1024), **options):
        self.res = tovector((res, res) if isinstance(res, int) else res)
        self.image = ti.Vector.field(3, float, self.res)
        self.units = {}
        self.models = {}

        self.W2V = ti.Matrix.field(4, 4, float, ())
        self.V2W = ti.Matrix.field(4, 4, float, ())
        self.bias = ti.Vector.field(2, float, ())
        self.depth = ti.field(float, self.res)
        # self.options = options
        # self.light_dir = ti.Vector(3, ti.f32, ())

        @ti.materialize_callback
        @ti.kernel
        def init_engine():
            self.W2V[None] = ti.Matrix.identity(float, 4)
            self.W2V[None][2, 2] = -1
            self.V2W[None] = ti.Matrix.identity(float, 4)
            self.V2W[None][2, 2] = -1
            self.bias[None] = [0.5, 0.5]
        ti.materialize_callback(self.clear_depth)

    @ti.kernel
    def clear_depth(self):
        for P in ti.grouped(self.depth):
            self.depth[P] = -1

    def add_model(self, model, render_type='point'):
        global shader
        if render_type == 'point':
            shader = PointShader(self)
        elif render_type == 'line':
            shader = LineShader(self)
        elif render_type == 'triangle':
            texture = Texture(model.name+'_diffuse.tga')
            shader = TriangleShader(self, texture=texture)
        unit = tru.ColorUnit(self.image)
        self.units[model] = unit
        self.models[model] = namespace(shader=shader)

    def render(self):
        for model, oinfo in self.models.items():
            unit = self.units[model]
            # oinfo.shader.bind_texture(model.texture)
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
