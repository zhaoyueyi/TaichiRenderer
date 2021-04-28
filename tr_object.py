# @Time : 2021/4/25 18:12
# @Author : 赵曰艺
# @File : tr_object.py
# @Software: PyCharm
# coding:utf-8
import taichi as ti
import taichi_glsl as tg

@ti.data_oriented
class Object:
    def render(self):
        raise NotImplementedError

@ti.data_oriented
class Line(Object):
    def __init__(self, begin, end):  # [xxx, xxx, xxx], [xxx, xxx, xxx]
        self.begin = begin
        self.end = end

    @ti.func
    def render(self, renderer):
        width = 1
        A = self.begin
        B = self.end
        A, B = min(A, B), max(A, B)
        mold = tg.normalize(B - A)
        for X in ti.grouped(ti.ndrange((A.x - width, B.x + width),
                                       (A.y - width, B.y + width))):
            udf = abs(tg.cross(X - A, mold))
            renderer.image[int(X)] = tg.vec3(tg.smoothstep(udf, width, 0))

@ti.data_oriented
class Triangle(Object):
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def to_lines(self):
        return Line(self.b, self.c), Line(self.c, self.a), Line(self.a, self.b)

    @ti.func
    def render(self, renderer):
        A, B, C = self.to_lines()
        A.render(renderer)
        B.render(renderer)
        C.render(renderer)

