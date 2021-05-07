# @Time : 2021/4/30 23:02
# @Author : 赵曰艺
# @File : common.py
# @Software: PyCharm
# coding:utf-8

import taichi as ti
import taichi_glsl as ts
import numpy as np
from tr_utils import texture_as_field

MAX = 2**20

def V(*xs):
    return ti.Vector(xs)

@ti.pyfunc
def ifloor(x):
    return int(ti.floor(x))

@ti.pyfunc
def iceil(x):
    return int(ti.ceil(x))


@ti.func
def mapply(mat, pos, wei):
    res = ti.Vector([mat[i, 3] for i in range(3)]) * wei
    for i, j in ti.static(ti.ndrange(3, 3)):
        res[i] += mat[i, j] * pos[j]
    rew = mat[3, 3] * wei
    for i in ti.static(range(3)):
        rew += mat[3, i] * pos[i]
    return res, rew

@ti.func
def mapply_pos(mat, pos):
    res, rew = mapply(mat, pos, 1)
    return res / rew

def totuple(x):
    if x is None:
        x = []
    if isinstance(x, ti.Matrix):
        x = x.entries
    if isinstance(x, list):
        x = tuple(x)
    if not isinstance(x, tuple):
        x = [x]
    if isinstance(x, tuple) and len(x) and x[0] is None:
        x = []
    return tuple(x)


def tovector(x):
    return ti.Vector(totuple(x))

@ti.pyfunc
def Vprod(w):
    v = tovector(w)
    if ti.static(not v.entries):
        return 1
    x = v.entries[0]
    if ti.static(len(v.entries) > 1):
        for y in ti.static(v.entries[1:]):
            x *= y
    return x

@ti.data_oriented
class Node:
    arguments = []
    defaults = []

    def __init__(self, **kwargs):
        self.params = {}
        for dfl, key in zip(self.defaults, self.arguments):
            if key not in kwargs:
                if dfl is None:
                    raise ValueError(f'`{key}` must specified for `{type(self)}`')
                value = dfl
            else:
                value = kwargs[key]
                del kwargs[key]

            if isinstance(value, str):
                if any(value.endswith(x) for x in ['.tga', '.png', '.jpg', '.bmp']):
                    value = Texture(value)
            self.params[key] = value

        for key in kwargs.keys():
            raise TypeError(
                    f"{type(self).__name__}() got an unexpected keyword argument '{key}', supported keywords are: {self.arguments}")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(type(self))

    def param(self, key, *args, **kwargs):
        return self.params[key](*args, **kwargs)

@ti.data_oriented
class Texture:

    def __init__(self, path):
        self.texture = texture_as_field(path)

    @ti.func
    def get_color(self, texcoord):
        maxcoor = V(*self.texture.shape) - 1
        coor = texcoord * maxcoor
        return bilerp(self.texture, coor)

@ti.func
def bilerp(f: ti.template(), pos):
    p = float(pos)
    I = ifloor(p)
    x = p - I
    y = 1 - x
    return (f[I + V(1, 1)] * x[0] * x[1] +
            f[I + V(1, 0)] * x[0] * y[1] +
            f[I + V(0, 0)] * y[0] * y[1] +
            f[I + V(0, 1)] * y[0] * x[1])
