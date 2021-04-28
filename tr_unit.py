# @Time : 2021/4/27 17:21
# @Author : 赵曰艺
# @File : tr_unit.py
# @Software: PyCharm
# coding:utf-8
import taichi as ti

@ti.data_oriented
class TRUnit:
    def __init__(self, img):
        self.img = img

    def clear(self):
        self.img.fill(0)

    @ti.func
    def shade(self, pos, color):
        raise NotImplementedError

class ColorUnit(TRUnit):
    @ti.func
    def shade(self, pos, color):
        self.img[pos] = color