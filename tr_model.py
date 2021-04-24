# @Time : 2021/4/22 23:51
# @Author : 赵曰艺
# @File : ti_object.py
# @Software: PyCharm
# coding:utf-8
import taichi as ti
import tr_utils as tu
@ti.data_oriented
class TRModel:
    def __init__(self, obj_name:str):
        obj = tu.read_obj(obj_name)
        self.faces = ti.Matrix.field(3, 3, int, len(obj['f']))
        self.verts = ti.Vector.field(3, float, len(obj['v']))
        self.coors = ti.Vector.field(2, float, len(obj['vt']))
        self.norms = ti.Vector.field(3, float, len(obj['vn']))
        self.len_faces = len(obj['f'])
        self.len_verts = len(obj['v'])
        self.len_coors = len(obj['vt'])
        self.len_norms = len(obj['vn'])


    @staticmethod
    @ti.func
    def function():
        pass

    @ti.kernel
    def kernel(self):
        self.function()
