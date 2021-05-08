# @Time : 2021/4/22 23:51
# @Author : 赵曰艺
# @File : ti_object.py
# @Software: PyCharm
# coding:utf-8

from common import *
import tr_utils as tu

@ti.data_oriented
class TRModel:
    def __init__(self, obj_name:str):
        obj = tu.readobj(obj_name)

        self.faces = ti.Matrix.field(3, 3, int, len(obj['f']))
        self.verts = ti.Vector.field(3, float, len(obj['v']))  # [[xxx, xxx, xxx], [], []...]
        self.coors = ti.Vector.field(2, float, len(obj['vt']))  # [[xxx, xxx], [], []...]
        self.norms = ti.Vector.field(3, float, len(obj['vn']))  # [[xxx, xxx, xxx], [], []...]
        self.name  = obj_name

        @ti.materialize_callback
        def init_mesh():
            faces = obj['f']
            if len(faces.shape) == 2:
                faces = np.stack([faces, faces, faces], axis=2)
            self.faces.from_numpy(faces.astype(np.uint32))
            self.verts.from_numpy(obj['v'])
            self.coors.from_numpy(obj['vt'])
            self.norms.from_numpy(obj['vn'])

    @ti.func
    def get_texture_size(self):
        return self.texture.shape[0], self.texture.shape[1]

    @ti.func
    def get_texture_color(self, p):
        p = ifloor(p*self.get_texture_size())
        # return self.texture[p]

    @ti.func
    def get_nfaces(self):  # 获取三角形面数
        return self.faces.shape[0]  # [xxx]

    @ti.func
    def _get_face_props(self, prop, index: ti.template(), n):  # 获取面参数
        a = prop[self.faces[n][0, index]]
        b = prop[self.faces[n][1, index]]
        c = prop[self.faces[n][2, index]]
        return a, b, c  # [xxx, xxx, xxx]

    @ti.func
    def get_face_verts(self, n):  # 获取面顶点序号
        return self._get_face_props(self.verts, 0, n)

    @ti.func
    def get_face_coors(self, n):  # 获取面纹理坐标
        return self._get_face_props(self.coors, 1, n)

    @ti.func
    def get_face_norms(self, n):
        return self._get_face_props(self.norms, 2, n)
