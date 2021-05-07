# @Time : 2021/4/26 0:00
# @Author : 赵曰艺
# @File : tr_shader.py
# @Software: PyCharm
# coding:utf-8
import numpy as np

from common import *
from hacker import *
import tr_utils as tu
import tr_unit

@ti.data_oriented
class PointShader:
    def __init__(self, renderer, maxfaces=MAX):
        self.maxfaces = maxfaces
        self.nfaces = ti.field(int, ())
        self.verts = ti.Vector.field(3, float, (maxfaces, 3))

        self.renderer = renderer
        self.res = self.renderer.res
        self.img = self.renderer.image
        self.model = None
        self.occup = ti.field(int, self.res)

    @ti.pyfunc
    def get_faces_range(self):
        for i in range(self.nfaces[None]):
            yield i

    @ti.pyfunc
    def get_face_vertices(self, f):
        a, b, c = self.verts[f, 0], self.verts[f, 1], self.verts[f, 2]
        return a, b, c

    @ti.kernel
    def set_model(self, model: ti.template()):
        self.model = model
        self.nfaces[None] = model.get_nfaces()
        for i in range(self.nfaces[None]):
            verts = model.get_face_verts(i)
            for k in ti.static(range(3)):
                self.verts[i, k] = verts[k]

    @ti.kernel
    def render(self, unit:ti.template()):
        color = V(1., 1., 1.)
        for P in ti.grouped(self.occup):
            self.occup[P] = -1
        for f in ti.smart(self.get_faces_range()):
            # A, B, C = self.get_face_vertices(f)
            Tee = self.get_face_vertices(f)
            for i in ti.static(range(3)):
                x = int(Tee[i].x*(self.res[0]/2)+(self.res[0]/2))
                y = int(Tee[i].y*(self.res[1]/2)+(self.res[1]/2))
                # self.occup[x, y] = 1
                self.img[x, y] = color
        # for P in ti.grouped(self.occup):
        #     if self.occup[P] == -1:continue
        #     color = V(1., 1., 1.)
        #     unit.shade(P, color)

@ti.data_oriented
class LineShader:
    def __init__(self, renderer, maxfaces=MAX):
        self.maxfaces = maxfaces
        self.nfaces = ti.field(int, ())
        self.verts =  ti.Vector.field(3, float, (maxfaces, 3))

        self.renderer = renderer
        self.res = self.renderer.res
        self.img = self.renderer.image
        self.occup = ti.field(int, self.res)

    @ti.pyfunc
    def get_faces_range(self):
        for i in range(self.nfaces[None]):
            yield i

    @ti.pyfunc
    def get_face_vertices(self, f):
        A, B, C = self.verts[f, 0], self.verts[f, 1], self.verts[f, 2]
        return A, B, C

    @ti.kernel
    def set_model(self, model: ti.template()):
        self.nfaces[None] = model.get_nfaces()
        for i in range(self.nfaces[None]):
            verts = model.get_face_verts(i)
            for k in ti.static(range(3)):
                self.verts[i, k] = verts[k]

    @ti.func
    def to_viewport(self, p):
        return (p.xy * 0.5 + 0.5) * self.res

    @ti.func
    def reverse(self, x, y):
        return y, x

    # 直线光栅化
    @ti.func
    def line(self, a, b, color):
        # #
        # for t in ti.smart(tu.frange(0.0, 1.0, 0.01)):
        #     x = A[0]*(1.-t) + B[0]*t;
        #     y = A[1]*(1.-t) + B[1]*t;
        #     self.img[int(x), int(y)] = color

        # bresenham
        delta = b - a
        delta_abs = abs(delta)
        step, distance = V(1.0, 1.0), 0
        if delta_abs.x >= delta_abs.y:
            step.x = 1.0 if delta.x>0 else -1.0
            step.y = (delta.y/delta.x) * step.x
            distance = ifloor(delta_abs.x)
        else:
            step.y = 1.0 if delta.y>0 else -1.0
            step.x = (delta.x / delta.y) * step.y
            distance = ifloor(delta_abs.y)
        for i in range(distance):
            pos = ifloor(a+i*step)
            self.img[pos] = color

    @ti.kernel
    def render(self, unit:ti.template()):
        color = V(1., 1., 1.)
        for f in ti.smart(self.get_faces_range()):
            A, B, C = self.get_face_vertices(f)
            a, b, c = [self.to_viewport(p) for p in [A, B, C]]
            self.line(a, b, color)
            self.line(b, c, color)
            self.line(c, a, color)

@ti.data_oriented
class TriangleShader:
    def __init__(self, renderer, texture, maxfaces=MAX):
        self.maxfaces = maxfaces
        self.verts = ti.Vector.field(3, float, (maxfaces, 3))
        self.norms = ti.Vector.field(3, float, (maxfaces, 3))
        self.coors = ti.Vector.field(2, float, (maxfaces, 3))
        self.nfaces = ti.field(int, ())

        self.renderer = renderer
        self.res = self.renderer.res
        self.img = self.renderer.image
        self.occup = ti.field(int, self.res)
        self.texture = texture
        # self.mat_view[None] = ti.Matrix.identity(float, 4)
        # self.mat_view.from_numpy(np.array(m, dtype=np.float32))

        self.bcn = ti.Vector.field(2, float, maxfaces)
        self.can = ti.Vector.field(2, float, maxfaces)
        self.boo = ti.Vector.field(2, float, maxfaces)
        self.coo = ti.Vector.field(2, float, maxfaces)
        self.wsc = ti.Vector.field(3, float, maxfaces)


    @ti.func
    def to_viewport(self, p):
        # print(p.xy)
        return (p.xy * 0.5 + 0.5) * self.res
        # return p.xy

    @ti.func
    def get_faces_range(self):
        for i in range(self.nfaces[None]):
            yield i

    @ti.func
    def get_face_vertices(self, f):
        A, B, C = self.verts[f, 0], self.verts[f, 1], self.verts[f, 2]
        return A, B, C

    @ti.func
    def get_face_normals(self, f):
        A, B, C = self.norms[f, 0], self.norms[f, 1], self.norms[f, 2]
        return A, B, C

    @ti.func
    def get_face_texcoords(self, f):
        A, B, C = self.coors[f, 0], self.coors[f, 1], self.coors[f, 2]
        return A, B, C

    @ti.func
    def get_texture_size(self):
        return self.texture[0].shape[0], self.texture[0].shape[1]

    @ti.func
    def get_texture_color(self, p):
        p = ifloor(p * self.get_texture_size())

    @ti.kernel
    def set_model(self, model: ti.template()):
        self.nfaces[None] = model.get_nfaces()
        for i in range(self.nfaces[None]):
            verts = model.get_face_verts(i)
            for k in ti.static(range(3)):
                self.verts[i, k] = verts[k]
            # if ti.static(self.smoothing):
            norms = model.get_face_norms(i)
            for k in ti.static(range(3)):
                self.norms[i, k] = norms[k]
        # if ti.static(self.texturing):
            coors = model.get_face_coors(i)
            for k in ti.static(range(3)):
                self.coors[i, k] = coors[k]

    @ti.func
    def barycentric(self, A, B, C, P):
        v0 = C.xy - A.xy
        v1 = B.xy - A.xy
        v2 = P.xy - A.xy

        d00 = v0.dot(v0)
        d01 = v0.dot(v1)
        d02 = v0.dot(v2)
        d11 = v1.dot(v1)
        d12 = v1.dot(v2)

        inver = 1 / (d00*d11 - d01*d01)
        u = (d11*d02 - d01*d12) * inver
        v = (d00*d12 - d01*d02) * inver
        return 1 if u>0 and v>0 and u+v<=1 else -1

    @ti.kernel
    def render(self, unit:ti.template()):
        color = V(0., 1., 1.)
        for f in ti.smart(self.get_faces_range()):
            A, B, C = self.get_face_vertices(f)
            # mat_pers = ti.Matrix([[1,   0, 0,  0], [0,  1, 0,  0], [0, 0,    1,    0], [0, 0, -10, 1]])
            # mat_view = ti.Matrix([[512, 0, 0,640], [0,512, 0,640], [0, 0,127.5,127.5], [0, 0,   0, 1]])
            # # print(self.mat_view[None])
            # mat = mat_pers @ mat_view
            Av, Bv, Cv = [self.renderer.apply_mat(p) for p in [A, B, C]]
            facing = (Bv.xy - Av.xy).cross(Cv.xy - Av.xy)
            # if facing <= 0:continue

            a, b, c = [self.to_viewport(p) for p in [Av, Bv, Cv]]
            At, Bt, Ct = self.get_face_texcoords(f)
            bot, top = ifloor(min(a, b, c)), iceil(max(a, b, c))
            bot, top = max(bot, 0), min(top, self.res-1)

            normal = (B-A).cross(C-A)
            normal = normal.normalized()
            intensity = normal.dot(color)/color.norm()
            intensity = intensity if intensity>=0 else 0
            light = V(intensity, intensity, intensity)

            n = (b - a).cross(c - a)  # 矩阵叉乘
            bcn = (b - c) / n
            can = (c - a) / n
            wscale = 1 / ti.Vector([mapply(self.renderer.W2V[None], p, 1)[1] for p in [A, B, C]])
            for P in ti.grouped(ti.ndrange((bot.x, top.x+1), (bot.y, top.y+1))):
                pos = float(P) + self.renderer.bias[None]
                w_bc = (pos - b).cross(bcn)
                w_ca = (pos - c).cross(can)
                wei = V(w_bc, w_ca, 1 - w_bc - w_ca) * wscale
                wei /= wei.x + wei.y + wei.z
                if any(wei < 0): continue
                depth_f = wei.x * A.z + wei.y * B.z + wei.z * C.z  # 求z-value
                if self.renderer.depth[P] < depth_f:
                    self.renderer.depth[P] = depth_f
                    texcoord = wei.x*At+wei.y*Bt+wei.z*Ct
                    self.img[P] = self.texture.get_color(texcoord)*intensity


@ti.func
def mapply(mat, pos, wei):
    res = ti.Vector([mat[i, 3] for i in range(3)]) * wei
    for i, j in ti.static(ti.ndrange(3, 3)):
        res[i] += mat[i, j] * pos[j]
    rew = mat[3, 3] * wei
    for i in ti.static(range(3)):
        rew += mat[3, i] * pos[i]
    return res, rew

@ti.data_oriented
class IShader:
    def __init__(self, img):
        self.img = img

    def clear_buffer(self):
        self.img.fill(0)

    @ti.func
    def shade_color(self, engine, P, p, f, pos, normal, texcoord, color):
        raise NotImplementedError

class TexcoordShader(IShader):
    @ti.func
    def shade_color(self, engine, P, p, f, pos, normal, texcoord, color):
        self.img[P] = texcoord

class ColorShader(IShader):
    @ti.func
    def shade_color(self, engine, P, p, f, pos, normal, texcoord, color):
        self.img[P] = color

class ShaderGroup(IShader):
    def __init__(self, shaders=()):
        self.shaders = shaders

    def shade_color(self, *args):
        for shader in self.shaders:
            shader.shade_color(*args)
