# @Time : 2021/4/26 0:00
# @Author : 赵曰艺
# @File : tr_shader.py
# @Software: PyCharm
# coding:utf-8
import taichi as ti
import tr_unit
from hacker import *

MAX = 2**20

@ti.data_oriented
class PointShader:
    def __init__(self, renderer, maxfaces=MAX):
        self.maxfaces = maxfaces
        self.nfaces = ti.field(int, ())
        self.verts =  ti.Vector.field(3, float, (maxfaces, 3))

        self.renderer = renderer
        self.res = self.renderer.res
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

    @ti.kernel
    def render(self, unit:ti.template()):
        for P in ti.grouped(self.occup):
            self.occup[P] = -1
        for f in ti.smart(self.get_faces_range()):
            # A, B, C = self.get_face_vertices(f)
            Tee = self.get_face_vertices(f)
            for i in ti.static(range(3)):
                x = int(Tee[i].x*(self.res[0]/2)+(self.res[0]/2))
                y = int(Tee[i].y*(self.res[1]/2)+(self.res[1]/2))
                self.occup[x, y] = 1
        for P in ti.grouped(self.occup):
            if self.occup[P] == -1:continue
            color = V(1., 1., 1.)
            unit.shade(P, color)

@ti.data_oriented
class TriangleShader:
    def __init__(self, renderer, maxfaces):
        self.maxfaces = maxfaces
        self.verts = ti.Vector.field(3, float, (maxfaces, 3))
        self.norms = ti.Vector.field(3, float, (maxfaces, 3))
        self.coors = ti.Vector.field(2, float, (maxfaces, 3))
        self.nfaces = ti.field(int, ())

        self.renderer = renderer
        self.res = self.renderer.res
        self.occup = ti.field(int, self.res)

        self.bcn = ti.Vector.field(2, float, maxfaces)
        self.can = ti.Vector.field(2, float, maxfaces)
        self.boo = ti.Vector.field(2, float, maxfaces)
        self.coo = ti.Vector.field(2, float, maxfaces)
        self.wsc = ti.Vector.field(3, float, maxfaces)

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

    @ti.kernel
    def render_occup(self):
        for P in ti.grouped(self.occup):
            self.occup[P] = -1
        # for f in ti.smart(self.get_faces_range()):
        #     Al, Bl, Cl = self.get_face_vertices(f)
        #     # Av, Bv, Cv = [self.engine.to_viewspace(p) for p in [Al, Bl, Cl]]
        #     Av, Bv, Cv = [p for p in [Al, Bl, Cl]]
        #     facing = (Bv.xy - Av.xy).cross(Cv.xy - Av.xy)
        #     if facing <= 0:
        #         # if ti.static(self.culling):    ############################################################
        #         continue
        #
        #     # if ti.static(self.clipping):   ############################################################
        #     #     if not all(-1 <= Av <= 1):
        #     #         if not all(-1 <= Bv <= 1):
        #     #             if not all(-1 <= Cv <= 1):
        #     #                 continue
        #
        #     # a, b, c = [self.engine.to_viewport(p) for p in [Av, Bv, Cv]]   ############################################################
        #     a, b, c = [p for p in [Av, Bv, Cv]]
        #
        #     bot, top = ifloor(min(a, b, c)), iceil(max(a, b, c))
        #     bot, top = max(bot, 0), min(top, self.res - 1)
        #     n = (b - a).cross(c - a)
        #     bcn = (b - c) / n
        #     can = (c - a) / n
        #     wscale = 1 / ti.Vector([mapply(self.engine.W2V[None], p, 1)[1] for p in [Al, Bl, Cl]])   ############################################################
        #     for P in ti.grouped(ti.ndrange((bot.x, top.x + 1), (bot.y, top.y + 1))):
        #         pos = float(P) + self.engine.bias[None]   ############################################################
        #         w_bc = (pos - b).cross(bcn)
        #         w_ca = (pos - c).cross(can)
        #         wei = V(w_bc, w_ca, 1 - w_bc - w_ca) * wscale
        #         wei /= wei.x + wei.y + wei.z
        #         if all(wei >= 0):
        #             depth_f = wei.x * Av.z + wei.y * Bv.z + wei.z * Cv.z
        #             depth = int(depth_f * self.engine.maxdepth)   ############################################################
        #             if ti.atomic_min(self.engine.depth[P], depth) > depth:   ############################################################
        #                 if self.engine.depth[P] >= depth:   ############################################################
        #                     self.occup[P] = f

            self.bcn[f] = bcn
            self.can[f] = can
            self.boo[f] = b
            self.coo[f] = c
            self.wsc[f] = wscale

    @ti.kernel
    def render_color(self, shader: ti.template()):
        for P in ti.grouped(self.occup):
            f = self.occup[P]
            if f == -1:
                continue

            Al, Bl, Cl = self.get_face_vertices(f)

            bcn = self.bcn[f]
            can = self.can[f]
            b = self.boo[f]
            c = self.coo[f]
            wscale = self.wsc[f]
            p = float(P) + self.engine.bias[None]   ############################################################
            w_bc = (p - b).cross(bcn)
            w_ca = (p - c).cross(can)
            wei = V(w_bc, w_ca, 1 - w_bc - w_ca) * wscale
            wei /= wei.x + wei.y + wei.z

            self.interpolate(shader, P, p, f, wei, Al, Bl, Cl)

    @ti.func
    def interpolate(self, shader: ti.template(), P, p, f, wei, A, B, C):
        pos = wei.x * A + wei.y * B + wei.z * C

        normal = V(0., 0., 0.)
        if ti.static(self.smoothing):   ############################################################
            An, Bn, Cn = self.get_face_normals(f)
            normal = wei.x * An + wei.y * Bn + wei.z * Cn
        else:
            normal = (B - A).cross(C - A)  # let the shader normalize it
        normal = normal.normalized()

        texcoord = V(0., 0.)
        if ti.static(self.texturing):   ############################################################
            At, Bt, Ct = self.get_face_texcoords(f)
            texcoord = wei.x * At + wei.y * Bt + wei.z * Ct

        color = V(1., 1., 1.)
        shader.shade_color(self.engine, P, p, f, pos, normal, texcoord, color)   ############################################################

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

def V(*xs):
    return ti.Vector(xs)

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

# @eval('lambda x: x()')
# def _():
#     if hasattr(ti, 'smart'):
#         return
#
#     ti.smart = lambda x: x