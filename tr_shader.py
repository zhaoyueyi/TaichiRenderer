# @Time : 2021/4/26 0:00
# @Author : 赵曰艺
# @File : tr_shader.py
# @Software: PyCharm
# coding:utf-8

from common import *
from hacker import *

@ti.data_oriented
class PointShader:
    def __init__(self, renderer, maxfaces=MAX):
        self.maxfaces = maxfaces
        self.nfaces = ti.field(int, ())
        self.verts = ti.Vector.field(3, float, (maxfaces, 3))

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
        a, b, c = self.verts[f, 0], self.verts[f, 1], self.verts[f, 2]
        return a, b, c

    def set_texture(self, nouse):
        pass

    @ti.kernel
    def set_model(self, model: ti.template()):
        self.nfaces[None] = model.get_nfaces()
        for i in range(self.nfaces[None]):
            verts = model.get_face_verts(i)
            for k in ti.static(range(3)):
                self.verts[i, k] = verts[k]

    @ti.kernel
    def render(self):
        color = V(1., 1., 1.)
        for f in ti.smart(self.get_faces_range()):
            A, B, C = self.get_face_vertices(f)
            Av, Bv, Cv = [mapply_pos(self.renderer.matrix_fin[None], p) for p in [A, B, C]]
            a, b, c = [self.renderer.to_viewport(p) for p in [Av, Bv, Cv]]
            self.img[a.x, a.y] = color
            self.img[b.x, b.y] = color
            self.img[c.x, c.y] = color

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

    def set_texture(self, nouse):
        pass

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
    def render(self):
        color = V(1., 1., 1.)
        for f in ti.smart(self.get_faces_range()):
            A, B, C = self.get_face_vertices(f)
            a, b, c = [self.to_viewport(p) for p in [A, B, C]]
            self.line(a, b, color)
            self.line(b, c, color)
            self.line(c, a, color)

@ti.data_oriented
class TriangleShader:
    def __init__(self, renderer, texture=NoTexture, maxfaces=MAX):
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

        self.bcn = ti.Vector.field(2, float, maxfaces)
        self.can = ti.Vector.field(2, float, maxfaces)
        self.boo = ti.Vector.field(2, float, maxfaces)
        self.coo = ti.Vector.field(2, float, maxfaces)
        self.wsc = ti.Vector.field(3, float, maxfaces)

    @ti.func
    def to_viewport(self, p):
        return (p.xy * 0.5 + 0.5) * self.res

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

    def set_texture(self, texture):
        self.texture = texture

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
    def render(self):
        light_dir = V(1., 1., 1.)
        for f in ti.smart(self.get_faces_range()):
            A, B, C = self.get_face_vertices(f)
            Av, Bv, Cv = [mapply_pos(self.renderer.matrix_fin[None], p) for p in [A, B, C]]
            facing = (Bv.xy - Av.xy).cross(Cv.xy - Av.xy)
            if facing <= 0:continue

            a, b, c = [self.to_viewport(p) for p in [Av, Bv, Cv]]
            At, Bt, Ct = self.get_face_texcoords(f)
            An, Bn, Cn = self.get_face_normals(f)
            bot, top = ifloor(min(a, b, c)), iceil(max(a, b, c))
            bot, top = max(bot, 0), min(top, self.res-1)

            normal = (B-A).cross(C-A)
            normal = normal.normalized()
            intensity = normal.dot(light_dir)/light_dir.norm()
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
                    norcoord = wei.x*An+wei.y*Bn+wei.z*Cn
                    intensity = norcoord.dot(light_dir)
                    self.img[P] = self.texture.get_color(texcoord)*intensity


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
