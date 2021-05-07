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

        # self.camera = V(0, 0, 3)
        self.mat_projection = ti.Matrix.field(4, 4, float, ())
        self.mat_viewport = ti.Matrix.field(4, 4, float, ())

        self.camera = V(0, 0, 0.01)

        # self.mat_pers = ti.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, self.camera.norm(), 1]])
        self.mat_pers = ti.Matrix([[1.732, 0, 0, 0], [0, 1.732, 0, 0], [0, 0, -1, -0.1], [0, 0, -1, 0]])
        # mm = perspective()
        # print(mm)
        # self.mat_pers = ti.Matrix(4, 4, float, ())
        # self.mat = ti.Matrix(4, 4, float, ())
        #
        # self.mat_view = ti.Matrix([[384, 0, 0, 512], [0, 384, 0, 512], [0, 0, 1, 0], [0, 0, 0, 1]])
        # self.mat_view = ti.Matrix([[1, 0, 0, 0], [0, 1, -1.e-05, 0], [0, 1.e-05, 1, -3.e+00], [0, 0, 0, 1]])
        self.mat_lookat = ti.Matrix([[1, 0, 0, 0], [0, 1, -0.00001, 0], [0, 0.00001, 1, -3.e+00], [0, 0, 0, 1]])
        self.mat = ti.Matrix([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
        self.set_camera()
        # self.mat_pers.from_numpy(ndarray=mm)
        # print(self.mat)

        # self.options = options
        # self.light_dir = ti.Vector(3, ti.f32, ())

        @ti.materialize_callback
        @ti.kernel
        def init_engine():
            self.mat_projection[None] = ti.Matrix.identity(float, 4)
            self.mat_projection[None][3, 2] = -1/self.camera.z
            # viewport = self.viewport(self.res[0] / 8, self.res[1] / 8, self.res[0] * 3 / 4, self.res[1] * 3 / 4)
            # self.mat_viewport[None].from_numpy(np.array(viewport, dtype=np.float32))
            self.W2V[None] = ti.Matrix.identity(float, 4)
            self.W2V[None][2, 2] = -1
            '''
            1 0  0 0
            0 1  0 0
            0 0 -1 0
            0 0  0 1
            '''
            self.V2W[None] = ti.Matrix.identity(float, 4)
            self.V2W[None][2, 2] = -1
            self.bias[None] = [0.5, 0.5]

        ti.materialize_callback(self.clear_depth)

    def set_camera(self):
        mat_lookat = self.lookat()
        for i in range(3):
            self.mat_lookat[i, 0] = mat_lookat[i, 0]
            self.mat_lookat[i, 1] = mat_lookat[i, 1]
            self.mat_lookat[i, 2] = mat_lookat[i, 2]
            self.mat_lookat[i, 3] = mat_lookat[i, 3]
        # print(self.mat_view)
        self.mat = self.mat_pers @ self.mat_lookat
        # self.W2V.from_numpy(np.array(W2V, dtype=np.float32))
        # print(mat_lookat)

    def lookat(self, eye=(0, 0, 3), center=(0, 0, 0), up=(0, 1, 1e-12)):
        center = np.array(center, dtype=float)
        eye = np.array(eye, dtype=float)
        up = np.array(up, dtype=float)

        fwd = -eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)

        lin = np.transpose(np.stack([right, up, -fwd]))
        return np.linalg.inv(affine(lin, (center + eye)))

    @ti.func
    def apply_mat(self, pos):
        # p = V(pos[0], pos[1], pos[2], 1)
        # for i in ti.static(range(3)):
        #     p[i] = pos[0]*self.mat[i, 0]+pos[1]*self.mat[i, 1]+pos[2]*self.mat[i, 2]+self.mat[i, 3]
        # return V(p[0]/p[3], p[1]/p[3], p[2]/p[3])
        return mapply_pos(self.mat, pos)

    @ti.func
    def to_viewspace1(self, p):
        return mapply_pos(self.W2V[None], p)
        # print(self.W2V[None])
        # return mapply_pos(self.mat_projection[None], p)

    @ti.func
    def to_viewport(self, p):
        return (p.xy * 0.5 + 0.5) * self.res

    @ti.func
    def viewport(self, x, y, w, h):
        mat = np.eye(4)
        mat[0, 3] = x+w/2
        mat[1, 3] = y+h/2
        mat[2, 3] = -1/2
        mat[0, 0] = w/2
        mat[1, 1] = h/2
        mat[2, 2] = -1/2
        return mat

    @ti.kernel
    def clear_depth(self):
        for P in ti.grouped(self.depth):
            self.depth[P] = -1

    # def set_camera(self, view, proj):
    #     W2V = proj @ view
    #     V2W = np.linalg.inv(W2V)
    #     self.W2V.from_numpy(np.array(W2V, dtype=np.float32))
    #     self.V2W.from_numpy(np.array(V2W, dtype=np.float32))
    #

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
        self.render()
        # gui.set_image(self.image)
        # gui.show(save_file)
        while gui.running:
        # #     self.render()
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

def frustum(left=-1, right=1, bottom=-1, top=1, near=1, far=100):
    lin = np.eye(4)
    lin[0, 0] = 2 * near / (right - left)
    lin[1, 1] = 2 * near / (top - bottom)
    lin[0, 2] = (right + left) / (right - left)
    lin[1, 2] = (top + bottom) / (top - bottom)
    lin[2, 2] = -(far + near) / (far - near)
    lin[2, 3] = -2 * far * near / (far - near)
    lin[3, 2] = -1
    lin[3, 3] = 0
    return lin

def affine(lin, pos):
    lin = np.concatenate([lin, np.zeros((1, 3))], axis=0)
    pos = np.concatenate([pos, np.ones(1)])
    lin = np.concatenate([lin, pos[:, None]], axis=1)
    return lin

'''
透视投影矩阵： y方向视角，纵横比，近剪裁面到原点距离，远剪裁面到原点距离
'''
def perspective(fov=60, aspect=1, near=0.05, far=500):
    fov = np.tan(np.radians(fov) / 2)
    ax, ay = fov * aspect, fov
    return frustum(-near * ax, near * ax, -near * ay, near * ay, near, far)


class namespace(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from None
