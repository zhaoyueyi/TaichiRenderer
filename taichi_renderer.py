# @Time : 2021/4/21 12:05
# @Author : 赵曰艺
# @File : taichi_renderer.py
# @Software: PyCharm
# coding:utf-8

from common import *
from tr_shader import *
import tr_unit as tru

@ti.data_oriented
class TiRenderer:
    def __init__(self, res=(1024, 1024), **options):
        self.res = tovector((res, res) if isinstance(res, int) else res)
        self.image = ti.Vector.field(3, float, self.res)
        self.units = {}
        self.models = {}
        self.shaders = []

        self.background_color = 0

        self.W2V = ti.Matrix.field(4, 4, float, ())
        self.V2W = ti.Matrix.field(4, 4, float, ())
        self.bias = ti.Vector.field(2, float, ())
        self.depth = ti.field(float, self.res)

        self.matrix_perspective = ti.Matrix.field(4, 4, float, ())
        self.matrix_lookat = ti.Matrix.field(4, 4, float, ())
        self.matrix_fin = ti.Matrix.field(4, 4, float, ())
        self.camera = V(1, 1, 3)
        self.light_dir = V(1., 1., 1.)

        self.temp = perspective()

        self.ps = PointShader(self)
        self.shaders.append(self.ps)
        self.ls = LineShader(self)
        self.shaders.append(self.ls)
        self.ts = TriangleShader(self)
        self.shaders.append(self.ts)

        self.render_type = 2

        @ti.materialize_callback
        @ti.kernel
        def init_engine():
            for i in ti.static(range(4)):
                self.matrix_perspective[None][i, 0] = self.temp[i][0]
                self.matrix_perspective[None][i, 1] = self.temp[i][1]
                self.matrix_perspective[None][i, 2] = self.temp[i][2]
                self.matrix_perspective[None][i, 3] = self.temp[i][3]
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
        mat_lookat = self.lookat(eye=self.camera)
        for i in range(4):
            self.matrix_lookat[None][i, 0] = mat_lookat[i, 0]
            self.matrix_lookat[None][i, 1] = mat_lookat[i, 1]
            self.matrix_lookat[None][i, 2] = mat_lookat[i, 2]
            self.matrix_lookat[None][i, 3] = mat_lookat[i, 3]
        self.apply()

    @ti.kernel
    def apply(self):
        self.matrix_fin[None] = self.matrix_perspective[None]@self.matrix_lookat[None]

    def lookat(self, eye, center=(0, 0, 0), up=(0, 1, 1e-12)):
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
        return mapply_pos(self.mat, pos)

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

    def add_model(self, model, render_type=0):
        shader = self.shaders[render_type]
        if render_type == 2:
            shader.set_texture(Texture(model.name))
        self.models[model] = namespace(shader=shader)

    def render(self):
        self.image.fill(self.background_color)
        self.clear_depth()
        for model, oinfo in self.models.items():
            self.shaders[self.render_type].set_model(model)
            self.shaders[self.render_type].render()
            # oinfo.shader.set_model(model)
            # oinfo.shader.render()

    def change_render_type(self):
        # self.render_type = 'point' if self.render_type == 'triangle' else 'triangle'
        self.render_type = 2 if self.render_type == 0 else 0

    def show(self, save_file=None):
        gui = ti.GUI('Taichi Renderer', self.res)
        # gui.button('hello', self.change_render_type())
        while gui.running:
            self.set_camera()
            self.render()
            gui.set_image(self.image)
            gui.show(save_file)
            for e in gui.get_events():
                if e.type == gui.PRESS:
                    if e.key == gui.LEFT:
                        self.camera[0] -= 1
                    elif e.key == gui.RIGHT:
                        self.camera[0] += 1
                    elif e.key == gui.UP:
                        self.camera[1] += 1
                    elif e.key == gui.DOWN:
                        self.camera[1] -= 1
                    elif e.key == gui.SPACE:
                        self.render_type = 2 if self.render_type == 0 else 0
                    self.set_camera()
                elif e.type == gui.MOTION:
                    if e.key == gui.WHEEL:
                        delta = e.delta[1] / 120
                        self.camera[2] -= delta
                        self.set_camera()



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
