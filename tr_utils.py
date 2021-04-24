# @Time : 2021/4/24 23:42
# @Author : 赵曰艺
# @File : ti_utils.py
# @Software: PyCharm
# coding:utf-8
import numpy as np

def read_obj(filename: str):
    verts      = []
    verts_tex  = []
    verts_norm = []
    faces      = []
    obj = {}

    with open(filename + '.obj', mode='r') as file_in:
        lines = file_in.readlines()
        file_in.close()
        for line in lines:
            try:
                type, values = line.split(maxsplit=1)
                values = [float(_) for _ in values.split()]
            except ValueError:
                continue
            if type == b'v':
                verts.append(values)
            elif type == b'vn':
                verts_norm.append(values)
            elif type == b'vt':
                verts_tex.append(values)
            elif type == b'f':
                for facet in values:
                    props = facet.split('/')
                    props[0] = int(props[0]) - 1
                    props[1] = int(props[1]) - 1
                    props[2] = int(props[2]) - 1
                    faces.append(props)

    obj['v'] = np.array([[0, 0, 0]], dtype=np.float32) if len(verts) == 0 else np.array(vert, dtype=np.float32)
    obj['vt'] = np.array([[0, 0]], dtype=np.float32) if len(verts_tex) == 0 else np.array(vert_tex, dtype=np.float32)
    obj['vn'] = np.array([[0, 0, 0]], dtype=np.float32) if len(verts_norm) == 0 else np.array(vert_norm, dtype=np.float32)
    obj['f'] = np.zeros((1, 3, 3), dtype=np.int32) if len(faces) == 0 else np.array(faces, dtype=np.int32)
    obj['name'] = filename

    return obj
