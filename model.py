# @Time : 2021/4/18 21:55
# @Author : 赵曰艺
# @File : model.py
# @Software: PyCharm
# coding:utf-8
import re
from PIL import Image
import numpy as np

class Model:
    # private
    __verts = []  # array of vertices
    __uv    = []  # array of tex coords
    __norms = []  # array of normal vectors
    __facet_vrt = []  #
    __facet_tex = []  # indices in the above arrays per triangle
    __facet_nrm = []  #

    __diffusemap:Image  = None  # diffuse color texture
    __normalmap:Image   = None  # normal map texture
    __specularmap:Image = None  # specular map texture

    def __init__(self, filename:str):
        with open(filename+'.obj', mode='r') as file_in:
            list = file_in.readlines()
            file_in.close()
            for i in list:
                i = i.strip('\n')
                values = re.split(r"[ ]+", i)
                if values[0] == 'v':
                    del values[0]
                    values[0] = float(values[0])
                    values[1] = float(values[1])
                    values[2] = float(values[2])
                    self.__verts.append(values)
                elif values[0] == 'vn':
                    del values[0]
                    values[0] = float(values[0])
                    values[1] = float(values[1])
                    values[2] = float(values[2])
                    self.__norms.append(values)
                elif values[0] == 'vt':
                    del values[0]
                    values[0] = float(values[0])
                    values[1] = float(values[1])
                    values[2] = float(values[2])
                    self.__uv.append(values)
                elif values[0] == 'f':
                    del values[0]
                    cnt:int = 0
                    for facet in values:
                        props = facet.split('/')
                        props[0] = int(props[0])-1
                        props[1] = int(props[1])-1
                        props[2] = int(props[2])-1
                        self.__facet_vrt.append(props[0])
                        self.__facet_tex.append(props[1])
                        self.__facet_nrm.append(props[2])
                        cnt += 1
                    if 3 != cnt:
                        exit(1)
        self.__diffusemap  = self.__load_texture(filename, '_diffuse.tga')
        self.__normalmap   = self.__load_texture(filename, '_nm_tangent.tga')
        self.__specularmap = self.__load_texture(filename, '_spec.tga')

        # test
        # print(self.__verts)
        # print(self.__norms)
        # print(self.__uv)
        # print(self.__facet_vrt)
        # print(self.__facet_nrm)
        # print(self.__facet_tex)
        # self.__normalmap.show()
        # self.__diffusemap.show()
        # self.__specularmap.show()

    def nverts(self):
        return len(self.__verts)

    def faces(self):
        return np.reshape(self.__facet_vrt, (-1, 3))

    def nfaces(self):
        return len(self.__facet_vrt)/3

    def vert(self, i:int):
        return self.__verts[i]

    def vert2(self, iface:int, nthvert:int):
        return self.__verts[self.__facet_vrt[iface*3+nthvert]]

    def __load_texture(self, filename:str, suffix:str):
        return Image.open(filename+suffix).transpose(Image.FLIP_TOP_BOTTOM)

    def diffuse(self, uvf):
        pass

    def normal(self, uvf):
        pass

    def specular(self, uvf):
        pass

    def uv(self, iface:int, nthvert:int):
        return self.__uv[self.__facet_tex[iface*3+nthvert]]

    def normal(self, iface:int, nthvert:int):
        return self.__norms[self.__facet_nrm[iface*3+nthvert]]

def main():
    i = Model('obj/african_head/african_head')


if __name__ == '__main__':
    main()