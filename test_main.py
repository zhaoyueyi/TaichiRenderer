# @Time : 2021/4/25 0:52
# @Author : 赵曰艺
# @File : test_main.py
# @Software: PyCharm
# coding:utf-8

import taichi as ti
from taichi_renderer import TiRenderer
from tr_model import TRModel

def main():
    # init
    ti.init(arch=ti.cpu)
    renderer = TiRenderer()
    model = TRModel('obj/african_head/african_head')
    model2 = TRModel('obj/african_head/african_head_eye_inner')
    # model = TRModel('obj/boggie/body')
    # model2 = TRModel('obj/boggie/head')
    renderer.add_model(model, render_type=2)
    renderer.add_model(model2, render_type=2)
    # renderer.add_model(model2, render_type='triangle')
    renderer.show()


if __name__ == '__main__':
    main()
