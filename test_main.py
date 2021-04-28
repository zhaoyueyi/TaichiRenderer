# @Time : 2021/4/25 0:52
# @Author : 赵曰艺
# @File : test_main.py
# @Software: PyCharm
# coding:utf-8
# @Time : 2021/4/21 12:05
# @Author : 赵曰艺
# @File : taichi_renderer.py
# @Software: PyCharm
# coding:utf-8
import taichi as ti
from taichi_renderer import TiRenderer
from tr_model import TRModel

def main():
    # init
    ti.init(arch=ti.cpu, _test_mode=True)
    renderer = TiRenderer()
    model = TRModel('obj/african_head/african_head')
    renderer.add_model(model)
    renderer.show('output.png')


if __name__ == '__main__':
    main()
