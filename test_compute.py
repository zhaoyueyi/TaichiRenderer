# @Time : 2021/4/19 15:03 

# @Author : 赵曰艺

# @File : test_compute.py 

# @Software: PyCharm

# coding:utf-8
# import math
import numpy
import time

def main():
    start = time.perf_counter()
    for i in range(1000):
        i += 0.25
        j = numpy.abs(i)  #
        # j = math.fabs(i)  # 0.00018479999999998498
        # j = abs(i)  # 0.00010270000000001112 9.9600000000033e-05
    end = time.perf_counter()
    print(end - start)

if __name__ == '__main__':
    main()
