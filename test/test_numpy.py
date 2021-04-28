# @Time : 2021/4/18 20:52 

# @Author : 赵曰艺

# @File : test_numpy.py 

# @Software: PyCharm

# coding:utf-8
matrix_image = matrix_image.swapaxes(1, 0)  # 矩阵转置

# # tkinter
# window = tk.Tk()
# tk_image = tk.PhotoImage(file='output.tga')
# lable = tk.Label(window, image=tk_image)
# lable.pack()
# window.mainloop()


# # cv2
# matrix_image = np.full((100, 100, 4), 0)
# # matrix_image = np.arange([0, 0, 0, 255]).reshape(100, 100)
# print(matrix_image)
# TGA_image = cv2.cvtColor(matrix_image, cv2.COLOR_RGBA2BGRA)
# cv2.imshow('test', TGA_image)
# # TGA_image = cv2.resize(matrix_image, (100, 100))

# image = pyTGA.Image(data=TGA_image)
# image.save("output.tga")
# matrix_image[1][1] = TGA_red

# matrix_image = Image.new('RGBA', (100, 100), (0, 0, 0, 255))
# matrix_image.save("output.tga")
# matrix_image = np.asarray(matrix_image)
# matrix_image.flags.writeable = True
# line(13, 20, 80, 40, matrix_image, TGA_white);  # 画线