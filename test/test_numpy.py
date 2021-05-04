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








Vec3i t0, Vec3i t1, Vec3i t2,  # 模型顶点
Vec2i uv0, Vec2i uv1, Vec2i uv2,  # 纹理坐标顶点

int total_height = t2.y-t0.y;  # 两顶点高度差
for (int i=0; i<total_height; i++) {  # 遍历高度
bool second_half = i>t1.y-t0.y || t1.y==t0.y;  #
int segment_height = second_half ? t2.y-t1.y : t1.y-t0.y;

float alpha = (float)i / total_height;
float beta = (float)(i - (second_half ? t1.y-t0.y: 0)) / segment_height; # be careful:with above conditions no division by zero here

Vec3i A   =               t0  + Vec3f(t2-t0  )*alpha;
Vec3i B   = second_half ? t1  + Vec3f(t2-t1  )*beta : t0  + Vec3f(t1-t0  )*beta;

Vec2i uvA =               uv0 +      (uv2-uv0)*alpha;
Vec2i uvB = second_half ? uv1 +      (uv2-uv1)*beta : uv0 +      (uv1-uv0)*beta;

for (int j=A.x; j <= B.x; j++)
float phi = B.x==A.x ? 1. : (float)(j-A.x)/(float)(B.x-A.x);

Vec2i uvP =     uvA +   (uvB-uvA)*phi;
TGAColor color = model->diffuse(uvP);

TGAColor Model::diffuse(Vec2i uv) {
    return diffusemap_.get(uv.x, uv.y);
}