# -*- coding: utf-8 -*-

'''
@Time    : 2020/4/21 22:47
@Author  : HHNa
@FileName: xianxingdaishu.py
@Software: PyCharm
 
'''
import numpy as np

a = np.array([[1, 1], [0, 1]])
b = np.array([[2, 0], [3, 4]])

# a 与 b的矩阵乘法
multiply = a.dot(b)
print(multiply)
# [[5 4]
#  [3 4]]

subtract = a - b
print(subtract)
# [[-1  1]
#  [-3 -3]]

# *乘积运算符 表示是NumPy数组中按元素进行运算
e = a * b
print(e)
# [[2 0]
#  [0 4]]

# 矩阵的转置
f = np.transpose(subtract)
print(f)
# [[-1 -3]
#  [ 1 -3]]

'''求 a 的逆'''

a = np.array([[1, 1, 1], [0, 2, 5], [2, 5, -1]])
print('数组 a：')
print(a)
# [[ 1  1  1]
#  [ 0  2  5]
#  [ 2  5 -1]]

ainv = np.linalg.inv(a)
print('a 的逆：')
print(ainv)
# [ 1.28571429 -0.28571429 -0.14285714]
#  [-0.47619048  0.14285714  0.23809524]
#  [ 0.19047619  0.14285714 -0.0952381 ]]

# 矩阵对象可以通过 .I 更方便的求逆
A = np.matrix(a)
print('通过矩阵求 a 的逆：')
print(A.I)

'''特征值和特征向量'''
# 定义矩阵
A = np.mat("1 4 2; 0 -3 4; 0 4 3")
print(A)
# [[ 1  4  2]
#  [ 0 -3  4]
#  [ 0  4  3]]

b = np.array([0, 8, -9])

# 线性方程组求解
x = np.linalg.solve(A, b)
print(x)
# [ 9.2 -2.4  0.2]

# 返回矩阵的特征值
y = np.linalg.eigvals(A)
print(y)
# [ 1. -5.  5.]

# 返回矩阵的特征值和特征向量的元组
(w, v) = np.linalg.eig(A)
print(w)
print(v)
# [ 1. -5.  5.]
# [[ 1.          0.40824829 -0.66666667]
#  [ 0.         -0.81649658 -0.33333333]
#  [ 0.          0.40824829 -0.66666667]]


'''奇异值分解svd'''

A = np.array([[1, 2, 3], [4, 5, 6]])
U, sigma, VT = np.linalg.svd(A)
print(U)
# [[-0.3863177  -0.92236578]
#  [-0.92236578  0.3863177 ]]
print(sigma)
# [9.508032   0.77286964]
print(VT)
# [[-0.42866713 -0.56630692 -0.7039467 ]
#  [ 0.80596391  0.11238241 -0.58119908]
#  [ 0.40824829 -0.81649658  0.40824829]]

