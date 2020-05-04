# -*- coding: utf-8 -*-

'''
@Time    : 2020/5/4 16:51
@Author  : HHNa
@FileName: gd.py
@Software: PyCharm

'''
import numpy as np
np.set_printoptions(precision=6, suppress=True)


class MySGD:
    def __init__(self, eta=0.01, max_iter=10000, tol=1e-8):
        self.eta = eta  # 学习率
        self.max_iter = max_iter  # 迭代次数
        self.tol = tol

    def solve(self, x0, fx, dfx):
        for n in range(self.max_iter):
            xn = x0 - self.eta*dfx(x0)
            error = np.linalg.norm(x0-xn)  # np.linalg.norm(求范数)
            if n % 100 == 0:
                print('iter:%d x0:%s fx:%.6f xn:%s error:%.9f' % (n, x0, fx(x0), xn, error))
            if error < self.tol:
                break
            x0 = xn
        print('iter:%d x0:%s fx:%0.6f xn:%s error:%.9f' % (n, x0, fx(x0), xn, error))
        return x0, fx(x0)


if __name__ == "__main__":
    # 初始点
    x0 = np.array([2, 2])
    # 求解的函数
    fx = lambda x: x[0]**2+x[1]**2  # lambda匿名函数，在ep8后就丢弃了
    # 梯度
    dfx = lambda x: np.array([2*x[0], 2*x[1]])
    # 调用类函数
    result = MySGD().solve(x0, fx, dfx)
    print('\n min value %s => %.9f' % (result[0], result[1]))
