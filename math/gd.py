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
    # # 初始点
    # x0 = np.array([2, 2])
    # # 求解的函数
    # fx = lambda x: x[0]**2+x[1]**2  # lambda匿名函数，在ep8后就丢弃了
    # # 梯度
    # dfx = lambda x: np.array([2*x[0], 2*x[1]])
    # # 调用类函数
    # result = MySGD().solve(x0, fx, dfx)
    # print('\n min value %s => %.9f' % (result[0], result[1]))
    Class = 1
    print(Class)
    eval("12+23")
    e, f = 1, 2
    a = b = c =1
    # I = ["i","love",'HCT']
    # for k,v in enumerate(I):
    #     if k == 1:
    #         continue
    #     print(v,end="")

    I = ["i", "love", 'python','a','little','more']
    first,*middle,last= I
    print(first)
    print(*middle)
    print(last)
    print(len(list(middle)))

    result = ''
    for s in I:
        result +=s
    print(result)
    print("i\'m in HCT.")
    print(r'i\'m in HCT.')
    print("""i'm in HCT.""")


    print(1.2-1.0 == 0.2)
    print(1)
    a =(1,2)
    print(type(a))
    dict = {'1':1,'2':2}
    the = dict.copy()
    dict["1"]=5
    print(dict['1']+the['1'])
    a = [i for i in map(lambda x,y:x+y,['hui','hc','Artifical'],['ke','tech','intelligence'])]
    print(a)

    alist =[{'name':'a','age':20},{'name':'b','age':30},{'name':'c','age':25}]
    # alist.sort(key=lambda x:x['age'],reverse=True)
    # print(alist)
    alist= sorted(alist,key=lambda x:x['age'],reverse=True)
    print(alist)
