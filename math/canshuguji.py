# -*- coding: utf-8 -*-

'''
@Time    : 2020/5/2 11:11
@Author  : HHNa
@FileName: canshuguji.py
@Software: PyCharm
'''



def fuction_MLE_python():
    '''
    [(机器学习)概率统计]极大似然估计MLE原理+python实现
    '''
    import numpy as np
    import matplotlib.pyplot as plt

    mu = 30  # mean of distribution
    sigma = 2  # standard deviation of distribution
    x = mu + sigma * np.random.randn(10000)


    def mle(x):
        """
        极大似然估计
        :param x:
        :return:
        """
        u = np.mean(x)
        return u, np.sqrt(np.dot(x - u, (x - u).T) / x.shape[0])


    print(mle(x))
    num_bins = 100
    plt.hist(x, num_bins)
    plt.show()


def function_Python():
    import numpy as np
    from scipy.stats import norm
    import matplotlib.pyplot as plt

    μ = 30  # 数学期望
    σ = 2  # 方差
    x = μ + σ * np.random.randn(10000)  # 正态分布
    plt.hist(x, bins=100)  # 直方图显示
    plt.show()
    print(norm.fit(x))  # 返回极大似然估计，估计出参数约为30和2


'''python简单实现最大似然估计&scipy库的使用'''


def function_scipy_MLE():
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    import numpy as np

    '''
    norm.cdf 返回对应的累计分布函数值
    norm.pdf 返回对应的概率密度函数值
    norm.rvs 产生指定参数的随机变量
    norm.fit 返回给定数据下，各参数的最大似然估计（MLE）值
    '''
    x_norm = norm.rvs(size=200)
    # 在这组数据下，正态分布参数的最大似然估计值
    x_mean, x_std = norm.fit(x_norm)
    print('mean, ', x_mean)
    print('x_std, ', x_std)
    # 归一化直方图（用出现频率代替次数），将划分区间变为 20（默认 10）
    plt.hist(x_norm, normed=True, bins=15)
    x = np.linspace(-3, 3, 50)  # 在在(-3,3)之间返回均匀间隔的50个数字。
    plt.plot(x, norm.pdf(x), 'r-')
    plt.show()


if __name__ == "__main__":
    # python简单实现最大似然估计&scipy库的使用
    # function_scipy_MLE()
    # fuction_MLE_python()
    function_Python()