# -*- coding: utf-8 -*-

'''
@Time    : 2020/5/4 23:20
@Author  : HHNa
@FileName: sgd+bgd.py
@Software: PyCharm
 
'''
# -*- coding: utf-8 -*-

'''
@Time    : 2020/5/4 16:51
@Author  : HHNa
@FileName: gd.py
@Software: PyCharm

'''

import numpy as np
import matplotlib.pyplot as plt
import random
import datetime


def bgd(samples, y, step_size=0.01, max_iteration_count=10000):
    """
    批梯度下降法Batch Gradient Descent
    :param samples: 样本
    :param y: 结果
    :param step_size: 每一接迭代的步长
    :param max_iteration_count: 最大的迭代次数
    :return:
    """
    sample_num, dimension = samples.shape
    w = np.ones((dimension, 1), dtype=np.float32)
    loss_collection = []
    loss = 1
    iteration_count = 0
    #  当loss大于阈值并且迭代次数小于最大迭代次数时进行迭代
    while loss > 0.001 and iteration_count < max_iteration_count:
        loss = 0
        gradient = np.zeros((dimension, 1), dtype=np.float32)
        #  计算（批）梯度
        for i in range(sample_num):
            predict_y = np.dot(w.T, samples[i])
            for j in range(dimension):
                gradient[j] += (predict_y - y[i]) * samples[i][j]
        #  更新权重
        for j in range(dimension):
            w[j] -= step_size * gradient[j]
        #  计算当前loss值
        for i in range(sample_num):
            predict_y = np.dot(w.T, samples[i])
            loss += np.power((predict_y - y[i]), 2)
        #  将loss存储到链表里，以便后续画图
        loss_collection.append(loss)
        iteration_count += 1
    return w, loss_collection


def sgd(samples, y, step_size=0.001, max_iteration_count=10000):
    """
    随机梯度下降法Stochastic Gradient Descent
    :param samples: 样本
    :param y: 结果
    :param step_size: 每一接迭代的步长
    :param max_iteration_count: 最大的迭代次数
    :return:
    """
    sample_num, dimension = samples.shape
    w = np.ones((dimension, 1), dtype=np.float32)
    loss_collection = []
    loss = 1
    iteration_count = 0
    while loss > 0.001 and iteration_count < max_iteration_count:
        loss = 0
        gradient = np.zeros((dimension, 1), dtype=np.float32)
        #  不同于BGD的是，这里随机取一个样本进行权重更新
        sample_index = random.randint(0, sample_num - 1)
        predict_y = np.dot(w.T, samples[sample_index])
        for j in range(dimension):
            gradient[j] += (predict_y - y[sample_index]) * samples[sample_index][j]
            w[j] -= step_size * gradient[j]

        for i in range(sample_num):
            predict_y = np.dot(w.T, samples[i])
            loss += np.power((predict_y - y[i]), 2)

        loss_collection.append(loss)
        iteration_count += 1
    return w, loss_collection


if __name__ == '__main__':
    samples = np.array([[1, 2, 5, 4],
                        [2, 5, 1, 2]]).T
    y = np.array([19, 26, 19, 20]).reshape((4, 1))
    #  当前时间
    time = datetime.datetime.now()
    bgd_w, bgd_loss_collection = bgd(samples, y, 0.001, 1000)
    #  经过BGD后的时间
    time_afterBGD = datetime.datetime.now()
    sgd_w, sgd_loss_collection = sgd(samples, y, 0.001, 1000)
    #  经过SGD后的时间
    time_afterSGD = datetime.datetime.now()
    # 画出loss走向图
    epochs = range(1, len(bgd_loss_collection) + 1)
    plt.plot(epochs, bgd_loss_collection, 'b', label='BGD loss')
    plt.plot(epochs, sgd_loss_collection, 'r', label='SGD loss')
    plt.title('GD and SGD loss')
    plt.legend()
    plt.savefig('GD and SGD loss.png', dpi=300)
    #  打印出相关结果
    print('bgd_w:', bgd_w)
    print('bgd predict_y:', np.dot(bgd_w.T, samples.T))
    print('bgd_time:', (time_afterBGD - time).total_seconds(), 's')
    print('sgd_w:', sgd_w)
    print('sgd predict_y:', np.dot(sgd_w.T, samples.T))
    print('sgd_time:', (time_afterSGD - time_afterBGD).total_seconds(), 's')