import numpy as np

# p在当前的数据集下，标签中一共有四类，
# 0代表background，1，2，3分别代表图像每个像素的预测分类，比如是人、车、树。
# predict

# pre_image的前一步是分割网络的输出。那么会是一个4×4×4的矩阵，
# 针对每个像素是一个1×1×4的向量，就是对于这个像素的分类结果输出四个概率，
# 那么我们取预测概率最大的那个结果作为当前像素的预测分类。
# 从而就从一个4×4×4的预测概率的三维输出降维成了4×4×1。


''' 
    得到混淆矩阵
'''
pre_image = np.array([[0, 1, 0],
                     [2, 1, 0],
                     [2, 2, 1]])

# groudtruth
gt_image = np.array([[0, 2, 0],
                     [2, 1, 0],
                     [1, 2, 1]])

num_class = 3
confusion_matrix = np.zeros((num_class,)*2)
# 把255 和 负数去掉
mask = (gt_image >= 0) & (gt_image < num_class)

label = num_class * gt_image[mask].astype('int') + pre_image[mask]

print(label)
count = np.bincount(label, minlength=num_class**2)
print(count)
confusion_matrix = count.reshape(num_class, num_class)
print(confusion_matrix)


# 预测正确的像素个数/图像像素大小
pa = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
# 混淆矩阵的对角线元素之和/图像像素大小
print(pa)

# step1:每一类别的预测准确率
Acc = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
# 混淆矩阵对角线元素除以
print(Acc)
# step2:7类准确率的平均值
mpa = np.nanmean(Acc)
print(mpa)



intersection = np.diag(confusion_matrix)
union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, 0) - np.diag(confusion_matrix)
IoU = intersection / union  # 返回列表，其值为各个类别的IoU
print('Iou', IoU)
mIoU = np.nanmean(IoU) # 求各类别IoU的平均
print('miou', mIoU)


''' 
    关于对bincount的使用
'''
w = np.array([0.3, 0.5, 0.2, 0.7, 1., -0.6])
x = np.array([2, 1, 3, 4, 4, 3])
# 没有参数的时候
print(np.bincount(x))
# 没有参数的时候
print(np.bincount(x, w))
# 本来bin的数量为4，现在我们指定了参数为7，因此现在bin的数量为7，所以现在它的索引值为0->6
print(np.bincount(x, minlength=7))