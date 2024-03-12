#coding:utf-8
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
# #设置字体为楷体
# matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
def cal_Accuracy(true_label, pred_label, class_num):
    C,c1 = np.zeros((class_num + 1, class_num + 1)), confusion_matrix(true_label.reshape(-1), pred_label.reshape(-1))  # (7,7)
    M, N = 0, np.sum(np.sum(c1, axis=1))
    # 混淆矩阵
    C[0:class_num, 0:class_num], C[0:class_num, class_num], C[class_num, 0:class_num], C[class_num, class_num] = c1, np.sum(c1, axis=1), np.sum(c1, axis=0), N
    #结果
    every_class = np.zeros((class_num + 3,))
    for i in range(class_num):
        acc = C[i, i] / C[i, class_num]
        M = M + C[class_num, i] * C[i, class_num]
        every_class[i] = acc

    kappa = (N * np.trace(C[0:class_num, 0:class_num]) - M) / (N * N - M)
    OA = np.trace(C[0:class_num, 0:class_num]) / N
    AA = np.sum(every_class, axis=0) / class_num
    every_class[class_num] = OA # PA
    every_class[class_num + 1] = AA # MPA
    every_class[class_num + 2] = kappa
    return every_class, C

def visualize_conf_matrix(confusion_mat, n_classes, dice):
    conf_matrix = confusion_mat[0:n_classes, 0:n_classes]

    if n_classes == 5:
        labels = [ u'背景', u'核膜', u'核仁',u'细胞膜',u'细胞质基质']  # 每种类别的标签
    # labels = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6']
    else:
        labels = [u'细胞背景', u'细胞']
    # 显示数据
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(dpi=300, figsize=(9, 5))
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)

    # 在图中标注数量/概率信息
    thresh = conf_matrix.max() / 2  # 数值颜色阈值，如果数值超过这个，就颜色加深。
    for x in range(n_classes):
        for y in range(n_classes):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            info = int(conf_matrix[y, x])
            plt.text(x, y, info,
                     verticalalignment='center',
                     horizontalalignment='center',
                     color="white" if info > thresh else "black")

    plt.tight_layout()  # 保证图不重叠
    plt.yticks(range(n_classes), labels)
    plt.xticks(range(n_classes), labels, rotation=15)  # X轴字体倾斜45°
    plt.subplots_adjust(bottom=0.1)

    plt.savefig('./result/conf_matrix_show_{}分类_dice{}.jpg'.format(n_classes, dice))
    plt.show()
    plt.close()