# 读取train_loss.npy文件并绘制图像
import matplotlib.pyplot as plt
import numpy as np

n_classes = 5
def plot_loss():
    y1 = []
    y2 = []
    enc = np.load('/SwinFusionUnet/checkpoint_U6/train_losss_UC.npy')
    y1 = enc.tolist()
    enc = np.load('/SwinFusionUnet/checkpoint_U6/test_losss_UC.npy')
    y2 = enc.tolist()
    x = list(range(1, 101))

    fig, ax1 = plt.subplots()  # 创建图形和第一个y轴
    ax1.plot(x, y1, 'b-', label="train loss")
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('train loss')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()  # 创建第二个y轴
    ax2.plot(x, y2, 'r-', label="val loss")
    ax2.set_ylabel('test loss', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.grid()
    plt_title = 'BATCH_SIZE = 4; LEARNING_RATE:1e-5'
    plt.title(plt_title)

    # n_classes = input("Please enter n_classes: ")
    plt.savefig("image/loss_{}分类.jpg".format(n_classes))
    plt.show()


def plot_dice():
    y1 = []
    y2 = []
    enc = np.load('/SwinFusionUnet/checkpoint_U6/train_dices_UC.npy')
    y1 = enc.tolist()
    enc = np.load('/SwinFusionUnet/checkpoint_U6/test_dices_UC.npy')
    y2 = enc.tolist()
    x = list(range(1, 101))

    fig, ax1 = plt.subplots()
    ax1.plot(x, y1, 'b-', label="train dice")
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('train dice')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(x, y2, 'r-', label="test dice")
    ax2.set_ylabel('test dice', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.grid()
    plt_title = 'BATCH_SIZE = 4; LEARNING_RATE:1e-5'
    plt.title(plt_title)
    # plt.ylabel(input("Please enter the name of Y-axis: "))
    plt.legend()
    plt.savefig("image/dice_{}分类.jpg".format(n_classes))
    plt.show()

def plot_acc():
    y1 = []
    y2 = []
    enc = np.load('/SwinFusionUnet/checkpoint_U6/acc0s_UC.npy')
    y1 = enc.tolist()
    enc = np.load('/SwinFusionUnet/checkpoint_U6/acc1s_UC.npy')
    y2 = enc.tolist()
    x = list(range(1, 101))

    fig, ax1 = plt.subplots()
    ax1.plot(x, y1, 'b-', label="train dice")
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('acc0s')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(x, y2, 'r-', label="test dice")
    ax2.set_ylabel('acc1s', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.grid()
    plt_title = 'BATCH_SIZE = 4; LEARNING_RATE:1e-5'
    plt.title(plt_title)
    plt.legend()
    plt.savefig("image/acc_{}分类.jpg".format(n_classes))
    plt.show()


def plot_Kappas():
    # plt.figure(figsize=(8, 7))  # 窗口大小可以自己设置
    y1 = []
    y2 = []
    enc = np.load('/SwinFusionUnet/checkpoint_U6/Kappas_UC.npy')  # 文件返回数组
    y1 = enc.tolist()  # 转列表，但是列表内的元素的str格式的，如果plt画图不能直接用，系统会默认将str进行ASCII转换的。
    # enc = np.load('F:/MyCode/SwinFusionUnet/checkpoint_U6/OAs_UC.npy')
    # y2 = enc.tolist()
    # tempy = float(tempy)  # str转数值
    # y1.append(tempy)  # 由后放入y1中
    x1 = list(range(1, 101))

    plt.figure()  # 使用显示
    plt.plot(x1, y1, 'b-', label="train dice")
    # plt.plot(x1, y2, 'r-', label="test dice")
    plt.grid()  # 显示网格
    plt_title = 'BATCH_SIZE = 2; LEARNING_RATE:1e-5'
    plt.title(plt_title)  # 标题名
    plt.xlabel('epoch')  # 横坐标名
    plt.ylabel('Kappas', color='r') # 纵坐标名
    plt.legend()  # 显示曲线信息
    # n_classes = input("Please enter n_classes: ")
    plt.savefig("image/Kappas_{}分类.jpg".format(n_classes))  # 当前路径下保存图片名字
    plt.show()

def plot_AAOA():
    y1 = []
    y2 = []
    enc = np.load('/SwinFusionUnet/checkpoint_U6/AAs_UC.npy')
    y1 = enc.tolist()
    enc = np.load('/SwinFusionUnet/checkpoint_U6/OAs_UC.npy')
    y2 = enc.tolist()
    x = list(range(1, 101))

    fig, ax1 = plt.subplots()
    ax1.plot(x, y1, 'b-', label="train dice")
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('AA')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(x, y2, 'r-', label="test dice")
    ax2.set_ylabel('OA', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.grid()
    plt_title = 'BATCH_SIZE = 4; LEARNING_RATE:1e-5'
    plt.title(plt_title)
    # plt.ylabel(input("Please enter the name of Y-axis: "))
    plt.legend()
    plt.savefig("image/AA_OA_{}分类.jpg".format(n_classes))
    plt.show()

if __name__ == "__main__":
    # file = input("Please enter npy file's name: ")
    plot_loss()  # 文件数量
    plot_dice()
    plot_acc()
    plot_Kappas()
    plot_AAOA()

    # plot_dice()





