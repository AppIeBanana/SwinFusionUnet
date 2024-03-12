import matplotlib.pyplot as plt
import PIL
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import dataset


def show_gray_img(y, y_, windows_size, n_classes, dice):
    idxs = []
    if n_classes == 2:
        # 这五张图像是先输出一下所有（150张）图片，然后挑选一些质量好的 用数组存起来，然后读取数组展示质量比较好的。
        # idxs = [2, 3, 5, 12, 16]  # 2分类
        idxs = [1, 2, 3, 4, 5]
    if n_classes == 7:
        # idxs = [0, 3, 7, 11, 19]
        idxs = [1, 2, 3, 4, 5]

    plt.figure()
    for idx in range(len(idxs)):
        # 先reshape再astype(type): returns a copy of the array converted to the specified type.
        img_y = y[idxs[idx]].reshape(windows_size, windows_size).astype(int)
        img_y_ = y_[idxs[idx]].reshape(windows_size, windows_size).astype(int)
        for i in range(img_y.shape[0]):
            for j in range(img_y.shape[1]):
                # 六分类 分别用不同颜色（img_y[i][j]的数值）表示
                # img_y[i][j] == 0:属于背景
                if img_y[i][j] == 0:
                    img_y[i][j] = 0
                elif img_y[i][j] == 1:
                    img_y[i][j] = 15
                elif img_y[i][j] == 2:
                    img_y[i][j] = 38
                elif img_y[i][j] == 3:
                    img_y[i][j] = 53
                elif img_y[i][j] == 4:
                    img_y[i][j] = 75
                elif img_y[i][j] == 5:
                    img_y[i][j] = 90
                elif img_y[i][j] == 6:
                    img_y[i][j] = 113


        # plt.subplot(nrows, ncols, index)
        # 位置由三个整型数值构成：第一个代表行数，第二个代表列数，第三个代表索引位置。
        plt.subplot(2, 5, idx + 1)
        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值
        # astype(type): returns a copy of the array converted to uint8
        img_y = img_y.astype(np.uint8)
        # 实现array到image的转换
        img = PIL.Image.fromarray(img_y)
        # cmap: 颜色图谱（colormap), 默认绘制为RGB(A)颜色空间。
        plt.imshow(img, cmap=plt.cm.terrain)

        plt.subplot(2, 5, idx + len(idxs)+1)
        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值
        img_y_ = img_y_.astype(np.uint8)
        img2 = PIL.Image.fromarray(img_y_)
        plt.imshow(img2, cmap=plt.cm.terrain)
    plt.savefig("./result/Unet_rgb_{}分类_dice{}.jpg".format(n_classes, dice))
    plt.show()

def RGB(img, i, j):
    # （a, b, c)分别代表RGB颜色中红绿蓝（这三不确定顺序）的色度，三者共同决定最后的颜色/
    if img[i][j] == 0:
        return (0, 0, 0)
    if img[i][j] == 1:
        return (107, 0, 1)
    if img[i][j] == 2:
        return (17, 112, 3)  # background
    if img[i][j] == 3:
        return (16, 110, 109)
    if img[i][j] == 4:
        return (109, 111, 3)  # 细胞膜
    if img[i][j] == 5:
        return (-0, 0, 109)  # 细胞质基质
    if img[i][j] == 6:
        return (107, 0, 109)
    return (0, 0, 0)

def show_rgb_img(y, y_, windows_size, n_classes, dice):
    rgb_img_y, rgb_img_y_ = np.zeros((windows_size, windows_size, 3)), np.zeros((windows_size, windows_size, 3))
    # 展示的图像为空，因为上面定义的是np.zeros()
    plt.figure()
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
    idxs = []
    if n_classes == 2:
        # idxs = [2, 3, 5, 12, 16] #2分类
        idxs = [1, 2, 3, 4, 5]
    if n_classes == 7:
        # idxs = [0, 3, 7, 11, 19]
        idxs = [1, 2, 3, 4, 5]
    # tight_layout会自动调整子图参数,使之填充整个图像区域
    plt.tight_layout()
    for idx in range(len(idxs)):
        # len(idxs)=5
        # astype(type): returns a copy of the array converted to the specified type.
        img_y = y[idxs[idx]].reshape(windows_size, windows_size).astype(int)
        img_y_ = y_[idxs[idx]].reshape(windows_size, windows_size).astype(int)
        for i in range(img_y.shape[0]): # 224
            for j in range(img_y.shape[1]): # 224
                rgb_img_y[i][j], rgb_img_y_[i][j] = RGB(img_y, i, j), RGB(img_y_, i, j)

        # plt.subplot(nrows, ncols, index)
        # 位置由三个整型数值构成：第一个代表行数，第二个代表列数，第三个代表索引位置。
        plt.subplot(2, 5, idx + 1)
        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值
        img = PIL.Image.fromarray(np.uint8(rgb_img_y))
        img = img.convert('RGB')
        plt.imshow(img)

        plt.subplot(2, 5, idx + 6)
        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值
        img2 = PIL.Image.fromarray(np.uint8(rgb_img_y_))
        img2 = img2.convert("RGB")
        plt.imshow(img2)
    plt.show()
    # plt.savefig("./result/Unet_rgb_{}分类_dice{}.jpg".format(n_classes, dice))


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    batch_size = 2
    n_classes = 2
    test_loader = DataLoader(dataset=dataset.genDataSet("val", n_classes), batch_size=batch_size,shuffle=True)
    test_loss, test_dice, all_lab, all_pre = 0.0, 0, [], []
    all_x = []
    for idx, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):
        x = x.squeeze(1).float()
        y = y.squeeze(1).long()
        y_ = y
        all_lab.append(y.view(batch_size, -1).detach().cpu().numpy())
        all_pre.append(y_.view(batch_size, -1).detach().cpu().numpy())
        all_x.append(x.view(batch_size, -1).detach().cpu().numpy())
    all_Y = np.concatenate(all_lab)
    print("True lbl:", np.unique(all_Y))
    all_Y_ = np.concatenate(all_pre)
    # show_rgb_img(x)
    show_rgb_img(all_Y, all_Y_, 224, n_classes, test_dice)
    show_gray_img(all_Y, all_Y_, 224, n_classes, test_dice)
    # show_rgb_data(all_x, all_x, 224, n_classes, test_dice)