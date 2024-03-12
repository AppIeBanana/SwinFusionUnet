import hdf5storage
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import scipy.io as sio
import matplotlib.pyplot as plt


# from osgeo.gdal_array import DatasetReadAsArray
# from osgeo import gdal


class MyDataSet(Dataset):
    def __init__(self, img_size, num_classes, flag):
        data1_dir = 'dataset/data1_channel8_{}.npy'.format(img_size)
        data2_dir = 'dataset/data2_channel8_{}.npy'.format(img_size)
        label_dir = 'dataset/label_fusion_class{}_{}.npy'.format(num_classes, img_size)

        # data1_dir = 'dataset/data1_channel64.npy'
        # data2_dir = 'dataset/data2_channel64.npy'
        # label_dir = 'dataset/label_class3.npy'

        data1 = np.load(data1_dir)
        data2 = np.load(data2_dir)
        label = np.load(label_dir)
        # print(data1.shape,data2.shape,label.shape)

        data1 = data1.astype(np.float32)
        data2 = data2.astype(np.float32)
        label = label.astype(np.float32)

        # data1 = np.squeeze(data1)
        # data2 = np.squeeze(data2)
        # label = np.squeeze(label)

        test_size = 0.1
        train_x1, test_x1, train_x2, test_x2 = train_test_split(data1, data2, test_size=test_size, shuffle=False,
                                                                random_state=70)
        train_y, test_y = train_test_split(label, test_size=test_size, shuffle=False, random_state=70)

        if flag == "train":
            self._data1 = train_x1
            self._data2 = train_x2
            self._label = train_y
        elif flag == "val" or "test":
            self._data1 = test_x1
            self._data2 = test_x2
            self._label = test_y
        else:
            raise Exception("dataset must in train val test")

    def __getitem__(self, idx):
        img1 = self._data1[idx]
        img2 = self._data2[idx]
        label = self._label[idx]
        return img1, img2, label

    def __len__(self):
        return len(self._label)


if __name__ == "__main__":
    trainData = MyDataSet("train")
    print(trainData.__len__())

    testData = MyDataSet("test")
    print(trainData.__len__())

    valData = MyDataSet("val")
    print(trainData.__len__())
    # data.shape=(819, 1, 16, 224, 224)

    # Botswana
    # data = sio.loadmat('Botswana/Botswana.mat')
    # label = sio.loadmat('Botswana/Botswana_gt.mat')
    # image_data = data['Botswana']  # H,W,C (1476,256,145)
    # label_data = label['Botswana_gt']  # H,W (1476,256)
    # np.save('Botswana/Botswana.npy', image_data)
    # np.save('Botswana/Botswana_gt.npy', label_data)

    # KSC
    # data = sio.loadmat('KSC/KSC.mat')
    # label = sio.loadmat('KSC/KSC_gt.mat')
    # image_data = data['KSC']  # H,W,C (512,614,176)
    # label_data = label['KSC_gt']  # H,W (512,614)
    # np.save('KSC/KSC.npy', image_data)
    # np.save('KSC/KSC_gt.npy', label_data)

    # Xiong'an
    # data = hdf5storage.loadmat("Xiong'an/xiongan.mat")
    # label = hdf5storage.loadmat("Xiong'an/xiongan_gt.mat")
    # image_data = data['XiongAn']  # H,W,C (1580,3750,256)
    # label_data = label['xiongan_gt']  # H,W (1580,3750)
    # np.save("Xiong'an/xiongan.npy", image_data)
    # np.save("Xiong'an/xiongan_gt.npy", label_data)

    # Xuzhou
    # data = sio.loadmat('Xuzhou/xuzhou.mat')
    # label = sio.loadmat('Xuzhou/xuzhou_gt.mat')
    # image_data = data['xuzhou']  # H,W,C (500,260,436)
    # label_data = label['xuzhou_gt']  # H,W (500,260)
    # np.save("Xuzhou/xuzhou.npy", image_data)
    # np.save("Xuzhou/xuzhou_gt.npy", label_data)

    # Indian_pines
    # data = hdf5storage.loadmat('Indian_pines/indian_pines.mat')
    # data_cor = hdf5storage.loadmat('Indian_pines/indian_pines_large_corrected.mat')
    # data_gt52 = hdf5storage.loadmat('Indian_pines/indian_pines_large_gt52.mat')
    # data_gt58 = hdf5storage.loadmat('Indian_pines/indian_pines_large_gt58.mat')
    # label = hdf5storage.loadmat('Indian_pines/indian_pines_gt.mat')
    # image_data = data['HSI_original']  # H,W,C (145,145,200)
    # image_data_cor = data_cor['indian_pines_large_corrected']  # (1403,614,200)
    # image_data_gt52 = data_gt52['indian_pines_large_gt']  # (1403,614)
    # image_data_gt58 = data_gt58['indian_pines_large_gt']  # (1403,614)
    # label_data = label['Data_gt']  # (145,145)
    # np.save('Indian_pines/Indian_pines.npy', image_data)
    # np.save('Indian_pines/Indian_pines_large_corrected.npy', image_data_cor)
    # np.save('Indian_pines/Indian_pines_large_gt52.npy', image_data_gt52)
    # np.save('Indian_pines/Indian_pines_large_gt58.npy', image_data_gt58)
    # np.save('Indian_pines/Indian_pines_gt.npy', label_data)

    # Pavia
    # data = sio.loadmat('Pavia/Pavia.mat')
    # label = sio.loadmat('Pavia/Pavia_gt.mat')
    # image_data = data['HSI_original']  # H,W,C (1096,715,102)
    # label_data = label['Data_gt']  # H,W (1096,715)
    # np.save('Pavia/Pavia.npy', image_data)
    # np.save('Pavia/Pavia_gt.npy', label_data)
    #
    # datas = sio.loadmat('Pavia/Pavia_s.mat')
    # labels = sio.loadmat('Pavia/Pavia_s_gt.mat')
    # image_datas = datas['pavia_s']
    # label_datas = labels['pavia_s_gt']
    # np.save('Pavia/Pavia_s.npy', image_datas)
    # np.save('Pavia/Pavia_s_gt.npy', label_datas)
    #
    # dataU = sio.loadmat('Pavia/Pavia.mat')
    # dataU_PCA = sio.loadmat('Pavia/PaviaU_PCA.mat')
    # labelU = sio.loadmat('Pavia/PaviaU_PCA.mat')
    # image_dataU = dataU['HSI_original']  # H,W,C (1096,715,102)
    # image_dataU_PCA = dataU_PCA['HSI_PCA']  # H,W,C (604,340,3)
    # label_dataU = labelU['HSI_PCA']  # H,W (1096,715)
    # np.save('Pavia/PaviaU.npy', image_dataU)
    # np.save('Pavia/PaviaU_PCA.npy', image_dataU_PCA)
    # np.save('Pavia/PaviaU_gt.npy', label_dataU)

    # Houston
    # data = sio.loadmat('Houston/Houston.mat')
    # label = sio.loadmat('Houston/Houston_gt.mat')
    # image_data = data['Houston']  # H,W,C (349,1905,144)
    # label_data = label['Houston_gt']  # H,W (349,1905)
    # np.save('Houston/Houston.npy', image_data)
    # np.save('Houston/Houston_gt.npy', label_data)

    # dataU = sio.loadmat('Houston/Houston.mat')
    # labelU = sio.loadmat('Houston/Houston_gt.mat')
    # image_dataU = dataU['Houston']  # H,W,C (349,1905,144)
    # label_dataU = labelU['Houston_gt']  # H,W (349,1905)
    # np.save('Houston/HoustonU.npy', image_dataU)
    # np.save('Houston/HoustonU_gt.npy', label_dataU)
