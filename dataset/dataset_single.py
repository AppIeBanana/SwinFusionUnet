import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, flag, img_size, num_classes):
        data_dir = 'dataset/data1_channel8_{}.npy'.format(img_size)
        label_dir = 'dataset/label_class{}_{}_an.npy'.format(num_classes, img_size)

        data = np.load(data_dir)
        label = np.load(label_dir)
        # print(data.shape,label.shape)
        data = data.astype(np.float32)
        label = label.astype(np.float32)

        test_size = 0.2
        train_x, test_x = train_test_split(data, test_size=test_size, shuffle=True,
                                           random_state=70)
        train_y, test_y = train_test_split(label, test_size=test_size, shuffle=True, random_state=70)
        if flag == "train":
            self._data = train_x
            self._label = train_y
        elif flag == "val" or "test":
            self._data = test_x
            self._label = test_y
        else:
            raise Exception("dataset must in train val test")

    def __getitem__(self, idx):
        img = self._data[idx]
        label = self._label[idx]
        return img, label

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
    #
