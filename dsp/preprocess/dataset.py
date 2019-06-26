import torchvision
from torchvision import transforms
import torch
import torch.utils.data.dataset
import os
import numpy as np
from preprocess.feature_extraction import mel_spec
import random
random.seed(233)

# todo: 
data_dir = "./speech_data"
class_num = 20
item_num = 20


class DataFeed(object):
    def __init__(self):
        self.data_dir = data_dir  
        # 获得以学号命名的文件夹列表     
        self.stuids = os.listdir(self.data_dir) 
        # 获得需要识别的孤立词列表
        self.words = "数字 语音 语言 识别 中国 总工 北京 背景 上海 商行 复旦 饭店 Speech Speaker Signal Process Print Open Close Project".split(' ')


    def __len__(self):
        # 得到学生数目
        return len(self.stuids)


    # 得到录音文件路径，参数：stu表示第几个学生，word表示第几个词，ith表示第几次录音
    def get_path(self, stu, word, ith):
        assert 0 <= stu < len(self)
        assert 0 <= word < 20
        assert 0 <= ith < 20
        stuid = self.stuids[stu]
        ret = os.path.join(self.data_dir, stuid,
                           "{0}-{1:02}-{2:02}.wav".format(stuid, word, ith + 1))
        return ret


    def get_blob(self, stu, word, ith):
        path = self.get_path(stu, word, ith)
        return open(path, "rb").read()


    # 录音文件路径加上标签（词是第几个词），形成列表，和学号一起返回
    def get_stu(self, stu):
        paths = []
        for i in range(class_num):
            for j in range(item_num):
                paths.append((self.get_path(stu, i, j), i))
        return paths, self.stuids[stu]


    def get_by_id(self, num):
        ith = num % 20
        num //= 20
        word = num % 20
        num //= 20
        assert num < len(self)
        return self.get_path(num, word, ith), word


def read_sample(path):
    try:
        spec = mel_spec(path)
        # 将二维数组变为三维
        spec = spec.reshape(1, spec.shape[0], -1)
        spec = torch.from_numpy(spec).type(torch.float)
    except:
        raise IOError("IOError {}".format(path))

    return spec


class SpecDataset(torch.utils.data.Dataset):

    def __init__(self, paths, cuda=True):
        self.paths = []
        self.cuda = cuda
        for t in paths:
            self.paths += t

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        (path, clas) = self.paths[item]
        spec = read_sample(path)
        return spec, torch.tensor(clas)


#  十折交叉验证，将总数据中的一份作为测试数据
#  folds保存了所有的数据，随机选出一份作为测试数据
def spec_folder(candidates, nfolds=10):
    d = DataFeed()
    paths = []
    for c in candidates:
        for i in range(class_num):
            for j in range(item_num):
                paths.append((d.get_path(c, i, j), i))
    random.shuffle(paths)
    tot = len(paths)
    fold_size = tot // nfolds
    folds = [paths[i: i + fold_size] for i in range(0, tot, fold_size)]
    return folds


def spec_cvloader(folds, nfold, batch_size, num_workers=0, shuffle=True, cuda=True):
    return torch.utils.data.DataLoader(SpecDataset(folds[:nfold] + folds[nfold + 1:]), pin_memory=cuda,
                                       batch_size=batch_size, shuffle=shuffle, num_workers=num_workers), \
        torch.utils.data.DataLoader(SpecDataset(folds[nfold: nfold + 1]), batch_size=batch_size, pin_memory=cuda,
                                    shuffle=shuffle, num_workers=num_workers)


def spec_loader(candidates, batch_size, num_workers=0, shuffle=True, cuda=True):
    d = DataFeed()
    paths = []
    for c in candidates:
        for i in range(class_num):
            for j in range(item_num):
                paths.append((d.get_path(c, i, j), i))
    return torch.utils.data.DataLoader(SpecDataset([paths]), pin_memory=cuda,
                                       batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
