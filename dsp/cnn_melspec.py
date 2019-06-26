import torch
import torch.nn as nn
import argparse
import os
from model.conv import vgg11_bn
from preprocess.dataset import *
from tensorboardX import SummaryWriter


# 相关文件放到这里
xwriter = SummaryWriter('cnn_melspec_log')
# 实例化
data_feed = DataFeed()


def train(model: torch.nn.Module, optimizer, spec_fd, nepoch, nbatch=32):
    criterion = nn.CrossEntropyLoss().cuda()  # 交叉熵
    losses = []
    model.train()  # 训练模式
    print("start train")

    for iepoch in range(nepoch):
        # 加载并交叉验证
        train_iter, val_iter = spec_cvloader(
            spec_fd, iepoch % len(spec_fd), nbatch)
        # acc = evaluate(model, val_iter)  # 模型评估
        for i, (X, Y) in enumerate(train_iter):
            # print(X.shape, Y.shape)
            X = X.cuda()
            Y = Y.cuda()
            Ym = model(X)
            loss = criterion(Ym, Y)  # 损失函数
            xwriter.add_scalar('train/{}th'.format(iepoch),
                               loss.item() / X.size(0), i)
            losses.append(loss.item() / X.size(0))
            optimizer.zero_grad()  # 梯度初始化为0
            loss.backward()  # 反向传播求梯度
            optimizer.step()  # 更新所有参数

        acc = evaluate(model, val_iter)  # 模型评估
        print("Loss: {:.3f} Acc: {:.3f}".format(losses[-1], acc))

    print("train finished")


def evaluate(model: torch.nn.Module, val_iter):
    model.eval()  # 测试模式
    acc, tot = 0, 0
    with torch.no_grad():
        for i, (X, Y) in enumerate(val_iter):
            X = X.cuda()
            Y = Y.cuda()
            Ym = model(X)
            Ym = torch.argmax(Ym, dim=1).view(-1)
            Y = Y.view(-1)
            tot += Ym.size(0)
            acc += (Ym == Y).sum().item()
    return acc / tot


def individual_test(model: torch.nn.Module, stu):
    iter = spec_loader([stu], 32)
    acc = evaluate(model, iter)
    print("outsider test acc: {:.3f}".format(acc))


def outsider_test(model: torch.nn.Module, outsiders):
    for o in outsiders:
        iter = spec_loader([o], 32)
        acc = evaluate(model, iter)
        print("outsider {} test acc: {:.3f}".format(data_feed.stuids[o], acc))


def infer(model: torch.nn.Module, sample_path):
    X = read_sample(sample_path)
    X = X[None, :, :, :]
    X = X.cuda()
    model.eval()  # 让model变成测试模式
    print(X)
    print(X.shape)
    Ym = model(X)
    print(Ym)
    return data_feed.cates[torch.argmax(Ym, dim=1).item()]  # 返回推断的词


def build_model(load=''):
    model = vgg11_bn()
    # 定义训练器 学习率为4e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-5)
    if load:
        # 从磁盘文件中读取一个通过torch.save()保存的对象。
        # checkpoint文件会记录保存信息，通过它可以定位最新保存的模型
        checkpoint = torch.load(load)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    model.cuda()  # 使用gpu
    return model, optimizer


if __name__ == "__main__":
    # 编写命令行接口
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--name", default="cnn_melspec", type=str)
    argparser.add_argument("--infer", default='', type=str)
    argparser.add_argument("--nepoch", default=10, type=int)
    argparser.add_argument("--save", default="save.ptr", type=str)
    argparser.add_argument("--load", default='', type=str)
    args = argparser.parse_args()

    model, optimizer = build_model(args.load)
    if args.infer:
        infer(model, args.infer)

    candidates = range(22)  # [0,1,2....21]
    outsiders = range(32)  # 0-31  一共32个同学

    spec_fd = spec_folder(candidates, 10)  # 返回划分后的数据集

    train(model, optimizer, spec_fd, args.nepoch)  # 使用该模型，该训练器，该数据集，训练次数
    xwriter.export_scalars_to_json("./test.json")  # 输出标量
    xwriter.close()

    # outsider_test(model, outsiders)

    # 保存模型
    checkpointer = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save(checkpointer, args.save)

# （1）batchsize：批大小。在深度学习中，一般采用SGD训练，即每次训练在训练集中取batchsize个样本训练；
# （2）iteration：1个iteration等于使用batchsize个样本训练一次；
# （3）epoch：1个epoch等于使用训练集中的全部样本训练一次；