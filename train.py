import argparse
import os
import time
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import my_dataset
from model.captcha_cnn_model import CNN
from model.vgg import vgg
from model.VGG_model import VGG11, VGG13, VGG16, VGG19
from captcha_test import test_acc, test_acc_test_tqdm
from utils.Logger import Logger


def count_img_nums():
    path = "./dataset/train"
    count = 0
    for _ in os.listdir(path):  # _ 表示的是文件名
        count = count + 1
    print("参与的训练的图片有{}张".format(count))


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--epochs', type=int, default=300, help='')
    parser.add_argument('--batch_size', type=int, default=64, help='')

    parser.add_argument('--learning_rate', type=float, default=0.00007, help='')

    opt = parser.parse_args()

    return opt


def execute_test():
    model = CNN()
    # model = vgg()
    # model = VGG16()
    print("Now we start init network--CNN")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("The train device is:", device)
    opt = parse_option()

    train_dataloader = my_dataset.get_train_data_loader()
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=opt.learning_rate,
        betas=(0.9, 0.999),
    )
    criterion = nn.MultiLabelSoftMarginLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    for epoch in range(opt.epochs):
        print("epoch:{}现已在进行中！当前学习率为:{}".format((epoch + 1), opt.learning_rate))
        epoch_start_time = datetime.now()
        for i, (images, labels) in enumerate(train_dataloader):
            images = Variable(images)
            labels = Variable(labels.float())
            images = images.to(device)  # 将tensor转换成了CUDA 类型
            predict_labels = model(images)
            labels = labels.to(device)  # 将tensor转换成了CUDA 类型
            # print(predict_labels.type)
            # print(labels.type)
            loss = criterion(predict_labels, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print("epoch:", epoch + 1, "step:", i + 1, "loss:", loss.item())
            if (i + 1) % 200 == 0:
                torch.save(model.state_dict(), "saved_model/model.pkl")  # current is model.pkl
                print("阶段性模型已保存")
                acc = test_acc()
                acc_theta = 0.9
                if acc > acc_theta:
                    torch.save(model.state_dict(), "saved_model/model.pkl")
                    print("当前精读已超过{}%，程序停止，保存最终模型！".format(acc_theta * 100))
                    break
        else:
            continue
        print("epoch:", epoch + 1, "step:", i + 1, "loss:", loss.item())
        epoch_end_time = datetime.now()
        print("当前第{}轮已经结束，准确率为：{}%，训练和测试时长为{}".format(epoch + 1, test_acc() * 100, epoch_end_time - epoch_start_time))
        break
    torch.save(model.state_dict(), "saved_model/model.pkl")  # current is model.pkl.
    print("最终模型存储完成")


def execute_only():
    model = CNN()
    print("Now we start init network--CNN")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("The train device is:", device)
    opt = parse_option()

    train_dataloader = my_dataset.get_train_data_loader()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=opt.learning_rate,
                                 betas=(0.9, 0.99)
                                 )
    criterion = nn.MultiLabelSoftMarginLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    for epoch in range(opt.epochs):
        print("epoch:{}现已在进行中！当前学习率为:{}".format((epoch + 1), opt.learning_rate))
        for i, (images, labels) in enumerate(train_dataloader):
            images = Variable(images)
            labels = Variable(labels.float())
            images = images.to(device)  # 将tensor转换成了CUDA 类型
            predict_labels = model(images)
            labels = labels.to(device)  # 将tensor转换成了CUDA 类型
            # print(predict_labels.type)
            # print(labels.type)
            loss = criterion(predict_labels, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print("epoch:", epoch + 1, "step:", i + 1, "loss:", loss.item())
            if (i + 1) % 200 == 0:
                torch.save(model.state_dict(), "saved_model/model.pkl")  # current is model.pkl
                print("阶段性模型已保存")
    torch.save(model.state_dict(), "saved_model/model.pkl")  # current is model.pkl.
    print("最终模型存储完成")


def execute_only_tqdm():
    model = CNN()
    print("Now we start init network--CNN")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("The train device is:", device)
    opt = parse_option()

    train_dataloader = my_dataset.get_train_data_loader()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=opt.learning_rate,
                                 betas=(0.9, 0.999)
                                 )
    criterion = nn.MultiLabelSoftMarginLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    for epoch in range(opt.epochs):
        for i, (images, labels) in enumerate(tqdm(train_dataloader)):
            images = Variable(images)
            labels = Variable(labels.float())
            images = images.to(device)  # 将tensor转换成了CUDA 类型
            predict_labels = model(images)
            labels = labels.to(device)  # 将tensor转换成了CUDA 类型
            # print(predict_labels.type)
            # print(labels.type)
            loss = criterion(predict_labels, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # if (i + 1) % 100 == 0:
            #     print("epoch:", epoch + 1, "step:", i + 1, "loss:", loss.item())
            # if (i + 1) % 200 == 0:
            #     torch.save(model.state_dict(), "saved_model/model.pkl")  # current is model.pkl
            # print("阶段性模型已保存")
        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), "saved_model/model.pkl")  # current is model.pkl
            print("阶段性模型已保存")
        print("epoch:{}    loss:{}".format((epoch + 1), loss))
    torch.save(model.state_dict(), "saved_model/model.pkl")  # current is model.pkl.
    print("最终模型存储完成")


def execute_only_tqdm_test():
    model = CNN()
    print("Now we start init network--CNN")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("The train device is:", device)
    opt = parse_option()

    train_dataloader = my_dataset.get_train_data_loader()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=opt.learning_rate,
                                 betas=(0.9, 0.999)
                                 )
    criterion = nn.MultiLabelSoftMarginLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        filename = input("请输入保存数据的文件名:")
        logger = Logger('./train_log/' + filename)
    for epoch in range(opt.epochs):
        for i, (images, labels) in enumerate(tqdm(train_dataloader)):
            images = Variable(images)
            labels = Variable(labels.float())
            images = images.to(device)  # 将tensor转换成了CUDA 类型
            predict_labels = model(images)
            labels = labels.to(device)  # 将tensor转换成了CUDA 类型
            # print(predict_labels.type)
            # print(labels.type)
            loss = criterion(predict_labels, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), "saved_model/model.pkl")  # current is model.pkl
        print("epoch:{}    loss:{}".format((epoch + 1), loss))

        time.sleep(2)
        accuracy, loss2 = test_acc_test_tqdm()
        theta = 0.85
        if accuracy > theta:
            break
        logger.write("epoch:{} train_loss:{} test_acc:{} test_loss:{}".format(epoch, loss, accuracy, loss2))
    torch.save(model.state_dict(), "saved_model/model.pkl")  # current is model.pkl.
    print("最终模型存储完成")


if __name__ == '__main__':
    count_img_nums()
    startTime = datetime.now()
    # execute_test()
    # execute_only_tqdm()
    execute_only_tqdm_test()
    endTime = datetime.now()
    print("训练总时间为：{}".format(endTime - startTime))

# # some ilus
"""
使用.zero_grad(set_to_none=True)而不是.zero_grad()。这样做会让内存分配器去处理梯度，而不是主动将它们设置为0。正如在文档中所说的那样，这会导致产生一个适度的加速，所以不要期待任何奇迹。

注意，这样做并不是没有副作用的！关于这一点的详细信息请查看文档。
"""
