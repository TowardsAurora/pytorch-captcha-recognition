from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import captcha_setting, my_dataset, one_hot_encoding
from model.captcha_cnn_model import CNN
from tqdm import tqdm


def main():
    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('saved_model/model.pkl'))
    print("load cnn net.")
    test_dataloader = my_dataset.get_test_data_loader()

    correct = 0
    total = 0
    for i, (images, labels) in enumerate(test_dataloader):
        image = images
        vimage = Variable(image)
        predict_label = cnn(vimage)
        c0 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
        true_label = one_hot_encoding.decode(labels.numpy()[0])
        total += labels.size(0)
        if predict_label == true_label:
            correct += 1
        if total % 500 == 0:
            print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
    print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))


def test_acc():
    startTime = datetime.now()

    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('saved_model/model.pkl'))
    print("开始测试---->")
    print("Test load cnn net.")

    test_dataloader = my_dataset.get_test_onek_data_loader()

    correct = 0
    total = 0
    for i, (images, labels) in enumerate(test_dataloader):
        image = images
        vimage = Variable(image)
        predict_label = cnn(vimage)

        c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
        true_label = one_hot_encoding.decode(labels.numpy()[0])
        total += labels.size(0)
        if predict_label == true_label:
            correct += 1
    accuracy = correct / total
    print('当前 %d 张图片测试的准确率为: %f %%' % (total, 100 * correct / total))
    endTime = datetime.now()
    print("当前{}张图片测试总时间为：{}".format(total, endTime - startTime))
    return accuracy


def test_acc_test():
    startTime = datetime.now()

    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('saved_model/model.pkl'))
    print("开始测试---->")
    print("Test load cnn net.")

    test_dataloader = my_dataset.get_test_data_loader()

    correct = 0
    total = 0
    for i, (images, labels) in enumerate(test_dataloader):
        image = images
        vimage = Variable(image)
        predict_label = cnn(vimage)

        c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
        true_label = one_hot_encoding.decode(labels.numpy()[0])
        total += labels.size(0)
        if predict_label == true_label:
            correct += 1
        # if total % 500 == 0:
        # print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
    # print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))

    accuracy = correct / total
    print('当前 %d 张图片测试的准确率为: %f %%' % (total, 100 * correct / total))
    # print("当前{}张图片测试的准确率为:{}%".format(total,accuracy*100))
    endTime = datetime.now()
    print("当前{}张图片测试总时间为：{}".format(total, endTime - startTime))
    return accuracy


def test_acc_test_tqdm_cpu():
    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('saved_model/model.pkl'))
    print("开始测试 加载 CNN 模型")
    # print("Test load cnn net.")

    test_dataloader = my_dataset.get_test_data_loader()


    correct = 0
    total = 0
    for i, (images, labels) in enumerate(tqdm(test_dataloader)):
        image = images
        vimage = Variable(image)
        predict_label = cnn(vimage)

        c0 = captcha_setting.ALL_CHAR_SET[
            np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0,
            2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0,
            3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]

        predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
        true_label = one_hot_encoding.decode(labels.numpy()[0])
        total += labels.size(0)
        if predict_label == true_label:
            correct += 1
    accuracy = correct / total
    print('当前 %d 张图片测试的准确率为: %f %% '% (total, 100 * correct / total))
    return accuracy


def test_acc_test_tqdm():
    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('saved_model/model.pkl'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  ###########
    print("开始测试 加载 CNN 模型")
    # print("Test load cnn net.")

    test_dataloader = my_dataset.get_test_data_loader()

    criterion = nn.MultiLabelSoftMarginLoss()
    if torch.cuda.is_available():
        cnn = cnn.cuda()
        criterion = criterion.cuda()

    correct = 0
    total = 0
    for i, (images, labels) in enumerate(tqdm(test_dataloader)):
        image = images
        vimage = Variable(image)
        vimage = vimage.to(device)  # 将tensor转换成了CUDA 类型
        predict_label = cnn(vimage)
        labels = Variable(labels.float())
        labels = labels.to(device)  # 将tensor转换成了CUDA 类型
        loss = criterion(predict_label, labels)

        c0 = captcha_setting.ALL_CHAR_SET[
            np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.cpu().numpy())]
        c1 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.cpu().numpy())]
        c2 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0,
            2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.cpu().numpy())]
        c3 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0,
            3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.cpu().numpy())]

        predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
        true_label = one_hot_encoding.decode(labels.cpu().numpy()[0])
        total += labels.size(0)
        if predict_label == true_label:
            correct += 1
    accuracy = correct / total
    print('当前 %d 张图片测试的准确率为: %f %% loss: %f ' % (total, 100 * correct / total, loss))
    return accuracy,loss


if __name__ == '__main__':
    main()
