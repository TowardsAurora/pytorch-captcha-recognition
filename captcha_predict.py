import numpy as np
import torch
import os
from torch.autograd import Variable
from visdom import Visdom
from utils import captcha_setting, my_dataset
from model.captcha_cnn_model import CNN


def count_img_nums():
    path = "./dataset/predict"
    count = 0
    for _ in os.listdir(path):  # fn 表示的是文件名
        count = count + 1
    print("参与的预测的图片有{}张".format(count))


def main():
    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('saved_model/model.pkl'))
    print("load cnn net.")

    predict_dataloader = my_dataset.get_predict_data_loader()

    vis = Visdom()
    for i, (images, labels) in enumerate(predict_dataloader):
        image = images
        vimage = Variable(image)
        predict_label = cnn(vimage)

        c0 = captcha_setting.ALL_CHAR_SET[
            np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = captcha_setting.ALL_CHAR_SET[
            np.argmax(
                predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]

        c = '%s%s%s%s' % (c0, c1, c2, c3)
        print(c)

        # vis.images(image, opts=dict(caption=c))
        vis.images(image, opts=dict(caption=c))


def predict():
    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('saved_model/model.pkl'))
    print("load cnn net.")

    predict_dataloader = my_dataset.get_predict_no_shuffle_data_loader()

    vis = Visdom()
    for i, (images, labels) in enumerate(predict_dataloader):
        image = images
        vimage = Variable(image)
        predict_label = cnn(vimage)

        listdir = os.listdir('./dataset/predict')

        print("true:", listdir[i].split('_')[0])

        c0 = captcha_setting.ALL_CHAR_SET[
            np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = captcha_setting.ALL_CHAR_SET[
            np.argmax(
                predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]

        c = '%s%s%s%s' % (c0, c1, c2, c3)
        print("predict:", c)

        # vis.images(image, opts=dict(caption=c))
        vis.images(image, opts=dict(caption=c))        
        
        
        
if __name__ == '__main__':
    count_img_nums()
    main()
    # or # predict()
