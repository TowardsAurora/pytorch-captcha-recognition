import re

from matplotlib import pyplot as plt

log_path = '../train_log/vgg16.txt'


def plot_img(path, save_filename):
    with open(path, "r") as f:  # 打开文件
        data = f.read()
    # print(data)

    pattern = re.compile(r'(?<=train_loss:)\d+\.?\d*')
    train_loss = pattern.findall(data)
    # print(train_loss)
    train_loss = [float(i) for i in train_loss]

    pattern = re.compile(r'(?<=test_acc:)\d+\.?\d*')
    test_acc = pattern.findall(data)
    # print(test_acc)
    test_acc = [float(i) for i in test_acc]

    pattern = re.compile(r'(?<=test_loss:)\d+\.?\d*')
    test_loss = pattern.findall(data)
    # print(test_loss)
    test_loss = [float(i) for i in test_loss]

    epoch = range(len(train_loss))
    # print(epoch)

    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    lns1 = ax1.plot(epoch, train_loss, color='red', linewidth=1, linestyle="solid", label="train_loss")

    ax2 = ax1.twinx()  # this is the important function
    lns2 = ax2.plot(epoch, test_acc, color='green', linewidth=1, linestyle="solid", label="test_acc")

    ax3 = ax2.twinx()  # this is the important function
    lns3 = ax3.plot(epoch, test_loss, color='blue', linewidth=1, linestyle="solid", label="test_loss")

    ax3.set_ylabel('test_loss')
    ax2.set_ylabel('test_acc')
    ax1.set_ylabel('train_loss')
    ax1.set_xlabel('Epoch')

    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    plt.show()

    # fig.tight_layout()
    # fig.savefig(save_filename + '.jpg')


if __name__ == '__main__':
    plot_img(log_path, 'log_plot_100')
