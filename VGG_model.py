import torchvision
from torch import nn
from torchvision.models import list_models

from utils import captcha_setting

# print(list_models())


class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        self.num_cls = captcha_setting.MAX_CAPTCHA * captcha_setting.ALL_CHAR_SET_LEN
        self.base = torchvision.models.vgg11(pretrained=False)
        self.base.classifier[-1] = nn.Linear(4096, self.num_cls)

    def forward(self, x):
        out = self.base(x)
        return out


class VGG13(nn.Module):
    def __init__(self):
        super(VGG13, self).__init__()
        self.num_cls = captcha_setting.MAX_CAPTCHA * captcha_setting.ALL_CHAR_SET_LEN
        self.base = torchvision.models.vgg13(pretrained=False)
        self.base.classifier[-1] = nn.Linear(4096, self.num_cls)

    def forward(self, x):
        out = self.base(x)
        return out


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.num_cls = captcha_setting.MAX_CAPTCHA * captcha_setting.ALL_CHAR_SET_LEN
        self.base = torchvision.models.vgg16(pretrained=False)
        self.base.classifier[-1] = nn.Linear(4096, self.num_cls)

    def forward(self, x):
        out = self.base(x)
        return out


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.num_cls = captcha_setting.MAX_CAPTCHA * captcha_setting.ALL_CHAR_SET_LEN
        self.base = torchvision.models.vgg19(pretrained=False)
        self.base.classifier[-1] = nn.Linear(4096, self.num_cls)

    def forward(self, x):
        out = self.base(x)
        return out
