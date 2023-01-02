#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :models.py
@说明        :定义AdaIN需要的各种模型, 比如VGG, 
@时间        :2022/12/30 17:22:09
@作者        :Reggie
@版本        :1.0
'''
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transform
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt


# 加载预训练的vgg19模型
vgg = torchvision.models.vgg19(pretrained=True)
# 提取预训练vgg16的特征层
vgg_features_layers = vgg.features

class VGG_Encoder(nn.Module):
    '''
    输入shape: [B, 3, 64, 64]
    输出shape: [B, 512, 4, 4]
    '''
    def __init__(self):
        super().__init__()
        # 分层, 按输出通道数进行分层, 
        # 比如 
        #   3 -> 64 一层; 
        #   64 -> 128 二层
        #   128 -> 256 三层
        #   256 -> 512 四层
        self.slice1 = vgg_features_layers[0:2]
        self.slice2 = vgg_features_layers[2:7]
        self.slice3 = vgg_features_layers[7:12]
        self.slice4 = vgg_features_layers[12:21]
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input, only_get_last_output=False):
        output1 = self.slice1(input)
        output2 = self.slice2(output1)
        output3 = self.slice3(output2)
        output = self.slice4(output3)
        if only_get_last_output:
            return output
        else:
            return output1, output2, output3, output

def get_mean_std(features, eps=1e-5):
    '''
    features.shape ==> [B, C, H, W]
    return: [B, C, 1, 1]

    步骤:
        1. [B, C, H, W] --> [B, C, H*W] 4维张量转为3维
        2. 计算每个通道的均值和标准差, i.e., 在dim=2上进行均值和标准差计算
    '''
    # step 1: [B, C, H, W] --> [B, C, H*W]
    batch_size, channels = features.shape[:2]
    features = features.view((batch_size, channels, -1))
    # step 2: 计算均值和标准差
    mean = features.mean(dim=2).view((batch_size, channels, 1, 1))
    std = (features.var(dim=2) + eps).view((batch_size, channels, 1, 1))
    return mean, std

def adaIN(content_features, style_features):
    '''
    content_features 和 style_features是由content_image 和 style_image经过VGG_encoder后提取到的特征向量
    两个特征向量具有一样的形状 (B, 512, H, W)
    '''
    # step 1: 计算content_features和style_features的均值和标准差
    content_mean, content_std = get_mean_std(content_features)
    style_mean, style_std = get_mean_std(style_features)
    # step 2: style_std * ((x-mean) / std) + style_mean
    adaIN_features = style_std * ((content_features - content_mean) / content_std) + style_mean
    return adaIN_features

class Decoder(nn.Module):
    '''
    输入shape: [B, 512, 4, 4]
    输出shape: [B, 3, 64, 64]
    '''
    def __init__(self):
        super().__init__()
        def conv_relu_block(in_channels, out_channels, k=3, padding=1, last_block=False):
            if last_block:
                return nn.Sequential(
                    nn.ReflectionPad2d((padding, padding, padding, padding)),
                    nn.Conv2d(in_channels, out_channels, kernel_size=k)
                )
            else: 
                return nn.Sequential(
                    nn.ReflectionPad2d((padding, padding, padding, padding)),
                    nn.Conv2d(in_channels, out_channels, kernel_size=k),
                    nn.ReLU(inplace=True)
                )
        self.model = nn.Sequential(
            conv_relu_block(512, 256, k=3, padding=1), 
            # 进行上采样
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv_relu_block(256, 256, k=3, padding=1),
            conv_relu_block(256, 256, k=3, padding=1),
            conv_relu_block(256, 256, k=3, padding=1),
            conv_relu_block(256, 128, k=3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv_relu_block(128, 128, k=3, padding=1),
            conv_relu_block(128, 64, k=3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv_relu_block(64, 64, k=3, padding=1),
            conv_relu_block(64, 3, k=3, padding=1, last_block=True),
        )

    def forward(self, input):
        output = self.model(input)
        return output

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

class AdaIN_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VGG_Encoder()
        self.decoder = Decoder().apply(init_weight)

    @staticmethod
    def get_loss_lc(gen_img_features, t):
        '''
        计算content_image的损失函数, 生成图像和特征t之间的欧式距离
        gen_img_features: 生成的图像通过Encoder后得到的特征向量
        '''
        return F.mse_loss(gen_img_features, t)

    @staticmethod
    def get_loss_ls(style_img_features, gen_img_features):
        ls = 0.0
        for i, (style, gen) in enumerate(zip(style_img_features, gen_img_features)):
            # 计算均值和标准差
            style_mean, style_std = get_mean_std(style)
            gen_mean, gen_std = get_mean_std(gen)
            # 计算LS, 两部分组成
            ls += F.mse_loss(style_mean, gen_mean) + F.mse_loss(style_std, gen_std)
        return ls

    def forward(self, content_images, style_images, lamda=10):
        content_features = self.encoder(content_images, only_get_last_output=True) # 用于合成特征t
        layered_style_features = self.encoder(style_images) # 用于计算LS
        style_features = layered_style_features[-1] # 用于合成特征t
        # 计算AdaIN特征
        t = adaIN(content_features, style_features)
        # 将 特征 t 通过Decoder
        generated_imgs = self.decoder(t)
        # 将合成的图像通过encoder获得其特征, 需要进行分层
        layered_gen_img_features = self.encoder(generated_imgs) # 这个特征用于计算LS
        gen_img_features = layered_gen_img_features[-1] # 这个特征用于计算LC
        # 计算loss
        lc_loss = self.get_loss_lc(gen_img_features, t)
        ls_loss = self.get_loss_ls(layered_style_features, layered_gen_img_features)
        loss = lc_loss + lamda * ls_loss
        return generated_imgs, loss

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--lr_decay', type=float, default=5e-4)
parser.add_argument('--channels', type=int, default=3)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument('--data_size', type=int, default=50000)
parser.add_argument('--content_dataset_path', type=str, default='D:\\BaiduNetdiskDownload\\seeprettyface_chs_wanghong\\xinggan_face')
parser.add_argument('--style_dataset_path', type=str, default='../data/faces')
parser.add_argument('--saved_model', type=str, default='model_state')
parser.add_argument('--save_path', type=str, default='imgs')
parser.add_argument('--gpu', type=bool, default=True)

opt = parser.parse_args()

cuda = True if opt.gpu and torch.cuda.is_available() else False

img_shape = torch.tensor([opt.channels, opt.img_size, opt.img_size])

class Content_Dataset(Dataset):
    def __init__(self):
        super().__init__()
        self.path = opt.content_dataset_path
        self.files = [os.path.join(self.path,x) for x in os.listdir(self.path)[:opt.data_size] if x.endswith(".jpg")]
        self.tfm = transform.Compose([
            transform.Resize(opt.img_size),
            transform.ToTensor()
        ])

    def __getitem__(self, index):
        img_name = self.files[index]
        img = Image.open(img_name)
        img = self.tfm(img)
        return img

    def __len__(self):
        return len(self.files)


class Style_Dataset(Dataset):
    def __init__(self):
        super().__init__()
        self.path = opt.style_dataset_path
        self.files = [os.path.join(self.path,x) for x in os.listdir(self.path)[:opt.data_size] if x.endswith(".jpg")]
        self.tfm = transform.Compose([
            transform.Resize(opt.img_size),
            transform.ToTensor()
        ])
    
    def __getitem__(self, index):
        img_name = self.files[index]
        img = Image.open(img_name)
        img = self.tfm(img)
        return img

    def __len__(self):
        return len(self.files)

# 构造loader
content_loader = DataLoader(
    dataset=Content_Dataset(),
    batch_size=opt.batch_size,
    shuffle=True
)
style_loader = DataLoader(
    dataset=Style_Dataset(),
    batch_size=opt.batch_size,
    shuffle=True
)

model = AdaIN_Net()
model = model.cuda() if cuda else model

# 优化器
optimiser = torch.optim.Adam(model.decoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# 创建目录
if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path, exist_ok=True)
    
if not os.path.exists(opt.saved_model):
    os.makedirs(opt.saved_model, exist_ok=True)

# 自适应学习率
def adjust_learning_rate(optimiser, iteration_count):
    """Imitating the original implementation"""
    lr = opt.lr / (1.0 + opt.lr_decay * iteration_count)
    for param_group in optimiser.param_groups:
        param_group['lr'] = lr

# 训练
for epoch in range(opt.epochs):
    for i, (content_imgs, style_imgs) in tqdm(enumerate(zip(content_loader, style_loader))):
        # adjust_learning_rate(optimiser, iteration_count=i)
        # 移动到GPU
        content_imgs = content_imgs.cuda() if cuda else content_imgs
        style_imgs = style_imgs.cuda() if cuda else style_imgs
        # 计算loss
        optimiser.zero_grad()
        generated_imgs, loss = model(content_imgs, style_imgs)
        loss.backward()
        optimiser.step()
    print(f'[epoch: {epoch+1} / {opt.epochs} loss: {float(loss):.6f}]')
    # 保存模型
    torch.save(model.state_dict(), f'{opt.saved_model}/{epoch}_epoch.pth')
    # 保存图像
    fp = os.path.join(opt.save_path, '%s.png'%(str(epoch + 1)))
    imgs = torch.cat([content_imgs, style_imgs, generated_imgs], dim=3)
    nrow = content_imgs.shape[0]
    save_image(imgs.data, fp=fp, nrow=nrow, normalize=True)