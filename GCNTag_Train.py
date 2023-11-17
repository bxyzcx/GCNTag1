import torch
from torch.utils.data import Dataset
import time
import logging
import logging.config
import deepnovo_config
import os
import datetime
from init_args import init_args
from train_func1 import train, build_model, validation, perplexity
logger = logging.getLogger(__name__)
from GCN_Train_data_reader import GCNTagTrainDataset,ReadPKL,collate_func
from torch import nn  # 完成神经欧网络的相关工作
from torch.nn import functional as F  # 常用的函数
from torch import optim  # 工具包
import matplotlib.pyplot as plt
import numpy as np
def Draw_z(epoch_losses):
    plt.figure()  # 设置图片信息 例如：plt.figure(num = 2,figsize=(640,480))
    plt.plot(epoch_losses, 'b', label='loss')  # epoch_losses 传入模型训练中的 loss[]列表,在训练过程中，先创建loss列表，将每一个epoch的loss 加进这个列表
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()  # 个性化图例（颜色、形状等）
    plt.show()




# 定义网络
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.max_pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.max_pool3 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.max_pool4 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(4608, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        in_size = x.size(0)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.relu(x)
        x = self.max_pool3(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.max_pool4(x)
        # 展开
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

class Net(nn.Module):
    def __init__(self,args):
        super(Net, self).__init__()

        # xw+b
        input_size = args.MAX_NUM_PEAK * 2 + 80
        self.fc1 = nn.Linear(input_size, 256)  #第一层 256 随机指定
        # self.fc1 = nn.Linear(580, 1)  #第一层 256 随机指定
        self.bacth1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.bacth2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)  # 第三层 因为是个十个分类所以最后应该是个10

    def forward(self, x):
        # x = F.relu((self.fc1(x)))
        # x = F.relu(self.fc2(x))
        # x = torch.sigmoid(self.fc3(x))

        x = F.relu(self.bacth1((self.fc1(x))))
        x = F.relu(self.bacth2(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))
        return x

def Train_GCNTag(args):
    batch_size = 128
    epochs = 50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"training mode")
    torch.cuda.empty_cache()
    tag_path = "E:/sdut/GCNTag_V3/PXD004565/test_20000_Data.tagbeamsearch"
    train_set = GCNTagTrainDataset(args.input_feature_file_train,
                                     args.input_spectrum_file_train,
                                     args=args, tagpath=tag_path)
    num_train_features = len(train_set)
    steps_per_epoch = int(num_train_features / batch_size)
    logger.info(f"{steps_per_epoch} steps per epoch")
    # print("line 48")
    train_data_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=0,
                                                    collate_fn=collate_func)
    net = Net(args).to(DEVICE)
    # model = ConvNet().to(DEVICE)
    train_loss = []
    # momentum=0.9 动量，动量梯度下降
    loss_list = []
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    loss1 = 10
    criterion = nn.BCELoss()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (x, y) in enumerate(train_data_loader):
            x,y =x.to(DEVICE),y.to(DEVICE).float()
            x = x.to(torch.float32)
            y = y.to(torch.float32)
            # print("shape1",x.shape)
            # print("x",x)
            optimizer.zero_grad()
            out = net(x)
            # print("out", out.shape,out)
            # print("y", y.shape,y)

            loss = F.binary_cross_entropy(out, y)

            # loss = criterion(out, y)
            # print("shape2", x.shape, y.shape)
            # print("loss",loss)
            # if loss<loss1:
            #     torch.save(net, 'model.pth')
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()  # 每一个批次的损失
            if (batch_idx + 1) % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                    epoch, (batch_idx + 1) * len(x), len(train_data_loader.dataset),

                           100. * (batch_idx + 1) / len(train_data_loader), loss.item()))
        epoch_loss /= (batch_idx + 1)
        if len(loss_list) != 0:
            if epoch_loss < loss_list[-1] :
                torch.save(net, 'model4.pth')
        loss_list.append(epoch_loss)
    Draw_z(loss_list)


def init_log(log_file_name):
    d = {
        'version': 1,
        'disable_existing_loggers': False,  # this fixes the problem
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'filename': log_file_name,
                'mode': 'w',
                'formatter': 'standard',
            }
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        }
    }
    logging.config.dictConfig(d)


if __name__ == '__main__':
    param_path = "./param/cross.9high_80k.exclude_bacillus_4_PeakNetwork_NH3H2O_InternalIons_Edge.cfg"
    log_path = "./log/GCNTag_log/[test77]"
    if os.path.isfile(param_path):
        log_path += (param_path.split("/")[-1] + "_")
        dir, param_file = os.path.split(param_path)
        # log_file_name = "top5_" + param_file[-4] + ".log"
        now = datetime.datetime.now().strftime("%Y%m%d%H%M")
        args = init_args(param_path)
        # log_file_name = "./log/" + now + "(" + str(args.engine_model) + ").log"
        log_file_name = log_path + now + "(" + str(args.engine_model) + ").log"
        init_log(log_file_name=log_file_name)
        if os.path.exists(args.train_dir):
            pass
        else:
            os.makedirs(args.train_dir)
        Train_GCNTag(args)

    elif os.path.isdir(param_path):
        list_dir = os.listdir(param_path)
        list_dir.sort(key=lambda x: int(x[33]))
        print(list_dir)
        for file in list_dir:
            one_param_path = os.path.join(param_path, file)
            if os.path.isfile(one_param_path):
                now = datetime.datetime.now().strftime("%Y%m%d%H%M")
                args = init_args(one_param_path)
                log_file_name = log_path + file+"_" + now + "(" + str(args.engine_model) + ").log"
                init_log(log_file_name=log_file_name)
                if os.path.exists(args.train_dir):
                    pass
                else:
                    os.makedirs(args.train_dir)
                Train_GCNTag(args)