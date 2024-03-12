import torch
import torch.nn as nn
import numpy as np
from torchsummaryX import summary
from collections import OrderedDict


class BasicBlock(nn.Module):
    # 整体网络的重要组件层，由两个一维卷积、两个batch_norm、两个relu和一个最大值池化组成
    def __init__(self,in_ch, out_ch, std=2):
        super(BasicBlock, self).__init__()
        self.stride = std

        self.conv1 = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=13, stride=1, padding=6, bias=False)

        self.bn1 = nn.BatchNorm1d(num_features=out_ch)
        self.relu1 = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(in_channels=out_ch, out_channels=out_ch, kernel_size=13, stride=std, padding=6, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_ch)
        self.relu2 = nn.ReLU()

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=std, padding=1)

    def forward(self, x):
        signal = x
        signal = self.conv1(signal)
        signal = self.bn1(signal)
        signal = self.relu1(signal)
        # signal = self.dropout(signal)
        signal = self.conv2(signal)
        signal = self.bn2(signal)
        signal = self.relu2(signal)

        if self.stride!=1:
            x = self.maxpool(x)
        signal += x
        # print('signal',signal.shape)
        return signal

class ResnetBlock(nn.Module):
    # 整体网络架构，包括由卷积层为核心的head层；多个basicblock层；一个lstm层；一个多层全连接网络构成的输出层
    def __init__(self, block, block_stride, in_channel=1, out_channel=32, classes=5):
        super(ResnetBlock, self).__init__()
        self.block_nem = len(block_stride)
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.head = self._make_head()
        self.in_channel = self.out_channel

        self.layers = self._make_layers(block, block_stride, self.out_channel)
        self.tail = self._make_tail()

        self.tail_maxpool = nn.MaxPool1d(kernel_size=3, stride=2,padding=1)

        self.lstm = nn.LSTM(input_size=32, hidden_size=32, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.classes = classes

        self.classifier = self._make_classifier()

    def init_state(self):
        state = (torch.zeros(size=(1, self.current_batch, 32)),
                 torch.zeros(size=(1, self.current_batch, 32)))
        return state

    def _make_classifier(self):
        classifier = nn.Sequential()
        classifier.add_module('fc1', nn.Linear(in_features=480, out_features=24*6))
        classifier.add_module('bn1', nn.BatchNorm1d(24*6))
        classifier.add_module('relu1', nn.ReLU())
        classifier.add_module('fc2', nn.Linear(in_features=24 * 6, out_features=48))
        classifier.add_module('fc3', nn.Linear(in_features=48, out_features=self.classes))
        return classifier

    def _make_layers(self, block, block_stride, out_channel):
        layers = []
        for i in range(len(block_stride)):
            stride = block_stride[i]
            layers.append(('basicblock_{}'.format(i), block(self.in_channel, out_channel, std=stride)))
            self.in_channel = out_channel
        return nn.Sequential(OrderedDict(layers))

    def _make_head(self):
        head = nn.Sequential()
        head.add_module('conv1', nn.Conv1d(in_channels=self.in_channel, out_channels=self.out_channel,\
                                           kernel_size=33, stride=1, padding=16, bias=False))
        head.add_module('bn1', nn.BatchNorm1d(num_features=self.out_channel))
        head.add_module('relu1', nn.ReLU())
        return head

    def _make_tail(self):
        tail = nn.Sequential()
        tail.add_module('conv1', nn.Conv1d(in_channels=self.in_channel, out_channels=self.out_channel, \
                                           kernel_size=3, stride=2, padding=1, bias=False))
        tail.add_module('bn1', nn.BatchNorm1d(num_features=self.out_channel))
        tail.add_module('relu1', nn.ReLU())
        # tail.add_module('dropout', nn.Dropout(p=0.2))
        return tail

    def forward(self, x):
        self.current_batch = len(x)
        self.state = self.init_state()
        device = torch.device("cuda" if x.is_cuda else "cpu")
        self.state = (self.state[0].to(device), self.state[1].to(device))
        out = self.head(x)
        # (bn, 1, 6000)-->(bn, out_channel, 6000)
        out = self.layers(out)
        # (bn, out_channel, 6000)-->(bn, out_channel, 47)
        out = self.tail_maxpool(out) + self.tail(out)
        # (bn, out_channel, 47)-->(bn, out_channel, 24)
        out = torch.permute(out,[0,2,1])
        # (bn, out_channel, 24)-->(bn, 24, out_channel)
        out, state = self.lstm(out, self.state)
        # (bn, 24, out_channel)-->(bn, 24, out_channel)
        # out = self.dropout(out)
        out = out.reshape(self.current_batch, -1)
        # (bn, 24, out_channel)-->(bn, 24*32)
        out = self.classifier(out)
        return out

if __name__ == '__main__':
    # 对网络进行简单测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ResnetBlock(block=BasicBlock, block_stride=[1, 4, 4, 4, 2], in_channel=1, out_channel=32)
    x = torch.randn((2, 1, 3750))
    out = net.forward(x)
    loss_fcn = nn.CrossEntropyLoss(weight=torch.tensor([1, 1, 1, 1, 1]))
    print('net', out)