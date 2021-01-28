'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-01-28 09:57:52
LastEditors: ZhangHongYu
LastEditTime: 2021-01-28 10:34:37
'''
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        # 初始网络的内部结构
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # 一次正向行走过程
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


def train():
    print('------      构建数据集      ------')
    pd.read_csv('')
    # torch.linspace是为了生成连续间断的数据，第一个参数表示起点，第二个参数表示终点，第三个参数表示将这个区间分成平均几份，即生成几个数据
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    # torch.rand返回的是[0,1]之间的均匀分布   这里是使用一个计算式子来构造出一个关联结果，当然后期要学的也就是这个式子
    y = x.pow(2) + 0.2 * torch.rand(x.size())
    # Variable是将tensor封装了下，用于自动求导使用
    x, y = Variable(x), Variable(y)
    # 绘图展示
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.show()

    print('------      搭建网络      ------')
    # 使用固定的方式继承并重写 init和forword两个类

    net = Net(n_feature=1, n_hidden=1000, n_output=1)
    print('网络结构为：', net)

    print('------      启动训练      ------')
    loss_func = F.mse_loss
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

    # 使用数据 进行正向训练，并对Variable变量进行反向梯度传播  启动100次训练
    for t in range(10000):
        # 使用全量数据 进行正向行走
        prediction = net(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()  # 清除上一梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 应用梯度

        # 间隔一段，对训练过程进行可视化展示
        if t % 5 == 0:
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())  # 绘制真是曲线
            plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
            plt.text(0.5, 0, 'Loss=' +
                     str(loss.data.item()), fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)
    plt.ioff()
    plt.show()
    print('------      预测和可视化      ------')


if __name__ == '__main__':
    train()
