"""
A collection of useful code from https://github.com/d2l-ai

@author: Rui Zhu 
@creation time: 2025-04-08
"""
import numpy as np
import torch
from torch import nn
import torchvision
import time
from IPython import display
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
from astrokit.toolbox import sec_to_hms

class Timer:
    def __init__(self):
        self.times = []
        self.start()
    
    def now(self):
        return time.time()
    
    def start(self):
        self.tik = time.time()
    
    def stop(self):
        """
        停止计时器, 并将时间添加到times列表中, 然后返回最后一次计时的时间间隔
        """
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """
        返回平均时间
        """
        return sum(self.times) / len(self.times)
    
    def sum(self):
        """
        返回总时间
        """
        return sum(self.times)
    
    def cumsum(self):
        """
        返回累计时间
        例如: self.times = [1, 2, 3], 则self.cumsum() = [1, 3, 6]
        """
        return np.array(self.times).cumsum().tolist()


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """
    设置matplotlib的坐标轴
    """
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    if legend:
        axes.legend(legend)
    axes.grid()

class Animator:
    """
    在动画中绘制数据
    """
    def __init__(self, 
                 xlabel=None, ylabel=None, legend=None, 
                 xlim=None, ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), 
                 nrows=1, ncols=1, figsize=(5, 3)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        backend_inline.set_matplotlib_formats('svg')
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes,]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

    
def accuracy_num(y_hat, y):
    """
    计算分类正确的数量
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter, device):
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
    
    metric = Accumulator(n=2)  # 正确预测数和预测总数
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy_num(net(X), y), y.numel())
    return metric[0] / metric[1]  # 正确预测数 / 总数

def load_data_fashion_mnist(batch_size, dir_data, resize=None):
    """
    加载Fashion-MNIST数据集

    Parameters
    ----------
    batch_size : int
        批量大小

    dir_data : str
        数据集存储路径
        如果数据集不存在, 则会自动下载到该路径
    
    resize : tuple, optional
        图像大小, 默认None表示不缩放
    """
    # 将图像转换成PyTorch张量, 并将图像像素标准化到[0, 1]范围内
    trans = [torchvision.transforms.ToTensor()]
    if resize:
        trans.insert(0, torchvision.transforms.Resize(resize))
    trans = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root=dir_data, train=True, transform=trans, download=True
        )
    mnist_test = torchvision.datasets.FashionMNIST(
        root=dir_data, train=False, transform=trans, download=True
        )
    return (torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True, 
                                        num_workers=4),
            torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False, 
                                        num_workers=4))


def train_net(net, train_iter, test_iter, num_epochs, lr, device):
    st = time.time()
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)  # 初始化权重
    net.to(device)  # 移动到指定设备
    print(f'training on {device}')

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()  # 交叉熵损失函数

    animator = Animator(
        xlabel='epoch', xlim=[1, num_epochs], 
        legend=['train loss', 'train acc', 'test acc']
    )

    # 开始训练
    timer = Timer()
    num_batches = len(train_iter)
    for epoch in range(num_epochs):
        # 每个epoch开始时, 创建一个空的累加器
        metric = Accumulator(n=3)  # 训练损失, 正确预测数, 预测总数
        net.train()  # 设置为训练模式
        for i, (X, y) in enumerate(train_iter):
            timer.start()  # 开始计时
            optimizer.zero_grad()  # 清空梯度
            X, y = X.to(device), y.to(device)  # 移动到指定设备
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()  # 更新参数
            with torch.no_grad():  # 暂时不计算梯度
                metric.add(
                    l * X.shape[0], # 当前批次的总损失
                    accuracy_num(y_hat, y), # 准确正确数
                    X.shape[0] # 预测总数
                )
            timer.stop()

            # 更新图形
            train_loss = metric[0] / metric[2]  # 平均损失
            train_acc = metric[1] / metric[2]  # 准确率
            if (i+1) % (num_batches // 5) == 0 or i == (num_batches - 1):
                animator.add(
                    x=epoch + (i + 1) / num_batches,
                    y=[train_loss, train_acc, None]
                )
        test_acc = evaluate_accuracy(net, test_iter, device)
        animator.add(epoch + 1, [None, None, test_acc])

    print(f"train loss: {train_loss:.3f}, "
          f"train acc: {train_acc:.3f}, "
          f"test acc: {test_acc:.3f}")
    print(f"{metric[2] * num_epochs / timer.sum():.1f} examples/sec on "
          f"{str(device)}")
    print(f"Cost Time: {sec_to_hms(time.time() - st)}")