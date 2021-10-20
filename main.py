import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# x = torch.tensor(
#     [[1., 2., 3., 4.],
#      [5., 6., 7., 8.],
#      [9., 10., 11., 12.]]
# )

matplotlib.rcParams['figure.figsize'] = (25.0, 7.0)

x_train = torch.rand(1000)
x_train = x_train * 10.0 - 5.0

y_sub_train = torch.cos(x_train)**2

# правильный график
# plt.plot(x_train.numpy(), y_train.numpy(), 'o')
# plt.title('$ y = cos^2(x) $')
# plt.show()

# шум
noisy = torch.rand(y_sub_train.shape) / 3.
# plt.plot(x_train.numpy(), noisy.numpy(), 'o')
# plt.show()

# зашумленный график
y_train = y_sub_train + noisy
# plt.plot(x_train.numpy(), y_train.numpy(), 'o')
# plt.title(' noisy $ y = cos^2(x) $')
# plt.show()


x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

x_val = torch.linspace(-5, 5, 100)
y_val = torch.cos(x_val.data)**2
plt.plot(x_val.numpy(), y_val.numpy(), 'o')



class OurNet(nn.Module):
    def __init__(self, n_hid_n):
        super(OurNet, self).__init__()
        self.fc1 = nn.Linear(1, n_hid_n)
        self.act1 = nn.Sigmoid()
        self.fc3 = nn.Linear(n_hid_n, 1)

    def forward(self, a):
        a = self.fc1(a)
        a = self.act1(a)
        a = self.fc3(a)
        return a


our_net = OurNet(100)


def predict(net, a, y):
    y_pred = net.forward(a)
    plt.plot(a.numpy(), y.numpy(), 'o', label='То что должно быть')
    plt.plot(a.numpy(), y_pred.data.numpy(), 'o', c='r', label='Предсказание сети')
    plt.legend(loc='upper left')
    plt.show()


predict(our_net, x_val, y_val)


optimizer = torch.optim.Adam(our_net.parameters(), lr=0.001)


def loss(pred, true):
    sq = (pred-true)**2
    return sq.mean()


for e in range(10000):
    optimizer.zero_grad()
    y_pred = our_net.forward(x_train)
    loss_val = loss(y_pred, y_train)
    # print(loss_val)

    loss_val.backward()
    optimizer.step()

    if not e % 2000:
        print(loss_val)


predict(our_net, x_val, y_val)


