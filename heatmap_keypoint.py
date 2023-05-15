import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
bs = 256
heatmap_size = 56

def create_batch():

    batch = torch.rand((bs, heatmap_size, heatmap_size))
    soft_heatmaps = nn.Softmax(dim=1)(batch.view((bs, -1))) #.view((bs, heatmap_size, heatmap_size))
    _, max_ind = soft_heatmaps.max(dim=1)
    targets = torch.stack([torch.div(max_ind, heatmap_size), max_ind % heatmap_size], -1).to(torch.float)
    flattened = soft_heatmaps.view((bs, heatmap_size*heatmap_size))
    return flattened.to('cuda:0'), targets.to('cuda:0')


class MLP(nn.Module):

    def __init__(self, heatmap_size=56, hid_size=128):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(heatmap_size**2, hid_size)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hid_size, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

model = MLP().to('cuda:0')

mse_loss = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

count_targets = torch.zeros((heatmap_size, heatmap_size))

for i in range(1000000):
    batch, targets = create_batch()

    pos = range(heatmap_size*heatmap_size)
    softargmax = np.sum(batch*pos)

    # int_targets = targets.to(torch.long)
    # count_targets[int_targets[:, 0], int_targets[:, 1]] += 1
    # output = model(batch)
    # loss = mse_loss(output, targets)
    # loss.backward()
    # optimizer.step()
    # if i % 1000 == 0:
    #     print(loss)


beta = 12
y_est = np.array([[1.1, 3.0, 1.1, 1.3, 0.8]])
a = np.exp(beta*y_est)
b = np.sum(np.exp(beta*y_est))
softmax = a/b
max = np.sum(softmax*y_est)
print(max)
pos = range(y_est.size)
softargmax = np.sum(softmax*pos)
print(softargmax)

# print(train.max())