import torch
import torch.nn as nn
import numpy as np


class GceLoss(nn.Module):

    def __init__(self, q=0.3, k=0.5, num_train=50000, device='cuda:0'):
        super(GceLoss, self).__init__()
        self.q = q
        self.k = k
        self.device = device
        self.weight = torch.nn.Parameter(data=torch.ones(num_train, 1), requires_grad=False)

    def forward(self, logits, targets, index, split='train'):
        Yg = torch.gather(logits, 1, torch.unsqueeze(targets, 1))

        if split == 'train':
            loss = ((1 - Yg ** self.q) / self.q - ((1 - self.k ** self.q) / self.q)) * self.weight[index]
            loss = torch.mean(loss)
        else:
            loss = (1 - Yg ** self.q) / self.q - ((1 - self.k ** self.q) / self.q)
            loss = torch.mean(loss)
        return loss

    def update_weight(self, logits, targets, index):
        Yg = torch.gather(logits, 1, torch.unsqueeze(targets, 1))
        Lq = ((1 - (Yg ** self.q)) / self.q)
        Lqk = np.repeat(((1 - (self.k ** self.q)) / self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.FloatTensor).to(self.device)
        Lqk = torch.unsqueeze(Lqk, 1)

        condition = torch.gt(Lqk, Lq)
        self.weight[index] = condition.type(torch.FloatTensor).to(self.device)