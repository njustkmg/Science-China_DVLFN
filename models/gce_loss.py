import mindspore
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.common.initializer import One


class GceLoss(nn.Cell):

    def __init__(self, q=0.3, k=0.5, num_train=50000):
        super(GceLoss, self).__init__()
        self.q = q
        self.k = k
        # self.device = device
        self.weight =mindspore.Parameter(Tensor(shape=(num_train, 1), dtype=mstype.float32, init=One()), requires_grad=False)
        self.mean = ops.ReduceMean()
        self.expend = ops.ExpandDims()
        self.gather = ops.GatherD()

    def construct(self, logits, targets, index):
        Yg = self.gather(logits, 1, self.expend(targets, 1))

        # if split == 'train':
        loss = ((1 - Yg ** self.q) / self.q - ((1 - self.k ** self.q) / self.q)) * self.weight[index]
        loss = self.mean(loss)
        # else:
        #     loss = (1 - Yg ** self.q) / self.q - ((1 - self.k ** self.q) / self.q)
        #     loss = self.mean(loss)
        return loss

    def update_weight(self, logits, targets, index):
        Yg = self.gather(logits, 1, self.expend(targets, 1))
        Lq = ((1 - (Yg ** self.q)) / self.q)
        Lqk = np.repeat(((1 - (self.k ** self.q)) / self.q), targets.size(0))
        Lqk = Tensor.from_numpy(Lqk).type(mstype.float32)
        Lqk = self.expend(Lqk, 1)

        condition = self.greater(Lqk, Lq)
        self.weight[index] = condition.type(mstype.float32)


