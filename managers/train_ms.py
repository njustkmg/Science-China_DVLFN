import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore import ms_function
from tqdm import tqdm
import mindspore
from mindspore import dtype as mstype

class WithLossCell_Cro(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(WithLossCell_Cro, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self.expend = ops.ExpandDims()
        self.argmax = ops.Argmax(axis=1, output_type=mindspore.int32)

    def construct(self, txt, img_global, img_region, social, label, index):
        out = self._backbone(txt, img_global, img_region, social)
        out = self.argmax(out)
        out = out.asnumpy()
        out = mindspore.Tensor(out, mindspore.float32)
        out = self.expend(out, 0)
        # label = label.set_dtype(mstype.int64)
        label = label.asnumpy()
        label = mindspore.Tensor(label, mindspore.float32)
        label = self.expend(label, 0)
        # print(out)
        # print(label)
        return self._loss_fn(out, label)


class WithLossCell_self(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(WithLossCell_self, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, txt, img_global, img_region, social, label, index):
        out = self._backbone(txt, img_global, img_region, social)
        # print(out)
        # print(label)
        return self._loss_fn(out, label, index)

    @property
    def backbone_network(self):
        """
        Get the backbone network.

        Returns:
            Cell, the backbone network.
        """
        return self._backbone

# one step train
class Dvlfn_TrainOneStepCell(nn.Cell):
    """自定义训练网络"""

    def __init__(self, network, optimizer):
        """入参有两个：训练网络，优化器"""
        super(Dvlfn_TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network                           # 定义前向网络
        self.network.set_grad()                          # 构建反向网络
        self.optimizer = optimizer                       # 定义优化器
        self.weights = self.optimizer.parameters         # 待更新参数
        self.grad = ops.GradOperation(get_by_list=True)  # 反向传播获取梯度

    # @ms_function
    def construct(self, txt, img_global, img_region, social, label, index):
        loss = self.network(txt, img_global, img_region, social, label, index)                            # 计算当前输入的损失函数值
        grads = self.grad(self.network, self.weights)(txt, img_global, img_region, social, label, index)  # 进行反向传播，计算梯度
        self.optimizer(grads)                                   # 使用优化器更新权重参数
        return loss