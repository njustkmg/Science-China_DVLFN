import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
# from geomloss import SamplesLoss
from models.sample_loss import SamplesLoss
from models.ms_bert import Bert
from mindspore import ms_function
import datetime
import time

class DVLFN(nn.Cell):
    def __init__(self, hidden_dim, max_len, num_statistic, device, gamma=0.01):
        super(DVLFN, self).__init__()
        self.bert_encoder = Bert(max_len=max_len)
        self.fc_statistic = nn.SequentialCell(nn.Dense(num_statistic, 100), nn.ReLU())
        self.fc_global_txt = nn.SequentialCell(nn.Dense(868, hidden_dim), nn.ReLU())
        self.fc_global_img = nn.SequentialCell(nn.Dense(2048, hidden_dim), nn.ReLU())

        self.fc_region_txt = nn.SequentialCell(nn.Dense(768, 50), nn.ReLU())
        self.fc_region_img = nn.SequentialCell(nn.Dense(2048, 50), nn.ReLU())
        self.w_distance = SamplesLoss(loss='sinkhorn', p=2, blur=.05, scaling=0.1)

        self.fc_pred_mix = nn.SequentialCell(
            nn.Dense(200, 100),
            nn.ReLU(),
            nn.Dense(100, 2),
        )
        self.gamma = gamma
        self.expand_dims = ops.ExpandDims()
        self.con1 = ops.Concat(1)
        self.sum = ops.ReduceSum(keep_dims=False)
        self.softmax = ops.Softmax(axis=1)

    # @ms_function
    def construct(self, txt, img_global, img_region, social):
        txt = list(txt.asnumpy())
        # txt = tuple(txt)
        # print(txt)
        # print(type(txt))
        # txt_list = list(txt.keys())
        # print(txt_list)
        # txt_global, txt_region, attn_mask = self.bert_encoder(txt_list)
        # print(type(txt))
        # print(time.clock())
        self.bert_encoder.set_train(False)
        # print('begin to use bert')
        txt_global, txt_region, attn_mask = self.bert_encoder(txt)
        # print('over use bert')
        # print('-'*40)
        # print(time.clock())

        # print(txt_region)
        # print(txt_region.shape)
        # print(txt_global.shape)
        # print('-' * 40)
        # Fusion prediction
        social = self.fc_statistic(social)
        # print(txt_global.shape)
        # print(social.shape)
        # print('*' * 40)
        txt_global = self.con1((txt_global, social))
        txt_global = self.fc_global_txt(txt_global)
        img_global = self.fc_global_img(img_global)

        mix_pred = self.fc_pred_mix(txt_global + img_global)

        # Inconsistency prediction
        txt_region = self.fc_region_txt(txt_region)
        img_region = self.fc_region_img(img_region)

        bs = txt_region.shape[0]
        # get each sen's length
        # print(attn_mask)
        num_words = self.sum(attn_mask, -1)
        num_words = list(num_words.asnumpy())
        num_words = list(map(int, num_words))
        # num_words = mindspore.Tensor(num_words, mindspore.int32)
        # print(type(num_words))
        # print(num_words)

        # print(time.clock())
        # print('precessed the words')
        w_dis = []
        # print(bs)
        for i in range(bs):
            words_len = num_words[i]
            if words_len == max(num_words): words_len = words_len - 1

            # print(num_words[i])
            # print(txt_region[i])
            # print(txt_region[i].shape)
            # print(txt_region[i][:words_len].shape)
            # print(img_region[i].shape)
            # print(txt_region[i][:words_len])
            # print(img_region[i])
            # print(self.w_distance(txt_region[i][:words_len], img_region[i]))
            w_diss = self.w_distance(txt_region[i][:words_len], img_region[i])
            # print(w_diss)
            w_diss = (w_diss.asnumpy()).tolist()
            # print(type(w_diss))
            # print(w_diss.dtype)
            # print('-'*40)
            w_dis.append(w_diss)
            # print(w_dis)
            # print('-'*40)
        # print(time.clock())
        # print('words has processed')
        w_dis = mindspore.Tensor(w_dis, mindspore.float32)
        w_dis = self.expand_dims((w_dis), 1)
        # print(w_dis)
        # print('-'*40)

        # ot_dis = torch.exp(-ot_dis * self.gamma).unsqueeze(1)
        w_pred = self.con1((1 - w_dis * self.gamma, w_dis * self.gamma))
        # print(mix_pred)
        # print(w_pred)
        # print('-'*40)

        # Final prediction
        # condition = mindspore.numpy.full(mix_pred >= w_pred, [False, True])
        logit = mindspore.numpy.where(mix_pred >= w_pred, mix_pred, w_pred)
        # print(condition)
        # print(logit)
        # print('='*40)


        logit = self.softmax(logit)
        # print(logit)
        # print('-'*40)
        return logit
