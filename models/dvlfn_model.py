import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss
from models.bert import Bert


class DVLFN(nn.Module):
    def __init__(self, hidden_dim, max_len, num_statistic, device, gamma=0.01):
        super(DVLFN, self).__init__()
        self.bert_encoder = Bert(max_len=max_len, device=device)
        self.fc_statistic = nn.Sequential(nn.Linear(num_statistic, 100), nn.ReLU())
        self.fc_global_txt = nn.Sequential(nn.Linear(868, hidden_dim), nn.ReLU())
        self.fc_global_img = nn.Sequential(nn.Linear(2048, hidden_dim), nn.ReLU())

        self.fc_region_txt = nn.Sequential(nn.Linear(768, 50), nn.ReLU())
        self.fc_region_img = nn.Sequential(nn.Linear(2048, 50), nn.ReLU())
        self.w_distance = SamplesLoss(loss='sinkhorn', p=2, blur=.05, scaling=0.1)

        self.fc_pred_mix = nn.Sequential(
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
        )
        self.gamma = gamma

    def forward(self, txt, img_global, img_region, social):

        txt_global, txt_region, attn_mask = self.bert_encoder(txt)

        # Fusion prediction
        social = self.fc_statistic(social)
        txt_global = torch.cat((txt_global, social), dim=1)
        txt_global = self.fc_global_txt(txt_global)
        img_global = self.fc_global_img(img_global)

        mix_pred = self.fc_pred_mix(txt_global + img_global)

        # Inconsistency prediction
        txt_region = self.fc_region_txt(txt_region)
        img_region = self.fc_region_img(img_region)

        bs = txt_region.shape[0]
        num_words = torch.sum(attn_mask, dim=-1)
        w_dis = []
        for i in range(bs):
            w_dis.append(self.w_distance(txt_region[i][:num_words[i]], img_region[i]))

        w_dis = torch.stack(w_dis).unsqueeze(1)
        # ot_dis = torch.exp(-ot_dis * self.gamma).unsqueeze(1)
        w_pred = torch.cat((1-w_dis*self.gamma, w_dis*self.gamma), dim=1)

        # Final prediction
        logit = torch.where(mix_pred >= w_pred, mix_pred, w_pred)
        logit = F.softmax(logit, dim=1)

        return logit