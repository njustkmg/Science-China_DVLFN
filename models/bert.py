import torch
import torch.nn as nn
from transformers import *


class Bert(nn.Module):
    def __init__(self, max_len, device):
        super(Bert, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base')
        self.bert = BertModel.from_pretrained('bert-base')
        self.max_len = max_len
        self.device = device

    def forward(self, txt):
        batch_token = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=txt,
            add_special_tokens=True,
            truncation='only_first',
            max_length=self.max_len,
            padding=True,
        )
        input_ids = torch.tensor(batch_token['input_ids']).to(self.device)
        attn_mask = torch.tensor(batch_token['attention_mask']).to(self.device)
        with torch.no_grad():
            bert_output = self.bert(input_ids, attention_mask=attn_mask)
            txt_global = bert_output[0][:, 0, :]
            txt_region = bert_output[0][:, 1:, :]

        return txt_global, txt_region, attn_mask
