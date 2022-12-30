import mindspore
import mindspore.nn as nn
from mindspore.ops import functional as F
from transformers import *

class Bert(nn.Cell):
    def __init__(self, max_len, device):
        super(Bert, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert')
        self.bert = BertModel.from_pretrained('bert')
        self.max_len = max_len
        self.device = device

    def construct(self, txt):
        batch_token = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=txt,
            add_special_tokens=True,
            truncation='only_first',
            max_length=self.max_len,
            padding=True,
        )
        # print(batch_token)
        # print('-'*40)
        input_ids = mindspore.Tensor(batch_token['input_ids'])
        # input_ids = batch_token['input_ids']
        attn_mask = mindspore.Tensor(batch_token['attention_mask'])
        # print(input_ids)
        # print(type(input_ids))
        # F.stop_gradient():
        bert_output = self.bert(input_ids, attention_mask=attn_mask)
        print(bert_output)
        # bert_output = F.stop_gradient(self.bert(input_ids, attention_mask=attn_mask))
        txt_global = F.stop_gradient(bert_output[0][:, 0, :])
        txt_region = F.stop_gradient(bert_output[0][:, 1:, :])

        return txt_global, txt_region, attn_mask