import mindspore
from bert4ms import BertTokenizer, BertModel
import mindspore.nn as nn
from mindspore.ops import functional as F
import mindspore.ops as ops
from mindspore import ms_function

class Bert(nn.Cell):
    def __init__(self, max_len):
        super(Bert, self).__init__()
        self.tokenizer = BertTokenizer.load('bert-base-uncased')
        self.bert = BertModel.load('bert-base-uncased', from_torch=True)
        self.max_len = max_len
        self.oneslike = ops.ones_like

    def construct(self, txt):
        # batch_token = self.tokenizer.batch_encode_plus(
        #     batch_text_or_text_pairs=txt,
        #     add_special_tokens=True,
        #     truncation='only_first',
        #     max_length=self.max_len,
        #     padding=True,
        # )
        # print(batch_token)
        # print('-'*40)
        # input_ids = mindspore.Tensor(batch_token['input_ids'])
        ini_input_ids = []
        ini_attn_mask = []
        # get the ini inputs and attn_mask
        for txts in range(len(txt)):
            input_id = self.tokenizer.encode(txt[txts], add_special_tokens=True, truncate_first_sequence=True, max_length=self.max_len)
            # print(input_id)
            attn_masks = self.oneslike(mindspore.Tensor(input_id, mindspore.int32))
            attn_masks = list(attn_masks.asnumpy())
            # print(input_id)
            # print(attn_masks)
            ini_input_ids.append(input_id)
            ini_attn_mask.append(attn_masks)
        # get the max len of sublist
        max_sub = 0
        input_ids = []
        attn_mask = []
        for i in range(len(ini_input_ids)):
            if max_sub < len(ini_input_ids[i]): max_sub = len(ini_input_ids[i])
        # max_sub = max_sub - 1
        # print(max_sub)
        # padding the inputs and attn_masks
        for i in range(len(ini_input_ids)):
            input_id = list(ini_input_ids[i])
            attn_masks = list(ini_attn_mask[i])
            while len(input_id) < max_sub:
                input_id.append(0)
                attn_masks.append(0)
            # print(input_id)
            # print(attn_masks)
            # print(len(input_id))
            # print(len(attn_masks))
            input_ids.append(input_id)
            attn_mask.append(attn_masks)
        # print(input_ids)
        # print(attn_mask)

        input_ids = mindspore.Tensor(input_ids, mindspore.int32)
        attn_mask = mindspore.Tensor(attn_mask, mindspore.float32)
        # print(input_ids[0])
        # print(attn_mask[0])

        # input_ids = batch_token['input_ids']
        # attn_mask = mindspore.Tensor(batch_token['attention_mask'])
        # print(type(input_ids))
        # F.stop_gradient():
        self.bert.set_train(False)
        bert_output = self.bert(input_ids, attention_mask=attn_mask)
        # print(bert_output)
        # bert_output = F.stop_gradient(self.bert(input_ids, attention_mask=attn_mask))
        txt_global = F.stop_gradient(bert_output[0][:, 0, :])
        txt_region = F.stop_gradient(bert_output[0][:, 1:, :])

        # return  attn_mask
        return txt_global, txt_region, attn_mask