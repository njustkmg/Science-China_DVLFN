import torch
from torch.utils.data import Dataset


class NewsDataset(Dataset):

    def __init__(self, txt, label, img_global, img_region, social):

        self.label = label
        self.txt = txt
        self.img_global = img_global
        self.img_region = img_region
        self.social = social
        self.ids = list(range(len(label)))

    def __getitem__(self, index):
        ids = self.ids[index]
        txt = self.txt[index]
        img_global = self.img_global[index]
        img_region = self.img_region[index]
        social = self.social[index]
        label = self.label[index]

        return ids, txt, img_global, img_region, social, label

    def __len__(self):
        return len(self.label)
