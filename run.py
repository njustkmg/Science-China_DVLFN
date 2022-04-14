import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from managers.trainer import train_epoch
from managers.evaluator import evaluate
from models.dvsfn_model import DVSFN
from models.gce_loss import GceLoss
from utils.news_dataset import NewsDataset


def run_dvsfn(params):

    train = pd.read_csv(os.path.join(params.data_dir, 'train.csv'))
    valid = pd.read_csv(os.path.join(params.data_dir, 'valid.csv'))
    test = pd.read_csv(os.path.join(params.data_dir, 'test.csv'))

    train_txt = train['text'].tolist()
    valid_txt = valid['text'].tolist()
    test_txt = test['text'].tolist()

    train_label = train['label'].tolist()
    valid_label = valid['label'].tolist()
    test_label = test['label'].tolist()

    train_global_img = np.load(os.path.join(params.global_dir, 'train.npy'), mmap_mode='r+')
    valid_global_img = np.load(os.path.join(params.global_dir, 'valid.npy'), mmap_mode='r+')
    test_global_img = np.load(os.path.join(params.global_dir, 'test.npy'), mmap_mode='r+')

    train_region_img = np.load(os.path.join(params.region_dir, 'train.npy'), mmap_mode='r+')
    valid_region_img = np.load(os.path.join(params.region_dir, 'valid.npy'), mmap_mode='r+')
    test_region_img = np.load(os.path.join(params.region_dir, 'test.npy'), mmap_mode='r+')

    train_statistic = np.load(os.path.join(params.statistic_dir, 'train.npy'))
    valid_statistic = np.load(os.path.join(params.statistic_dir, 'valid.npy'))
    test_statistic = np.load(os.path.join(params.statistic_dir, 'test.npy'))

    train_dataset = NewsDataset(train_txt, train_label, train_global_img, train_region_img, train_statistic)
    valid_dataset = NewsDataset(valid_txt, valid_label, valid_global_img, valid_region_img, valid_statistic)
    test_dataset = NewsDataset(test_txt, test_label, test_global_img, test_region_img, test_statistic)
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)

    model = DVSFN(
        hidden_dim=params.hidden_dim,
        max_len=params.max_len,
        num_statistic=train_statistic.shape[1],
        device=params.device,
        gamma=params.gamma
    )
    model.to(params.device)
    optimizer = optim.Adam(list(model.parameters()), lr=params.lr, weight_decay=params.weight_decay)

    if params.use_gce:
        criterion = GceLoss(num_train=len(train), device=params.device).to(params.device)
    else:
        criterion = nn.CrossEntropyLoss()

    best_acc = 0
    patience = params.patience

    for epoch in range(params.epoch):
        train_loss = train_epoch(params, epoch, model, train_loader, criterion, optimizer)
        valid_result = evaluate(params, model, valid_loader, criterion)

        if valid_result['acc'] > best_acc:
            best_acc = valid_result['acc']
            torch.save(model, os.path.join(params.checkpoint_dir, params.model_name))
            patience = params.patience
        else:
            patience -= 1
            if patience <= 0:
                break
        print("Epoch {:02d} | Train Loss {:3f} | Best Acc {:3f}".format(epoch, train_loss, best_acc))
        print(valid_result)

    model = torch.load(os.path.join(params.checkpoint_dir, params.model_name))
    model.to(params.device)
    test_result = evaluate(params, model, test_loader, criterion)
    print('Finishing test: ')
    print(test_result)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='DVSFN model')

    # Experiment setup params
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default='dvsfn.pt')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--data_dir', type=str, default='../data/',
                        help='Directory for text and label')
    parser.add_argument('--region_dir', type=str, default='../data/fastrcnn/',
                        help='Directory for extracted region image features')
    parser.add_argument('--global_dir', type=str, default='../data/resnet101/',
                        help='Directory for extracted global image features')
    parser.add_argument('--statistic_dir', type=str, default='../data/statistic/',
                        help='Directory for extracted statistical features')

    # Training params
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--use_gce', type=bool, default=True,
                        help='Whether to use generalize cross-entropy')
    parser.add_argument('--start_prune', type=int, default=10,
                        help='Staring prune weight for generalize cross-entropy')

    # Model params
    parser.add_argument('--hidden_dim', type=int, default=200,
                        help='Dimension for modal embeddings')
    parser.add_argument('--max_len', type=int, default=256,
                        help='Max sequence length for bert')
    parser.add_argument('--gamma', type=float, default=0.01,
                        help='For smooth wasserstein distance prediction')

    params = parser.parse_args()

    if not os.path.isdir(params.checkpoint_dir):
        os.mkdir(params.checkpoint_dir)

    run_dvsfn(params)

