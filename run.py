import os
import argparse
import numpy as np
import pandas as pd
import mindspore
from mindspore import context
from mindspore import nn, Tensor, Model
from mindspore.nn.metrics import Accuracy, Precision, F1, Recall
from managers.trainer import train_epoch
from managers.evaluator import evaluate
from models.dvlfn_model import DVLFN
from models.gce_loss import GceLoss
from utils.news_dataset import NewsDataset
from mindspore.dataset import GeneratorDataset
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from managers.train_ms import *
from tqdm import tqdm

def run_dvsfn(params):

    # get the data and generate dataset
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
    train_loader = GeneratorDataset(train_dataset, ['ids', 'txt', 'img_global', 'img_region', 'social', 'label'], shuffle=False)
    train_loader = train_loader.batch(params.batch_size)
    valid_loader = GeneratorDataset(valid_dataset, ['ids', 'txt', 'img_global', 'img_region', 'social', 'label'], shuffle=False)
    valid_loader = valid_loader.batch(params.batch_size)
    test_loader = GeneratorDataset(test_dataset, ['ids', 'txt', 'img_global', 'img_region', 'social', 'label'], shuffle=False)
    test_loader = test_loader.batch(params.batch_size)

    # define the model
    model = DVLFN(
        hidden_dim=params.hidden_dim,
        max_len=params.max_len,
        num_statistic=train_statistic.shape[1],
        device=params.device,
        gamma=params.gamma
    )

    # model.to(params.device)
    optimizer = nn.Adam(list(model.trainable_params()), learning_rate=params.lr, weight_decay=params.weight_decay)

    # declare the loss
    if params.use_gce:
        criterion = GceLoss(num_train=len(train))
    else:
        criterion = nn.SoftmaxCrossEntropyWithLogits()

    # connect the forward network and loss funciton
    # net_loss = nn.WithLossCell(model, criterion)
    net_loss = WithLossCell_self(model, criterion)
    tran_net = Dvlfn_TrainOneStepCell(net_loss, optimizer)

    tran_net.set_train()
    for epoch in range(params.epoch):
        train_tqdm = tqdm(train_loader)
        # if epoch >= params.start_prune and epoch % 5 == 0:
        #     # prune weight for generalize cross-entropy
        #     train_tqdm = tqdm(train_loader)
        #     for batch in train_tqdm:
        #         index, txt, img_global, img_region, social, label = batch
        #         index = index.asnumpy().tolist()
        #         out = model(txt, img_global, img_region,
        #                     social).squeeze()
        #         criterion.update_weight(out, label, index)
        all_loss = []
        for batch in train_tqdm:
            index, txt, img_global, img_region, social, label = batch
            index = index.asnumpy().tolist()
            # print(1)
            # print('-'*40)
            loss = net_loss(txt, img_global, img_region, social, label, index)
            loss = loss.asnumpy().tolist()
            # print(loss)
            tran_net(txt, img_global, img_region, social, label, index)
            # print(loss)
            # print(type(loss))
            # print(loss)
            all_loss.append(loss)
            # print(all_loss)
            # print('-'*40)
            train_tqdm.set_description("Loss: %f" % (np.mean(all_loss)))





   # for epoch in range(params.epoch):
    #     train_loss = train_epoch(params, epoch, model, train_loader, criterion, optimizer)
    #     valid_result = evaluate(params, model, valid_loader, criterion)
    #
    #     if valid_result['acc'] > best_acc:
    #         best_acc = valid_result['acc']
    #         mindspore.save_checkpoint(model, os.path.join(params.checkpoint_dir, params.model_name))
    #         patience = params.patience
    #     else:
    #         patience -= 1
    #         if patience <= 0:
    #             break
    #     print("Epoch {:02d} | Train Loss {:3f} | Best Acc {:3f}".format(epoch, train_loss, best_acc))
    #     print(valid_result)


    # model = mindspore.load_checkpoint(os.path.join(params.checkpoint_dir, params.model_name))
    # model.to(params.device)
    # test_result = evaluate(params, model, test_loader, criterion)
    # print('Finishing test: ')
    # print(test_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DVSFN model')
    parser.add_argument('--device_target', type=str, default="GPU", choices=['Ascend', 'GPU', 'CPU'])

    # Experiment setup params
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default='dvsfn.pt')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data/',
                        help='Directory for text and label')
    parser.add_argument('--region_dir', type=str, default='./data/fastrcnn/',
                        help='Directory for extracted region image features')
    parser.add_argument('--global_dir', type=str, default='./data/resnet101/',
                        help='Directory for extracted global image features')
    parser.add_argument('--statistic_dir', type=str, default='./data/statistic/',
                        help='Directory for extracted statistical features')

    # Training params
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--use_gce', type=bool, default=False,
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
    # params = parser.parse_args()
    params = parser.parse_known_args()[0]
    # set information of model
    context.set_context(mode=context.GRAPH_MODE, device_target=params.device_target)

    run_dvsfn(params)

