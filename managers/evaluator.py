import numpy as np
from mindspore.ops import functional as F
import mindspore.ops as ops
import mindspore
# import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate(params, model, valid_loader, criterion):
    model.eval()
    all_score = []
    all_label = []
    all_loss = []
    for batch in valid_loader:
        with F.stop_gradient():
            index, txt, img_global, img_region, social, label = batch
            index = index.tolist()
            out = model(txt, img_global.to(params.device), img_region.to(params.device), social.to(params.device)).squeeze()

            if params.use_gce:
                loss = criterion(out, label.to(params.device), index, split='valid')
            else:
                loss = criterion(out, label.to(params.device))

            all_loss.append(loss.item())
            preds = ops.Argmax(out.detach().cpu(), 1)

        all_score += preds.tolist()
        all_label += label.cpu().tolist()

    acc = accuracy_score(all_label, all_score)
    p = precision_score(all_label, all_score)
    r = recall_score(all_label, all_score)
    f1 = f1_score(all_label, all_score)
    return {'loss': np.mean(all_loss), 'acc': acc, 'p': p, 'r': r, 'f1': f1}