import numpy as np
from tqdm import tqdm


def train_epoch_w_gce(params, epoch, model, train_loader, criterion, optimizer):
    if epoch >= params.start_prune and epoch % 5 == 0:
        # prune weight for generalize cross-entropy
        model.eval()
        train_tqdm = tqdm(train_loader)
        for batch in train_tqdm:
            index, txt, img_global, img_region, social, label = batch
            index = index.tolist()
            out = model(txt, img_global.to(params.device), img_region.to(params.device), social.to(params.device)).squeeze()
            criterion.update_weight(out, label.to(params.device), index)

    model.train()
    all_loss = []
    train_tqdm = tqdm(train_loader)
    for batch in train_tqdm:
        optimizer.zero_grad()
        index, txt, img_global, img_region, social, label = batch
        index = index.tolist()
        out = model(txt, img_global.to(params.device), img_region.to(params.device), social.to(params.device)).squeeze()
        loss = criterion(out, label.to(params.device), index)
        loss.backward()
        optimizer.step()

        all_loss.append(loss.item())
        train_tqdm.set_description("Loss: %f" % (np.mean(all_loss)))
    return np.mean(all_loss)


def train_epoch_wo_gce(params, epoch, model, train_loader, criterion, optimizer):
    model.train()
    all_loss = []
    train_tqdm = tqdm(train_loader)
    for batch in train_tqdm:
        optimizer.zero_grad()
        index, txt, img_global, img_region, social, label = batch
        out = model(txt, img_global.to(params.device), img_region.to(params.device), social.to(params.device)).squeeze()
        loss = criterion(out, label.to(params.device))
        loss.backward()
        optimizer.step()

        all_loss.append(loss.item())
        train_tqdm.set_description("Loss: %f" % (np.mean(all_loss)))
    return np.mean(all_loss)


def train_epoch(params, epoch, model, train_loader, criterion, optimizer):
    if params.use_gce:
        return train_epoch_w_gce(params, epoch, model, train_loader, criterion, optimizer)
    else:
        return train_epoch_wo_gce(params, epoch, model, train_loader, criterion, optimizer)
