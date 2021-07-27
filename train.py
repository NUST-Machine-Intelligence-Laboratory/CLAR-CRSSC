import os
import sys
import pathlib
import time
import datetime
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torch.nn.functional as F
from torch.distributions import Categorical
from apex import amp
from net import build_model
from utils.core import accuracy, evaluate
from utils.builder import *
from utils.utils import *
from utils.meter import AverageMeter
from utils.logger import Logger, print_to_logfile, print_to_console
from utils.plotter import plot_results
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Queue(object):
    def __init__(self, n_samples, memory_length=5):
        super().__init__()
        self.n_samples = n_samples
        self.memory_length = memory_length
        # the item in content_dict is as follows:
        #   dict {
        #         key: 'pred', value: [(pred class index, pred probability), ...];
        #         key: 'loss', value: [ loss, ... ];
        #         key: 'most_prob_label', value: predicted label with highest accumulated probability
        #        }
        self.content = np.array([
            {'pred': [], 'loss': [], 'label': -1} for i in range(n_samples)
        ])
        self.most_prob_labels = torch.Tensor([-1 for i in range(n_samples)]).long()

    def update(self, indices, losses, scores, labels):
        probs, preds = scores.max(dim=1)
        for i in range(indices.shape[0]):
            if len(self.content[indices[i].item()]['pred']) >= self.memory_length:
                self.content[indices[i].item()]['pred'].pop(0)
                self.content[indices[i].item()]['loss'].pop(0)
            self.content[indices[i].item()]['pred'].append((preds[i].item(), probs[i].item()))

            try:
                self.content[indices[i].item()]['loss'].append(losses[i].item())
            except:
                print(indices.shape, losses.shape)
                raise AssertionError()

            self.content[indices[i].item()]['label'] = labels[i].item()

        for i in range(indices.shape[0]):
            tmp = {}
            most_prob_label = -1
            highest_prob = 0
            for pred_idx, pred_prob in self.content[indices[i].item()]['pred']:
                if pred_idx not in tmp:
                    tmp[pred_idx] = pred_prob
                else:
                    tmp[pred_idx] += pred_prob
                if highest_prob < tmp[pred_idx]:
                    highest_prob = tmp[pred_idx]
                    most_prob_label = pred_idx
            self.most_prob_labels[indices[i].item()] = most_prob_label


def main(cfg, device):
    T_k = cfg.T_k
    mem_len = cfg.mem_len
    eps = cfg.eps
    conf_metric = cfg.conf_metric

    init_seeds()
    cfg.use_fp16 = False if device.type == 'cpu' else cfg.use_fp16

    # logging ----------------------------------------------------------------------------------------------------------------------------------------
    logger_root = f'Results/{cfg.dataset}'
    if not os.path.isdir(logger_root):
        os.makedirs(logger_root, exist_ok=True)
    logtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(logger_root, f'{logtime}-{cfg.log}')
    logger = Logger(logging_dir=result_dir, DEBUG=False)
    logger.set_logfile(logfile_name='log.txt')
    save_params(cfg, f'{result_dir}/params.json', json_format=True)
    logger.debug(f'Result Path: {result_dir}')

    # model, optimizer, scheduler --------------------------------------------------------------------------------------------------------------------
    n_classes = cfg.n_classes
    net = build_model(arch=cfg.net, n_classes=n_classes, pretrained=True, reduction_factor=cfg.r)
    # optimizer = build_adam_optimizer(net.parameters(), cfg.lr)
    # optimizer = build_sgd_optimizer(net.parameters(), cfg.lr, cfg.weight_decay)
    optimizer = build_optimizer(net.parameters(), cfg.lr, cfg.weight_decay, cfg.opt)
    scheduler = build_cosine_lr_scheduler(optimizer, cfg.epochs)
    opt_lvl = 'O1' if cfg.use_fp16 else 'O0'
    net, optimizer = amp.initialize(net.to(device), optimizer, opt_level=opt_lvl, keep_batchnorm_fp32=None, loss_scale=None, verbosity=0)

    # Adjust learning rate and betas for Adam Optimizer
    epoch_decay_start = 20
    mom1 = 0.9
    mom2 = 0.9
    lr_plan = [cfg.lr] * cfg.epochs
    beta1_plan = [mom1] * cfg.epochs
    for i in range(epoch_decay_start, cfg.epochs):
        lr_plan[i] = float(cfg.epochs - i) / (cfg.epochs - epoch_decay_start) * cfg.lr
        beta1_plan[i] = mom2

    # dataset, dataloader ----------------------------------------------------------------------------------------------------------------------------
    transform = build_transform(rescale_size=cfg.rescale_size, crop_size=cfg.crop_size)
    dataset = build_webfg_dataset(os.path.join(cfg.database, cfg.dataset), transform['train'], transform['test'])
    train_loader = DataLoader(dataset['train'], batch_size=cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=False, num_workers=8, pin_memory=False)

    # meters -----------------------------------------------------------------------------------------------------------------------------------------
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()
    epoch_train_time = AverageMeter()

    # training ---------------------------------------------------------------------------------------------------------------------------------------
    start_epoch = 0
    best_accuracy = 0.0
    best_epoch = None
    scheduler.last_epoch = start_epoch
    memory_pool = Queue(n_samples=len(dataset['train']), memory_length=mem_len)

    for epoch in range(start_epoch, cfg.epochs):
        start_time = time.time()

        # pre-step in this epoch
        net.train()
        train_loss.reset()
        train_accuracy.reset()
        curr_lr = [group['lr'] for group in optimizer.param_groups]
        logger.debug(f'Epoch:[{epoch + 1:>3d}/{cfg.epochs:>3d}]  Lr:[{curr_lr[0]:.5f}]')

        # train this epoch
        for it, sample in enumerate(train_loader):
            s = time.time()
            optimizer.zero_grad()
            idx = sample['index']
            x = sample['data'].to(device)
            y = sample['label'].to(device)
            # x, y = sample_selection(x, y, net)
            logits = net(x)['logits']

            losses, ce_loss = crssc_loss(logits, y, idx, T_k, epoch, memory_pool, eps, conf_metric)
            loss = losses.mean()
            memory_pool.update(indices=idx, losses=ce_loss.detach().data.cpu(), scores=F.softmax(logits, dim=1).detach().data.cpu(),
                               labels=y.detach().data.cpu())
            # loss = loss_func(logits, y)
            train_acc = accuracy(logits, y, topk=(1,))
            train_accuracy.update(train_acc[0], x.size(0))
            train_loss.update(loss.item(), x.size(0))
            if cfg.use_fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            epoch_train_time.update(time.time() - s, 1)
            if (cfg.log_freq is not None and (it + 1) % cfg.log_freq == 0) or (it + 1 == len(train_loader)):
                console_content = f"Epoch:[{epoch + 1:>3d}/{cfg.epochs:>3d}]  " \
                                  f"Iter:[{it + 1:>4d}/{len(train_loader):>4d}]  " \
                                  f"Train Accuracy:[{train_accuracy.avg:6.2f}]  " \
                                  f"Loss:[{train_loss.avg:4.4f}]  " \
                                  f"{epoch_train_time.avg:6.2f} sec/iter"
                logger.debug(console_content)

        # post-step in this epoch
        scheduler.step()

        # evaluate this epoch
        test_accuracy = evaluate(test_loader, net, device)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch + 1
            torch.save(net.state_dict(), f'{result_dir}/best_epoch.pth')

        # logging this epoch
        runtime = time.time() - start_time
        logger.info(f'epoch: {epoch + 1:>3d} | '
                    f'train loss: {train_loss.avg:>6.4f} | '
                    f'train accuracy: {train_accuracy.avg:>6.3f} | '
                    f'test accuracy: {test_accuracy:>6.3f} | '
                    f'epoch runtime: {runtime:6.2f} sec | '
                    f'best accuracy: {best_accuracy:6.3f} @ epoch: {best_epoch:03d}')
        plot_results(result_file=f'{result_dir}/log.txt')

    # rename results dir -----------------------------------------------------------------------------------------------------------------------------
    os.rename(result_dir, f'{result_dir}-bestAcc_{best_accuracy:.4f}')


def crssc_loss(logits, labels, indices, T_k, epoch, memory_pool, eps=0.1, certainty_measure='std', return_indices=False):
    ce_losses = label_smoothing_cross_entropy(logits, labels, epsilon=eps, reduction='none')

    # in the first T_k epochs, train with the entire training set
    if epoch < T_k:
        return ce_losses, ce_losses

    # after T_k epochs, start dividing training set into clean / uncertain / irrelevant
    ind_loss_sorted = torch.argsort(ce_losses.data)
    num_remember = torch.nonzero(ce_losses < ce_losses.mean()).shape[0]

    ind_clean = ind_loss_sorted[:num_remember]
    ind_forget = ind_loss_sorted[num_remember:]
    logits_clean = logits[ind_clean]
    labels_clean = labels[ind_clean]

    if ind_forget.shape[0] > 1:
        # for samples with high loss
        #   high loss, high std --> mislabeling
        #   high loss, low std  --> irrelevant category
        indices_forget = indices[ind_forget]
        logits_forget = logits[ind_forget]

        pred_distribution_forget = F.softmax(logits_forget, dim=1)
        pred_distribution_clean = F.softmax(logits_clean, dim=1)
        # high std: high confidence
        if certainty_measure == 'std':
            batch_cty = pred_distribution_forget.std(dim=1)
            flag = pred_distribution_clean.std(dim=1).mean().item()
        # low entropy: high confidence
        elif certainty_measure == 'entropy':
            batch_cty = - Categorical(probs=pred_distribution_forget).entropy()
            flag = - Categorical(probs=pred_distribution_clean).entropy().mean().item()
        # high peak to average ratio, high confidence
        else:
            batch_cty = pred_distribution_forget.max(dim=1)[0]
            flag = pred_distribution_clean.max(dim=1)[0].mean().item()

        batch_cty_sorted, ind_cty_sorted = torch.sort(batch_cty.data, descending=True)
        ind_split = split_set(batch_cty_sorted, flag)
        if ind_split is None:
            ind_split = -1

        # uncertain could be either mislabeled or hard example
        ind_uncertain = ind_cty_sorted[:(ind_split + 1)]

        logits_mislabeled = logits_forget[ind_uncertain]
        indices_mislabeled = indices_forget[ind_uncertain]
        labels_mislabeled = memory_pool.most_prob_labels[indices_mislabeled].to(logits_mislabeled.device)

        logits_final = torch.cat((logits_clean, logits_mislabeled), dim=0)
        labels_final = torch.cat((labels_clean, labels_mislabeled), dim=0)

        forget_indices = indices_forget.numpy().tolist()
        relabel_indices = indices_mislabeled.numpy().tolist()

    else:
        logits_final = logits_clean
        labels_final = labels_clean

        forget_indices, relabel_indices = [], []

    cty_losses = label_smoothing_cross_entropy(logits_final, labels_final, epsilon=eps, reduction='none')
    if return_indices:
        return cty_losses, ce_losses, forget_indices, relabel_indices
    else:
        return cty_losses, ce_losses


def split_set(x, flag):
    # split set based in interval
    # x shape is (N), x is sorted in descending
    # assert (x > 0).all()
    if x.shape[0] == 1:
        return None
    tmp = (x < flag).nonzero()
    if tmp.shape[0] == 0:
        return None
    else:
        return tmp[0, 0] - 1


def label_smoothing_cross_entropy(logit, label, epsilon=0.1, reduction='none'):
    N = label.size(0)
    C = logit.size(1)
    smoothed_label = torch.full(size=(N, C), fill_value=epsilon / (C - 1))
    smoothed_label.scatter_(dim=1, index=torch.unsqueeze(label, dim=1).cpu(), value=1 - epsilon)
    if logit.is_cuda:
        smoothed_label = smoothed_label.cuda()

    log_logit = F.log_softmax(logit, dim=1)
    losses = -torch.sum(log_logit * smoothed_label, dim=1)  # (N)
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / N
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--log_prefix', type=str)
    parser.add_argument('--log_freq', type=int)
    parser.add_argument('--net', type=str, default='resnet18')
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--T_k', type=int, default=10)
    parser.add_argument('--mem_len', type=int, default=10)
    parser.add_argument('--eps', type=float, default=0.2)
    parser.add_argument('--conf_metric', type=str, default='std')

    args = parser.parse_args()

    config = load_from_cfg(args.config)
    override_config_items = [k for k, v in args.__dict__.items() if k != 'config' and v is not None]
    for item in override_config_items:
        config.set_item(item, args.__dict__[item])
    # config.log = f'{config.net}-{config.noise_type}_closeset{config.closeset_ratio}_openset{config.openset_ratio}-crssc'
    config.log = f'{config.net}-clar_crssc'
    print(config)
    return config


if __name__ == '__main__':
    params = parse_args()
    dev = set_device(params.gpu)
    script_start_time = time.time()
    main(params, dev)
    script_runtime = time.time() - script_start_time
    print(f'Runtime of this script {str(pathlib.Path(__file__))} : {script_runtime:.1f} seconds ({script_runtime/3600:.3f} hours)')
