import time

import torch
import torch.nn as nn
from torch import optim
from logger import logger
from search_space.core.space import SpaceWrapper
from torch.utils.data import DataLoader
from utilslibs import utils
from sklearn.metrics import f1_score


class ModelTrainer:

    @classmethod
    def fully_train_arch(cls,
                         search_space_ins: SpaceWrapper,
                         arch_id: str,
                         use_test_acc: bool,
                         epoch_num,
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         test_loader: DataLoader,
                         args):
        """
        Args:
            search_space_ins:
            arch_id:
            use_test_acc:
            epoch_num: how many epoch, set by scheduler
            train_loader:
            val_loader:
            test_loader:
            args:
        Returns:
        """

        start_time, best_valid_auc = time.time(), 0.

        # training params
        device = args.device
        num_labels = args.num_labels
        lr = args.lr
        iter_per_epoch = args.iter_per_epoch
        report_freq = args.report_freq
        given_patience = args.patience

        # assign new values
        args.epoch_num = epoch_num

        # create model
        model = search_space_ins.new_architecture(arch_id).to(device)
        # optimizer

        # for multiple classification
        # opt_metric = nn.CrossEntropyLoss(reduction='mean')
        opt_metric = nn.BCEWithLogitsLoss(reduction='mean')
        opt_metric = opt_metric.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(train_loader) * epoch_num,  # Maximum number of iterations.
            eta_min=1e-4)  # Minimum learning rate.

        # gradient clipping, set the gradient value to be -1 - 1
        for p in model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -1., 1.))

        # for calculate F1 score at the validation datasets
        label_all = []
        prob_all = []

        patience_cnt = 0
        for epoch in range(epoch_num):
            logger.info(f'Epoch [{epoch:3d}/{epoch_num:3d}]')
            # train and eval
            ModelTrainer.run(epoch, iter_per_epoch, model, train_loader, opt_metric,
                             args, optimizer=optimizer, namespace='train')
            scheduler.step()
            valid_auc, label_per_epoch, prob_per_epoch = ModelTrainer.run(epoch, iter_per_epoch, model, val_loader,
                                                                          opt_metric,
                                                                          args, namespace='val')

            if use_test_acc:
                test_auc, label_per_epoch, prob_per_epoch = ModelTrainer.run(epoch, iter_per_epoch, model, test_loader,
                                                                             opt_metric,
                                                                             args, namespace='test')
            else:
                test_auc = -1

            label_all.extend(label_per_epoch)
            prob_all.extend(prob_per_epoch)

            # record best aue and save checkpoint
            if valid_auc >= best_valid_auc:
                patience_cnt = 0
                best_valid_auc, best_test_auc = valid_auc, test_auc
                logger.info(f'best valid auc: valid {valid_auc:.4f}, test {test_auc:.4f}')
            else:
                patience_cnt += 1
                logger.info(f'valid {valid_auc:.4f}, test {test_auc:.4f}')
                logger.info(f'Early stopped, {patience_cnt}-th best auc at epoch {epoch - 1}')
            if patience_cnt >= given_patience:
                logger.info(f'Final best valid auc {best_valid_auc:.4f}, with test auc {best_test_auc:.4f}')
                break

        # for multiple classification
        # target = torch.tensor([0, 1, 2, 0, 1, 2])
        # preds = torch.tensor([0, 2, 1, 0, 0, 1])
        # f1 = F1Score(task="multiclass", num_classes=3)
        # f1(preds, target)

        f1_res = f1_score(label_all, prob_all)
        logger.info(f' ----- model id: {arch_id}, F1-Score : {f1_res} Total running time: {utils.timeSince(since=start_time)}-----')
        return f1_res, time.time() - start_time

    #  train one epoch of train/val/test
    @classmethod
    def run(cls, epoch, iter_per_epoch, model, data_loader, opt_metric, args, optimizer=None, namespace='train'):
        if optimizer:
            model.train()
        else:
            model.eval()

        label_per_epoch = []
        prob_per_epoch = []

        time_avg, timestamp = utils.AvgrageMeter(), time.time()
        loss_avg, auc_avg = utils.AvgrageMeter(), utils.AvgrageMeter()

        for batch_idx, batch in enumerate(data_loader):
            # if suer set this, then only train fix number of iteras
            if namespace == 'train' and iter_per_epoch is not None and batch_idx >= iter_per_epoch:
                break

            target = batch['y'].to(args.device)
            batch['id'] = batch['id'].to(args.device)
            batch['value'] = batch['value'].to(args.device)

            if namespace == 'train':
                y = model(batch)
                loss = opt_metric(y, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    y = model(batch)
                    loss = opt_metric(y, target)

                    # measure F1 score
                    label_per_epoch.extend(target.tolist())
                    # for multiple classification
                    # prob_per_epoch.extend(torch.argmax(y, dim=1).tolist())
                    prob_per_epoch.extend(y.detach().numpy().round().tolist())

            # for multiple classification
            # auc = utils.roc_auc_compute_fn(torch.nn.functional.softmax(y, dim=1)[:, 1], target)
            auc = utils.roc_auc_compute_fn(y, target)
            loss_avg.update(loss.item(), target.size(0))
            auc_avg.update(auc, target.size(0))

            time_avg.update(time.time() - timestamp)
            timestamp = time.time()
            if batch_idx % args.report_freq == 0:
                logger.info(f'Epoch [{epoch:3d}/{args.epoch_num}][{batch_idx:3d}/{len(data_loader)}]\t'
                            f'{time_avg.val:.3f} ({time_avg.avg:.3f}) AUC {auc_avg.val:4f} ({auc_avg.avg:4f}) '
                            f'Loss {loss_avg.val:8.4f} ({loss_avg.avg:8.4f})')

            # stop training current epoch for evaluation
            if batch_idx >= args.eval_freq: break

        logger.info(f'{namespace}\tTime {utils.timeSince(s=time_avg.sum):>12s} '
                    f'AUC {auc_avg.avg:8.4f} Loss {loss_avg.avg:8.4f}')
        return auc_avg.avg, label_per_epoch, prob_per_epoch

