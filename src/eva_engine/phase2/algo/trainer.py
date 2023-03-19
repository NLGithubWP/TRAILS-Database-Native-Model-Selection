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
                         dataset: str,
                         epoch_num,
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         args
                         ) -> (float, float):
        """
        Args:
            val_loader:
            train_loader:
            args:
            search_space_ins: search space to use
            arch_id: arch id to init the model
            dataset:
            epoch_num:
        Returns:
        """

        # training params
        device = args.device
        num_labels = args.num_labels
        lr = args.lr
        iter_per_epoch = args.iter_per_epoch
        report_freq = args.report_freq

        # 1. get search space
        model = search_space_ins.new_architecture(arch_id).to(device)
        opt_metric = nn.CrossEntropyLoss(reduction='mean')
        opt_metric = opt_metric.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.iter_per_epoch,  # Maximum number of iterations.
            eta_min=1e-4)  # Minimum learning rate.

        # 2. training
        time_avg, timestamp = utils.AvgrageMeter(), time.time()
        loss_avg, auc_avg = utils.AvgrageMeter(), utils.AvgrageMeter()
        _begin_time = time.time()

        for cur_epoch in range(epoch_num):
            for cur_iter, batch in enumerate(train_loader):
                # 2.1 get data
                target = batch['y'].type(torch.LongTensor).to(device)
                batch['id'] = batch['id'].to(device)
                batch['value'] = batch['value'].to(device)

                # 2.2 train
                y = model(batch)
                loss = opt_metric(y, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # 2.3 report loss and data
                auc = utils.roc_auc_compute_fn(y, target)
                loss_avg.update(loss.item(), target.size(0))
                auc_avg.update(auc, target.size(0))
                time_avg.update(time.time() - timestamp)
                timestamp = time.time()
                if cur_iter % report_freq == 0 or cur_iter == iter_per_epoch:
                    logger.info(f'model id: {arch_id}, Training: Epoch [{cur_epoch:3d}/{epoch_num}][{cur_iter:3d}/{len(train_loader)}]\t'
                                f'{time_avg.val:.3f} ({time_avg.avg:.3f}) AUC {auc_avg.val:4f} ({auc_avg.avg:4f}) '
                                f'Loss {loss_avg.val:8.4f} ({loss_avg.avg:8.4f})')

        # 3. after training, do the validation
        time_avg, timestamp = utils.AvgrageMeter(), time.time()
        loss_avg, auc_avg = utils.AvgrageMeter(), utils.AvgrageMeter()

        # for calculate F1 score
        label_all = []
        prob_all = []
        for cur_iter, batch in enumerate(val_loader):

            # 3.1 get data
            target = batch['y'].type(torch.LongTensor).to(device)
            batch['id'] = batch['id'].to(device)
            batch['value'] = batch['value'].to(device)

            # 3.2 conduct the evaluation
            with torch.no_grad():
                y = model(batch)
                loss = opt_metric(y, target)

                # measure F1 score
                label_all.extend(target.tolist())
                prob_all.extend(torch.argmax(y, dim=1).tolist())

                # measure time, loss and num correct
                auc = utils.roc_auc_compute_fn(y, target)
                loss_avg.update(loss.item(), target.size(0))
                auc_avg.update(auc, target.size(0))
                time_avg.update(time.time() - timestamp)
                timestamp = time.time()

                if cur_iter % report_freq == 0 or cur_iter == len(val_loader)-1:
                    logger.info(f'model id: {arch_id}, Validation: Iteration [{cur_iter:3d}/{len(val_loader)}]\t'
                                f'{time_avg.val:.3f} ({time_avg.avg:.3f}) AUC {auc_avg.val:4f} ({auc_avg.avg:4f}) '
                                f'Loss {loss_avg.val:8.4f} ({loss_avg.avg:8.4f})')

        f1_res = f1_score(label_all, prob_all)
        logger.info(f' ----- model id: {arch_id}, F1-Score : {f1_res} -----')
        return f1_res, time.time() - _begin_time


