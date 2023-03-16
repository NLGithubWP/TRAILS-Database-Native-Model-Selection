import time

import torch.nn as nn
from torch import optim
from logger import logger
from search_space.core.space import SpaceWrapper
from torch.utils.data import DataLoader
from utilslibs import utils


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
        num_labels = args.num_label
        lr = args.lr
        iter_per_epoch = args.iter_per_epoch
        report_freq = args.report_freq

        # 1. get search space
        model = search_space_ins.new_architecture(arch_id).to(device)

        # optimizer
        opt_metric = nn.CrossEntropyLoss(reduction='mean')
        opt_metric = opt_metric.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 2. training
        time_avg, timestamp = utils.AvgrageMeter(), time.time()
        loss_avg, auc_avg = utils.AvgrageMeter(), utils.AvgrageMeter()
        _begin_time = time.time()

        cur_iter = 0
        cur_epoch = 0
        for _, batch in enumerate(train_loader):
            cur_iter += 1
            if cur_iter > 200:
                cur_epoch += 1

            if cur_epoch > epoch_num:
                break

            target = batch['y']
            batch['id'] = batch['id'].to(device)
            batch['value'] = batch['value'].to(device)
            target = target.to(device)

            # measure training for one epoch time
            y = model(batch['value'])
            loss = opt_metric(y, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # report the loss
            auc = utils.roc_auc_compute_fn(y, target)
            loss_avg.update(loss.item(), target.size(0))
            auc_avg.update(auc, target.size(0))

            time_avg.update(time.time() - timestamp)
            timestamp = time.time()
            if cur_iter % report_freq == 0:
                logger.info(f'Epoch [{cur_epoch:3d}/{epoch_num}][{cur_iter:3d}/{len(train_loader)}]\t'
                             f'{time_avg.val:.3f} ({time_avg.avg:.3f}) AUC {auc_avg.val:4f} ({auc_avg.avg:4f}) '
                             f'Loss {loss_avg.val:8.4f} ({loss_avg.avg:8.4f})')

        # 3. after training, do the validation
        # time_avg, timestamp = utils.AvgrageMeter(), time.time()
        # loss_avg, auc_avg = utils.AvgrageMeter(), utils.AvgrageMeter()
        # _begin_time = time.time()
        # for batch_idx, batch in enumerate(val_loader):
        #     target = batch['y']
        #     batch['id'] = batch['id'].to(device)
        #     batch['value'] = batch['value'].to(device)
        #     target = target.to(device)
        #
        #     with torch.no_grad():
        #         y = model(batch['value'])
        #         loss = opt_metric(y, target)
        #
        #         # report the loss
        #         auc = utils.roc_auc_compute_fn(y, target)
        #         loss_avg.update(loss.item(), target.size(0))
        #         auc_avg.update(auc, target.size(0))
        #
        #         time_avg.update(time.time() - timestamp)
        #         timestamp = time.time()
        #         if cur_iter % report_freq == 0:
        #             logger.info(f'Epoch [{cur_epoch:3d}/{epoch_num}][{cur_iter:3d}/{len(train_loader)}]\t'
        #                         f'{time_avg.val:.3f} ({time_avg.avg:.3f}) AUC {auc_avg.val:4f} ({auc_avg.avg:4f}) '
        #                         f'Loss {loss_avg.val:8.4f} ({loss_avg.avg:8.4f})')

        # todo, use training loss as performance indicator ?
        return auc_avg.avg, time.time() - _begin_time

