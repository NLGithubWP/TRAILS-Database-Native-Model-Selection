import time

import torch
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
        num_labels = args.num_labels
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
        loss_avg = utils.AvgrageMeter()
        _begin_time = time.time()

        cur_iter = 0
        cur_epoch = 0

        num_correct = 0
        num_total = 0
        for _, batch in enumerate(train_loader):
            cur_iter += 1
            if cur_iter > iter_per_epoch:
                cur_epoch += 1

            if cur_epoch > epoch_num:
                break
            # 2.1 get data
            target = batch['y'].type(torch.LongTensor)
            batch['id'] = batch['id'].to(device)
            batch['value'] = batch['value'].to(device)
            target = target.to(device)
            num_total += val_loader.batch_size

            # 2.2 train
            y = model(batch)
            loss = opt_metric(y, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 2.3 report loss and data
            loss_avg.update(loss.item(), target.size(0))
            time_avg.update(time.time() - timestamp)
            timestamp = time.time()
            num_correct += utils.get_correct_num(y, target)

            if cur_iter % report_freq == 0:
                logger.info(f'model id: {arch_id}, Training: Epoch [{cur_epoch:3d}/{epoch_num}][{cur_iter:3d}/{len(train_loader)}]\t'
                            f'{time_avg.val:.3f} ({time_avg.avg:.3f}) Acc {num_correct/num_total:4f} '
                            f'Loss {loss_avg.val:8.4f} ({loss_avg.avg:8.4f})')

        # 3. after training, do the validation
        time_avg, timestamp = utils.AvgrageMeter(), time.time()
        loss_avg = utils.AvgrageMeter()

        num_correct = 0
        num_total = 0
        for cur_iter, batch in enumerate(val_loader):

            # 3.1 get data
            target = batch['y'].type(torch.LongTensor)
            batch['id'] = batch['id'].to(device)
            batch['value'] = batch['value'].to(device)
            target = target.to(device)

            num_total += val_loader.batch_size

            # 3.2 conduct the evaluation
            with torch.no_grad():
                y = model(batch)
                loss = opt_metric(y, target)

                loss_avg.update(loss.item(), target.size(0))
                num_correct += utils.get_correct_num(y, target)

                time_avg.update(time.time() - timestamp)
                timestamp = time.time()

                if cur_iter % report_freq == 0:
                    logger.info(f'model id: {arch_id}, Validation: Iteration [{cur_iter*val_loader.batch_size:3d}/{len(val_loader)}]\t'
                                f'{time_avg.val:.3f} ({time_avg.avg:.3f}) Acc {num_correct/num_total:4f} '
                                f'Loss {loss_avg.val:8.4f} ({loss_avg.avg:8.4f})')

        # todo: use the auc as performance indictor or for multi-class?
        return num_correct/num_total, time.time() - _begin_time


