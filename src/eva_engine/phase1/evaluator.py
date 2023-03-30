

import torch
from common.constant import Config, CommonVars
from common.structure import ModelAcquireData
from eva_engine import evaluator_register
from eva_engine.phase2.algo.api_query import SimulateScore
from storage import dataset
from torch.utils.data import DataLoader


class P1Evaluator:

    def __init__(self, device: str, num_label: int, dataset_name: str, search_space_ins,
                 train_loader: DataLoader, is_simulate: bool):

        self.is_simulate = is_simulate

        self.dataset_name = dataset_name

        self.search_space_ins = search_space_ins
        self.train_loader = train_loader

        self.device = device
        self.num_labels = num_label

        self.score_getter = None

    def p1_evaluate(self, data_str: str) -> dict:
        """
        :param data_str: encoded ModelAcquireData
        :return:
        """

        model_acquire = ModelAcquireData.deserialize(data_str)

        if self.is_simulate:
            return self._p1_evaluate_simu(model_acquire.model_encoding)
        else:
            return self._p1_evaluate_online(model_acquire.model_encoding)

    def _p1_evaluate_online(self, model_encoding: str) -> dict:

        # load the data loader
        if self.dataset_name in [Config.c10, Config.c100, Config.imgNet]:
            # for img data
            mini_batch, mini_batch_targets = dataset.get_mini_batch(
                dataloader=self.train_loader,
                sample_alg="random",
                batch_size=32,
                num_classes=self.num_labels)
        elif self.dataset_name in [Config.Criteo, Config.Frappe, Config.UCIDataset]:
            # this is structure data
            batch = iter(self.train_loader).__next__()
            target = batch['y'].type(torch.LongTensor)
            # target = batch['y'].to(self.device)
            batch['id'] = batch['id'].to(self.device)
            batch['value'] = batch['value'].to(self.device)
            mini_batch = batch
            mini_batch_targets = target.to(self.device)
        else:
            raise NotImplementedError

        model_score = {}
        for alg, score_evaluator in evaluator_register.items():
            if alg == CommonVars.PRUNE_SYNFLOW:
                bn = False
            else:
                bn = True
            new_model = self.search_space_ins.new_arch_scratch_with_default_setting(model_encoding, bn=bn)
            new_model = new_model.to(self.device)
            naswot_score, _ = score_evaluator.evaluate_wrapper(
                arch=new_model,
                device=self.device,
                batch_data=mini_batch,
                batch_labels=mini_batch_targets)
            model_score[alg] = naswot_score

        # 1. Score NasWot
        # new_model = self.search_space_ins.new_arch_scratch_with_default_setting(model_encoding, bn=True)
        # new_model = new_model.to(self.device)
        # naswot_score, _ = evaluator_register[CommonVars.NAS_WOT].evaluate_wrapper(
        #     arch=new_model,
        #     device=self.device,
        #     batch_data=mini_batch,
        #     batch_labels=mini_batch_targets)

        # 2. Score SynFlow
        # new_model = self.search_space_ins.new_arch_scratch_with_default_setting(model_encoding, bn=False)
        # new_model = new_model.to(self.device)
        # synflow_score, _ = evaluator_register[CommonVars.PRUNE_SYNFLOW].evaluate_wrapper(
        #     arch=new_model,
        #     device=self.device,
        #     batch_data=mini_batch,
        #     batch_labels=mini_batch_targets)
        #
        # # 3. combine the result and return
        # model_score = {CommonVars.NAS_WOT: naswot_score,
        #                CommonVars.PRUNE_SYNFLOW: synflow_score}

        return model_score

    def _p1_evaluate_simu(self, model_encoding: str) -> dict:
        if self.score_getter is None:
            self.score_getter = SimulateScore(space_name=self.search_space_ins.name)
        model_score = self.score_getter.get_score_res(arch_id=model_encoding,  dataset=self.dataset_name)
        return model_score

