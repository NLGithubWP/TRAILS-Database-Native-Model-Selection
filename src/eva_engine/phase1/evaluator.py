

import torch
from common.constant import Config, CommonVars
from common.structure import ModelAcquireData
from eva_engine import evaluator_register
from search_space.mlp_api.model_params import MlpMacroCfg
from search_space.mlp_api.space import MlpSpace, MlpMicroCfg
from search_space.nas_101_api.model_params import NB101MacroCfg
from search_space.nas_101_api.space import NasBench101Space
from search_space.nas_201_api.model_params import NB201MacroCfg
from search_space.nas_201_api.space import NasBench201Space

from storage import dataset
from torch.utils.data import DataLoader


class P1Evaluator:

    def __init__(self, device, num_label, dataset_name, search_space_ins, train_loader: DataLoader):

        self.dataset_name = dataset_name

        self.search_space_ins = search_space_ins
        self.train_loader = train_loader

        self.device = device
        self.num_labels = num_label

    def p1_evaluate(self, data_str: str) -> dict:
        """
        :param data_str: encoded ModelAcquireData
        :return:
        """

        model_acquire = ModelAcquireData.deserialize(data_str)
        return self._p1_evaluate(model_acquire.model_encoding)

    def _p1_evaluate(self, model_encoding: str) -> dict:

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
            batch['id'] = batch['id'].to(self.device)
            batch['value'] = batch['value'].to(self.device)
            mini_batch = batch
            mini_batch_targets = target.to(self.device)
        else:
            raise NotImplementedError

        # 1. Score NasWot
        new_model = self.search_space_ins.new_arch_scratch_with_default_setting(model_encoding, bn=True)
        new_model = new_model.to(self.device)
        naswot_score, _ = evaluator_register[CommonVars.NAS_WOT].evaluate_wrapper(
            arch=new_model,
            device=self.device,
            batch_data=mini_batch,
            batch_labels=mini_batch_targets)

        # 2. Score SynFlow
        new_model = self.search_space_ins.new_arch_scratch_with_default_setting(model_encoding, bn=False)
        new_model = new_model.to(self.device)
        synflow_score, _ = evaluator_register[CommonVars.PRUNE_SYNFLOW].evaluate_wrapper(
            arch=new_model,
            device=self.device,
            batch_data=mini_batch,
            batch_labels=mini_batch_targets)

        # 3. combine the result and return
        model_score = {CommonVars.NAS_WOT: naswot_score,
                       CommonVars.PRUNE_SYNFLOW: synflow_score}

        return model_score
