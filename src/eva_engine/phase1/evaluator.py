

import torch
from common.constant import Config, CommonVars
from common.structure import ModelAcquireData
from eva_engine import evaluator_register
from search_space.mlp_api.model_params import MlpMacroCfg
from search_space.mlp_api.space import MlpSpace, DEFAULT_LAYER_CHOICES_20
from search_space.nas_101_api.model_params import NB101MacroCfg
from search_space.nas_101_api.space import NasBench101Space
from search_space.nas_201_api.model_params import NB201MacroCfg
from search_space.nas_201_api.space import NasBench201Space

from storage import dataset
from torch.utils.data import DataLoader


class P1Evaluator:

    def __init__(self, args, train_loader: DataLoader):
        self.args = args
        self.search_space = args.search_space
        self.device = args.device
        self.train_loader = train_loader
        self.num_labels = args.num_labels

    def p1_evaluate(self, data_str: str) -> dict:
        """
        :param data_str: encoded ModelAcquireData
        :return:
        """

        model_acquire = ModelAcquireData.deserialize(data_str)
        return self._p1_evaluate(model_acquire.model_encoding)

    def _p1_evaluate(self, model_encoding: str) -> dict:

        # 0. identify the load model method
        cfg_load_method = None

        # get a random batch.

        if self.train_loader.dataset in [Config.c10, Config.c100, Config.imgNet]:
            # for img data
            mini_batch, mini_batch_targets = dataset.get_mini_batch(
                dataloader=self.train_loader,
                sample_alg="random",
                batch_size=32,
                num_classes=self.num_labels)
        else:
            # this is structure data
            batch = iter(self.train_loader).__next__()
            target = batch['y'].type(torch.LongTensor)
            batch['id'] = batch['id'].to(self.device)
            mini_batch = batch['value'].to(self.device)
            mini_batch_targets = target.to(self.device)

        if self.search_space == Config.NB101:
            cfg_load_method = self._load_101_cfg
        elif self.search_space == Config.NB201:
            cfg_load_method = self._load_201_cfg
        elif self.search_space == Config.MLPSP:
            cfg_load_method = self._load_mlp_cfg
        else:
            raise NotImplementedError

        # 1. Score NasWot
        # new_model = cfg_load_method(model_encoding, bn=True)
        # new_model = new_model.to(self.device)
        # naswot_score, _ = evaluator_register[CommonVars.NAS_WOT].evaluate_wrapper(
        #     arch=new_model,
        #     device=self.device,
        #     batch_data=mini_batch,
        #     batch_labels=mini_batch_targets)

        # 2. Score SynFlow
        new_model = cfg_load_method(model_encoding, bn=False)
        new_model = new_model.to(self.device)
        synflow_score, _ = evaluator_register[CommonVars.PRUNE_SYNFLOW].evaluate_wrapper(
            arch=new_model,
            device=self.device,
            batch_data=mini_batch,
            batch_labels=mini_batch_targets)

        # 3. combine the result and return
        model_score = {CommonVars.NAS_WOT: 0,
                       CommonVars.PRUNE_SYNFLOW: synflow_score}

        return model_score

    def _load_101_cfg(self, model_encoding: str, bn: bool):
        model_cfg = NB101MacroCfg(
            bn=bn,
            init_channels=16,
            num_stacks=3,
            num_modules_per_stack=3,
            num_labels=self.num_labels
        )
        model_micro = NasBench101Space.deserialize_model_encoding(model_encoding)
        return NasBench101Space.new_arch_scratch(model_cfg, model_micro)

    def _load_201_cfg(self, model_encoding: str, bn: bool):
        model_cfg = NB201MacroCfg(
            bn=bn,
            init_channels=16,
            init_b_type="none",
            init_w_type="none",
            num_labels=self.num_labels
        )
        model_micro = NasBench201Space.deserialize_model_encoding(model_encoding)
        return NasBench201Space.new_arch_scratch(model_cfg, model_micro)

    def _load_mlp_cfg(self, model_encoding: str, bn: bool):
        model_cfg = MlpMacroCfg(
            bn=bn,
            input_fea_dims=self.args.init_channels,
            layer_choices=DEFAULT_LAYER_CHOICES_20,
            num_layers=self.args.num_layers,
            num_labels=self.num_labels
        )
        model_micro = MlpSpace.deserialize_model_encoding(model_encoding)
        return MlpSpace.new_arch_scratch(model_cfg, model_micro)

