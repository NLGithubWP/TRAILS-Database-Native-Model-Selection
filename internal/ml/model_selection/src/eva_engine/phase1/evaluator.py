# this is for checking the flops and params
try:
    from thop import profile
except:
    pass
from src.common.constant import Config, CommonVars
from src.common.structure import ModelAcquireData
from src.eva_engine import evaluator_register
from src.query_api.interface import SimulateScore
from src.dataset_utils import dataset
from torch.utils.data import DataLoader
import torch


class P1Evaluator:

    def __init__(self, device: str, num_label: int, dataset_name: str, search_space_ins,
                 train_loader: DataLoader, is_simulate: bool, metrics: str = CommonVars.ExpressFlow):
        """
        :param device:
        :param num_label:
        :param dataset_name:
        :param search_space_ins:
        :param search_space_ins:
        :param train_loader:
        :param is_simulate:
        :param metrics: which TFMEM to use?
        """
        self.metrics = metrics
        self.is_simulate = is_simulate

        self.dataset_name = dataset_name

        self.search_space_ins = search_space_ins
        self.train_loader = train_loader

        self.device = device
        self.num_labels = num_label

        self.score_getter = None

        # get one mini batch
        if self.dataset_name in [Config.c10, Config.c100, Config.imgNet]:
            # for img data
            self.mini_batch, self.mini_batch_targets = dataset.get_mini_batch(
                dataloader=self.train_loader,
                sample_alg="random",
                batch_size=32,
                num_classes=self.num_labels)
            self.mini_batch.to(self.device)
            self.mini_batch_targets.to(self.device)
        elif self.dataset_name in [Config.Criteo, Config.Frappe, Config.UCIDataset]:
            # this is structure data
            batch = iter(self.train_loader).__next__()
            target = batch['y'].type(torch.LongTensor).to(self.device)
            batch['id'] = batch['id'].to(self.device)
            batch['value'] = batch['value'].to(self.device)
            self.mini_batch = batch
            self.mini_batch_targets = target.to(self.device)
        else:
            raise NotImplementedError

    def p1_evaluate(self, data_str: str) -> dict:
        """
        :param data_str: encoded ModelAcquireData
        :return:
        """

        model_acquire = ModelAcquireData.deserialize(data_str)

        if self.is_simulate:
            return self._p1_evaluate_simu(model_acquire)
        else:
            return self._p1_evaluate_online(model_acquire)

    def measure_model_flops(self, data_str: str, batch_size: int, channel_size: int):
        # todo: check the package
        model_acquire = ModelAcquireData.deserialize(data_str)
        model_encoding = model_acquire.model_encoding
        new_model = self.search_space_ins.new_arch_scratch_with_default_setting(model_encoding, bn=True)
        new_model = new_model.to(self.device)
        flops, params = profile(new_model, inputs=(self.mini_batch,))
        print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
        print('Params = ' + str(params / 1000 ** 2) + 'M')

    def _p1_evaluate_online(self, model_acquire: ModelAcquireData) -> dict:

        # # 1. Score NasWot
        # new_model = self.search_space_ins.new_arch_scratch_with_default_setting(model_encoding, bn=True)
        # new_model = new_model.to(self.device)
        # naswot_score, _ = evaluator_register[CommonVars.NAS_WOT].evaluate_wrapper(
        #     arch=new_model,
        #     device=self.device,
        #     batch_data=self.mini_batch,
        #     batch_labels=self.mini_batch_targets)
        #
        # # 2. Score SynFlow
        # new_model = self.search_space_ins.new_arch_scratch_with_default_setting(model_encoding, bn=False)
        # new_model = new_model.to(self.device)
        # synflow_score, _ = evaluator_register[CommonVars.PRUNE_SYNFLOW].evaluate_wrapper(
        #     arch=new_model,
        #     device=self.device,
        #     batch_data=self.mini_batch,
        #     batch_labels=self.mini_batch_targets)
        #
        # # 3. combine the result and return
        # model_score = {CommonVars.NAS_WOT: naswot_score,
        #                CommonVars.PRUNE_SYNFLOW: synflow_score}

        model_encoding = model_acquire.model_encoding
        # score all tfmem
        if self.metrics == CommonVars.ALL_EVALUATOR:
            model_score = {}
            for alg, score_evaluator in evaluator_register.items():
                if alg == CommonVars.PRUNE_SYNFLOW or alg == CommonVars.ExpressFlow:
                    bn = False
                else:
                    bn = True
                new_model = self.search_space_ins.new_arch_scratch_with_default_setting(model_encoding, bn=bn)
                new_model = new_model.to(self.device)
                naswot_score, _ = score_evaluator.evaluate_wrapper(
                    arch=new_model,
                    device=self.device,
                    batch_data=self.mini_batch,
                    batch_labels=self.mini_batch_targets)
                model_score[alg] = naswot_score

                # clear the cache
                del new_model
                if "cuda" in self.device:
                    torch.cuda.empty_cache()
        else:
            # score using only one metrics
            if self.metrics == CommonVars.PRUNE_SYNFLOW or self.metrics == CommonVars.ExpressFlow:
                bn = False
            else:
                bn = True
            new_model = self.search_space_ins.new_arch_scratch_with_default_setting(model_encoding, bn=bn)
            new_model = new_model.to(self.device)

            _score, _ = evaluator_register[self.metrics].evaluate_wrapper(
                arch=new_model,
                device=self.device,
                batch_data=self.mini_batch,
                batch_labels=self.mini_batch_targets)

            model_score = {self.metrics: _score}

            del new_model
            if "cuda" in self.device:
                torch.cuda.empty_cache()

        return model_score

    def _p1_evaluate_simu(self, model_acquire: ModelAcquireData) -> dict:
        if self.score_getter is None:
            self.score_getter = SimulateScore(space_name=self.search_space_ins.name,
                                              dataset_name=self.dataset_name)

        model_score = self.score_getter.query_tfmem_rank_score(arch_id=model_acquire.model_id)

        return model_score
