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
import time
from torch import nn
from src.search_space.core.space import SpaceWrapper
import gc
import sys


class P1Evaluator:

    def __init__(self, device: str, num_label: int, dataset_name: str,
                 search_space_ins: SpaceWrapper,
                 train_loader: DataLoader, is_simulate: bool, metrics: str = CommonVars.ExpressFlow,
                 enable_cache: bool = False):
        """
        :param device:
        :param num_label:
        :param dataset_name:
        :param search_space_ins:
        :param search_space_ins:
        :param train_loader:
        :param is_simulate:
        :param metrics: which TFMEM to use?
        :param enable_cache: if cache embedding for scoring? only used on structued data
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
        if not self.is_simulate:
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
        self.processed_mini_batch = None

        self.time_usage = {
            "latency": 0.0,
            "io_latency": 0.0,
            "compute_latency": 0.0,
            "track_compute": [],  # compute time
            "track_io_model_init": [],  # init model weight
            "track_io_model_load": [],  # load model into GPU/CPU
            "track_io_res_load": [],    # load result into GPU/CPU
            "track_io_model_release_each_50": [],  # context switch
            "track_io_model_release": [],  # context switch
            "track_io_data": [],  # context switch
        }

        # this is to do the expeirment
        self.enable_cache = enable_cache
        self.model_cache = None

        # for gc
        self.explored_model = []

    def if_cuda_avaiable(self):
        if "cuda" in self.device:
            return True
        else:
            return False

    def p1_evaluate(self, data_str: str) -> dict:
        """
        :param data_str: encoded ModelAcquireData
        :return:
        """

        model_acquire = ModelAcquireData.deserialize(data_str)

        if self.is_simulate:
            if self.metrics == "jacflow":
                return self._p1_evaluate_simu_jacflow(model_acquire)
            else:
                return self._p1_evaluate_simu(model_acquire)
        else:
            return self._p1_evaluate_online(model_acquire)

    def measure_model_flops(self, data_str: str, batch_size: int, channel_size: int):
        # todo: check the package
        model_acquire = ModelAcquireData.deserialize(data_str)
        model_encoding = model_acquire.model_encoding
        new_model = self.search_space_ins.new_arch_scratch_with_default_setting(model_encoding, bn=True)
        if self.search_space_ins.name == Config.MLPSP:
            new_model.init_embedding(requires_grad=True)
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
        #     space_name = self.search_space_ins.name,
        #     batch_data=self.mini_batch,
        #     batch_labels=self.mini_batch_targets)
        #
        # # 2. Score SynFlow
        # new_model = self.search_space_ins.new_arch_scratch_with_default_setting(model_encoding, bn=False)
        # new_model = new_model.to(self.device)
        # synflow_score, _ = evaluator_register[CommonVars.PRUNE_SYNFLOW].evaluate_wrapper(
        #     arch=new_model,
        #     device=self.device,
        #     space_name = self.search_space_ins.name,
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
                if self.search_space_ins.name == Config.MLPSP:
                    new_model.init_embedding()
                new_model = new_model.to(self.device)

                # self.explored_model.append(new_model)

                mini_batch = self.data_pre_processing(alg, new_model)

                _score, _ = score_evaluator.evaluate_wrapper(
                    arch=new_model,
                    device=self.device,
                    space_name=self.search_space_ins.name,
                    batch_data=mini_batch,
                    batch_labels=self.mini_batch_targets)

                _score = _score.item()
                model_score[alg] = abs(_score)

                # clear the cache
                if "cuda" in self.device:
                    torch.cuda.empty_cache()
        else:
            # score using only one metrics
            if self.metrics == CommonVars.PRUNE_SYNFLOW or self.metrics == CommonVars.ExpressFlow:
                bn = False
            else:
                bn = True

            # measure model load time
            begin = time.time()
            new_model = self.search_space_ins.new_arch_scratch_with_default_setting(model_encoding, bn=bn)

            # mlp have embedding layer, which can be cached, optimization!
            if self.search_space_ins.name == Config.MLPSP:
                if self.enable_cache:
                    new_model.init_embedding(self.model_cache)
                    if self.model_cache is None:
                        self.model_cache = new_model.embedding.to(self.device)
                else:
                    # init embedding every time created a new model
                    new_model.init_embedding()

            # self.explored_model.append(new_model)

            self.time_usage["track_io_model_init"].append(time.time() - begin)

            if self.if_cuda_avaiable():
                begin = time.time()
                new_model = new_model.to(self.device)
                torch.cuda.synchronize()
                self.time_usage["track_io_model_load"].append(time.time() - begin)
            else:
                self.time_usage["track_io_model_load"].append(0)

            # measure data load time
            begin = time.time()
            mini_batch = self.data_pre_processing(self.metrics, new_model)
            self.time_usage["track_io_data"].append(time.time() - begin)

            _score, compute_time = evaluator_register[self.metrics].evaluate_wrapper(
                arch=new_model,
                device=self.device,
                space_name=self.search_space_ins.name,
                batch_data=mini_batch,
                batch_labels=self.mini_batch_targets)

            self.time_usage["track_compute"].append(compute_time)

            if self.if_cuda_avaiable():
                begin = time.time()
                _score = _score.item()
                torch.cuda.synchronize()
                self.time_usage["track_io_res_load"].append(time.time() - begin)
                # gc
                begin = time.time()
                del new_model
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self.time_usage["track_io_model_release"].append(time.time() - begin)

            else:
                _score = _score.item()
                self.time_usage["track_io_res_load"].append(0)

            model_score = {self.metrics: abs(_score)}

            # import torch
            # import gc
            # for obj in gc.get_objects():
            #     try:
            #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #             print(type(obj), obj.size())
            #     except:
            #         pass

        return model_score

    def force_gc(self):
        self.explored_model.clear()
        gc.collect()
        print(" force gc ... ")

    def _p1_evaluate_simu_jacflow(self, model_acquire: ModelAcquireData) -> dict:
        """
        This involves get rank, and get jacflow
        """
        if self.score_getter is None:
            self.score_getter = SimulateScore(space_name=self.search_space_ins.name,
                                              dataset_name=self.dataset_name)

        model_score = self.score_getter.query_tfmem_rank_score(arch_id=model_acquire.model_id)

        return model_score

    def _p1_evaluate_simu(self, model_acquire: ModelAcquireData) -> dict:
        """
        This involves simulate get alls core,
        """
        if self.score_getter is None:
            self.score_getter = SimulateScore(space_name=self.search_space_ins.name,
                                              dataset_name=self.dataset_name)

        score = self.score_getter.query_all_tfmem_score(arch_id=model_acquire.model_id)
        model_score = {self.metrics: abs(score[self.metrics])}
        return model_score

    def data_pre_processing(self, metrics: str, new_model: nn.Module):
        """
        To measure the io/compute time more acccuretely, we pick the data pre_processing here.
        """
        if self.processed_mini_batch is not None:
            return self.processed_mini_batch

        # for those two metrics, we use all one embedding for efficiency (as in their paper)
        if metrics in [CommonVars.ExpressFlow, CommonVars.PRUNE_SYNFLOW]:
            if isinstance(self.mini_batch, torch.Tensor):
                feature_dim = list(self.mini_batch[0, :].shape)
                # add one dimension to feature dim, [1] + [3, 32, 32] = [1, 3, 32, 32]
                mini_batch = torch.ones([1] + feature_dim).float().to(self.device)
            else:
                # this is for the tabular data,
                mini_batch = new_model.generate_all_ones_embedding().float().to(self.device)
        else:
            mini_batch = self.mini_batch

        # wait for moving data to GPU
        if self.if_cuda_avaiable():
            torch.cuda.synchronize()
        self.processed_mini_batch = mini_batch
        return self.processed_mini_batch
