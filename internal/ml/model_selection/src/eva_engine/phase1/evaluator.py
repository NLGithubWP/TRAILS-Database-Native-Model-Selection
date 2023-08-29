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
import psycopg2
from typing import Any, List, Dict, Tuple


class P1Evaluator:

    def __init__(self, device: str, num_label: int, dataset_name: str,
                 search_space_ins: SpaceWrapper,
                 train_loader: DataLoader, is_simulate: bool, metrics: str = CommonVars.ExpressFlow,
                 enable_cache: bool = False, db_config: Dict = None):
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
        :param db_config: how to connect to databaes
        """
        self.metrics = metrics
        self.is_simulate = is_simulate
        # used only is_simulate = True
        self.score_getter = None

        # dataset settings
        self.dataset_name = dataset_name
        self.train_loader = train_loader
        self.num_labels = num_label

        self.search_space_ins = search_space_ins

        self.device = device

        # this is to do the expeirment
        self.enable_cache = enable_cache
        self.model_cache = None

        # performance records
        self.time_usage = {
            "latency": 0.0,
            "io_latency": 0.0,
            "compute_latency": 0.0,
            "track_compute": [],  # compute time
            "track_io_model_init": [],  # init model weight
            "track_io_model_load": [],  # load model into GPU/CPU
            "track_io_res_load": [],  # load result into GPU/CPU
            "track_io_model_release_each_50": [],  # context switch
            "track_io_model_release": [],  # context switch
            "track_io_data": [],  # context switch
        }

        self.db_config = db_config

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
        mini_batch, mini_batch_targets, _ = self.retrievel_data()
        model_acquire = ModelAcquireData.deserialize(data_str)
        model_encoding = model_acquire.model_encoding
        new_model = self.search_space_ins.new_arch_scratch_with_default_setting(model_encoding, bn=True)
        if self.search_space_ins.name == Config.MLPSP:
            new_model.init_embedding(requires_grad=True)
        new_model = new_model.to(self.device)
        flops, params = profile(new_model, inputs=(mini_batch,))
        print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
        print('Params = ' + str(params / 1000 ** 2) + 'M')

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

    def _p1_evaluate_online(self, model_acquire: ModelAcquireData) -> dict:

        model_encoding = model_acquire.model_encoding

        # 1. Get a batch of data
        mini_batch, mini_batch_targets, data_load_time_usage = self.retrievel_data()

        # 2. Score all tfmem
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

                mini_batch = self.data_pre_processing(mini_batch, self.metrics, new_model)

                _score, _ = score_evaluator.evaluate_wrapper(
                    arch=new_model,
                    device=self.device,
                    space_name=self.search_space_ins.name,
                    batch_data=mini_batch,
                    batch_labels=mini_batch_targets)

                _score = _score.item()
                model_score[alg] = abs(_score)

                # clear the cache
                if "cuda" in self.device:
                    torch.cuda.empty_cache()

        # 2. score using only one metrics
        else:
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
            mini_batch = self.data_pre_processing(mini_batch, self.metrics, new_model)
            self.time_usage["track_io_data"].append(data_load_time_usage + (time.time() - begin))

            _score, compute_time = evaluator_register[self.metrics].evaluate_wrapper(
                arch=new_model,
                device=self.device,
                space_name=self.search_space_ins.name,
                batch_data=mini_batch,
                batch_labels=mini_batch_targets)

            self.time_usage["track_compute"].append(compute_time)

            if self.if_cuda_avaiable():
                begin = time.time()
                _score = _score.item()
                torch.cuda.synchronize()
                self.time_usage["track_io_res_load"].append(time.time() - begin)

            else:
                _score = _score.item()
                self.time_usage["track_io_res_load"].append(0)

            model_score = {self.metrics: abs(_score)}
            del new_model
        return model_score

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

    def retrievel_data(self):
        if not self.is_simulate:
            if self.dataset_name in [Config.c10, Config.c100, Config.imgNet]:
                if self.train_loader is None:
                    raise f"self.train_loader is None for {self.dataset_name}"
                # for img data
                begin =  time.time()
                mini_batch, mini_batch_targets = dataset.get_mini_batch(
                    dataloader=self.train_loader,
                    sample_alg="random",
                    batch_size=32,
                    num_classes=self.num_labels)
                mini_batch.to(self.device)
                mini_batch_targets.to(self.device)
                # wait for moving data to GPU
                if self.if_cuda_avaiable():
                    torch.cuda.synchronize()
                time_usage = time.time() - begin
                return mini_batch, mini_batch_targets, time_usage
            elif self.dataset_name in [Config.Criteo, Config.Frappe, Config.UCIDataset]:
                if self.train_loader is None:
                    batch, time_usage = self._retrievel_from_db()
                    begin = time.time()
                    id_tensor = torch.FloatTensor(batch['id']).to(self.device)
                    value_tensor = torch.FloatTensor(batch['value']).to(self.device)
                    y_tensor = torch.LongTensor(batch['y']).to(self.device)
                    data_tensor = {'id': id_tensor, 'value': value_tensor, 'y': y_tensor}
                    # wait for moving data to GPU
                    if self.if_cuda_avaiable():
                        torch.cuda.synchronize()
                    time_usage += time.time() - begin
                    return data_tensor, y_tensor, time_usage
                else:
                    # this is structure data
                    begin = time.time()
                    batch = iter(self.train_loader).__next__()
                    target = batch['y'].type(torch.LongTensor).to(self.device)
                    batch['id'] = batch['id'].to(self.device)
                    batch['value'] = batch['value'].to(self.device)

                    # wait for moving data to GPU
                    if self.if_cuda_avaiable():
                        torch.cuda.synchronize()
                    time_usage = time.time() - begin
                    return batch, target, time_usage
            else:
                raise NotImplementedError

    def _retrievel_from_db(self):

        def decode_libsvm(columns):
            map_func = lambda pair: (int(pair[0]), float(pair[1]))
            # 0 is id, 1 is label
            id, value = zip(*map(lambda col: map_func(col.split(':')), columns[2:]))
            sample = {'id': list(id),
                      'value': list(value),
                      'y': float(columns[1])}
            return sample

        def pre_processing(mini_batch_data: List[Tuple]):
            """
            mini_batch_data: [('0', '0', '123:123', '123:123', '123:123',)
            """
            sample_lines = len(mini_batch_data)
            feat_id = []
            feat_value = []
            y = []

            for i in range(sample_lines):
                row_value = mini_batch_data[i]
                sample = decode_libsvm(list(row_value))
                feat_id.append(sample['id'])
                feat_value.append(sample['value'])
                y.append(sample['y'])
            return {'id': feat_id, 'value': feat_value, 'y': y}

        begin_time = time.time()
        with psycopg2.connect(database=self.db_config["db_name"],
                              user=self.db_config["db_user"],
                              host=self.db_config["db_host"],
                              port=self.db_config["db_port"]) as conn:

            # fetch and preprocess data from PostgreSQL
            cur = conn.cursor()

            # columns_str = ', '.join(columns)
            # Select rows greater than last_id
            cur.execute(f"SELECT * FROM {self.dataset_name}_train "
                        f"ORDER BY RANDOM() LIMIT 32")

            rows = cur.fetchall()
            batch = pre_processing(rows)
            # block until a free slot is available
        time_usage = time.time() - begin_time
        return batch, time_usage

    def data_pre_processing(self, mini_batch, metrics: str, new_model: nn.Module):
        """
        To measure the io/compute time more acccuretely, we pick the data pre_processing here.
        """

        # for those two metrics, we use all one embedding for efficiency (as in their paper)
        if metrics in [CommonVars.ExpressFlow, CommonVars.PRUNE_SYNFLOW]:
            if isinstance(mini_batch, torch.Tensor):
                feature_dim = list(mini_batch[0, :].shape)
                # add one dimension to feature dim, [1] + [3, 32, 32] = [1, 3, 32, 32]
                mini_batch = torch.ones([1] + feature_dim).float().to(self.device)
            else:
                # this is for the tabular data,
                mini_batch = new_model.generate_all_ones_embedding().float().to(self.device)
        else:
            # for others, skip preprocessing
            pass

        # wait for moving data to GPU
        if self.if_cuda_avaiable():
            torch.cuda.synchronize()
        return mini_batch
