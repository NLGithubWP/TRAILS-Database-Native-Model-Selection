
import time
from common.constant import Config
from common.structure import ModelAcquireData, ModelEvaData
from controller.controler import Controller
from query_api.db_ops import fetch_from_db
from query_api.gt_api import Gt201, Gt101
from search_space import NasBench101Space


def read_from_queue():
    return "{}"


def send_to_queue(data_str):
    pass


def run_phase1_simulate(space_name, dataset, run_id, N, K) -> (list, float):
    arch_id, candidates, current_time = fetch_from_db(space_name, dataset, run_id, N)
    return candidates[-K:], current_time


class RunPhase1:
    def __init__(self, args, K: int, N: int, used_search_space):

        if args.search_space == Config.NB201:
            self.gt_api = Gt201()
        elif args.search_space == Config.NB101:
            self.gt_api = Gt101()

        self.used_search_space = used_search_space
        self.sampler = Controller(self.used_search_space, args)
        self.arch_generator = self.sampler.sample_next_arch(args.arch_size)

        # return K models
        self.K = K
        # explore N models
        self.N = N

    def run_phase1(self) -> list:
        explored_n = 0
        while True:
            request_json = read_from_queue()
            if request_json is None:
                time.sleep(0.01)
                continue
            # once reading score from queue, update and return
            if explored_n <= self.N:
                # communication, receive
                model_eva = ModelEvaData.deserialize(request_json)

                # fit sampler, None means first time acquire model
                self.sampler.fit_sampler(model_eva.model_id, model_eva.model_score)
                # generate new model
                arch_id, model_struc = self.arch_generator.__next__()
                if self.used_search_space.name == Config.NB101:
                    model_encoding = NasBench101Space.serialize_model_encoding(
                        model_struc.original_matrix.tolist(),
                        model_struc.original_ops)
                elif self.used_search_space.name == Config.NB201:
                    model_encoding = self.used_search_space.archid_to_hash(arch_id)
                else:
                    model_encoding, test_accuracy = None, None

                explored_n += 1
                model_acquire_data = \
                    ModelAcquireData(model_id=str(arch_id), model_encoding=model_encoding, is_last=False)
                data_str = model_acquire_data.serialize_model()
                # communication, send
                send_to_queue(data_str)
            else:
                model_acquire_data = \
                    ModelAcquireData(model_id=str(arch_id), model_encoding=model_encoding, is_last=True)
                data_str = model_acquire_data.serialize_model()
                # communication, send
                send_to_queue(data_str)
                break

        return self.sampler.get_current_top_k_models(self.K)


