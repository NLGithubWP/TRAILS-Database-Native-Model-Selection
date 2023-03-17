import random
import time

from controller import RegularizedEASampler
from controller.core.sample import Sampler
from third_party.models import CellStructure


class ModelScore:
    def __init__(self, model_id, score):
        self.model_id = model_id
        self.score = score

    def __repr__(self):
        return "m_{}_s_{}".format(self.model_id, self.score)


# for binary insert
def binary_insert_get_rank(rank_list: list, new_item: ModelScore) -> int:
    """
    Insert the new_item to rank_list, then get the rank of it.
    :param rank_list:
    :param new_item:
    :return:
    """
    index = search_position(rank_list, new_item)
    # search the position to insert into
    rank_list.insert(index, new_item)
    return index


# O(logN) search the position to insert into
def search_position(rank_list_m: list, new_item: ModelScore):
    if len(rank_list_m) == 0:
        return 0
    left = 0
    right = len(rank_list_m) - 1
    while left + 1 < right:
        mid = int((left + right) / 2)
        if rank_list_m[mid].score <= new_item.score:
            left = mid
        else:
            right = mid

    # consider the time.
    if rank_list_m[right].score <= new_item.score:
        return right + 1
    elif rank_list_m[left].score <= new_item.score:
        return left + 1
    else:
        return left


class SampleController(object):
    """
    Controller control the sample-score flow in the 1st phase.
    It records the results in the history.
    """

    def __init__(self, search_strategy: Sampler):
        # Current ea is better than others.
        self.search_strategy = search_strategy

        # this is pair of (model, score )
        self.model_rank = {}
        # model vite_score dict
        self.vote_model_id = []

        self.history = []

    def sample_next_arch(self, max_nodes: int) -> (str, CellStructure):
        """
        Return a generator
        :param max_nodes:
        :return:
        """
        return self.search_strategy.sample_next_arch(max_nodes)

    def fit_sampler(self, arch_id, alg_score, use_prue_score: bool = False):
        """

        :param arch_id:
        :param alg_score:
        :param use_prue_score: if simply sum multiple scores (good performing),
                             or sum over their rank (worse performing)
        :return:
        """
        if use_prue_score:
            score = self.use_pure_score_as_final_res(arch_id, alg_score)
        else:
            score = self._add_model_to_rank(arch_id, alg_score)
        self.search_strategy.fit_sampler(score)

    def _add_model_to_rank(self, model_id: str, alg_score: dict):
        # todo: bug: only all scores' under all arg is greater than previous one, then treat it as greater.
        for alg in alg_score:
            if alg not in self.model_rank:
                self.model_rank[alg] = []

        # add model and score to local list
        for alg, score in alg_score.items():
            binary_insert_get_rank(self.model_rank[alg], ModelScore(model_id, score))

        new_rank_score = self.re_rank_model_id(model_id, alg_score)
        return new_rank_score

    def use_pure_score_as_final_res(self, model_id: str, alg_score: dict):
        final_score = 0
        for alg in alg_score:
            final_score += float(alg_score[alg])
        index = binary_insert_get_rank(self.history, ModelScore(model_id, final_score))
        self.vote_model_id.insert(index, model_id)
        return final_score

    def re_rank_model_id(self, model_id: str, alg_score: dict):
        # todo: re-rank everything, to make it self.vote_model_id more accurate.
        model_new_rank_score = {}
        for alg, score in alg_score.items():
            for rank_index in range(len(self.model_rank[alg])):
                current_explored_models = len(self.model_rank[alg])
                ms_ins = self.model_rank[alg][rank_index]
                # rank = index + 1, since index can be 0
                if ms_ins.model_id in model_new_rank_score:
                    model_new_rank_score[ms_ins.model_id] += rank_index + 1
                else:
                    model_new_rank_score[ms_ins.model_id] = rank_index + 1

        for ele in model_new_rank_score.keys():
            model_new_rank_score[ele] = model_new_rank_score[ele] / current_explored_models

        self.vote_model_id = [k for k, v in sorted(model_new_rank_score.items(), key=lambda item: item[1])]
        new_rank_score = model_new_rank_score[model_id]
        return new_rank_score

    def get_current_top_k_models(self, k=10):
        """
        The model is already scored by: low -> high
        :param k:
        :return:
        """
        # return [ele.model_id for ele in self.vote_score[-k:]]
        return self.vote_model_id[-k:]


if __name__ == "__main__":

    rank_list = []
    begin = time.time()
    score_list = [1, 2, 3, 1, 2]
    for i in range(5):
        ms = ModelScore(i, score_list[i])
        binary_insert_get_rank(rank_list, ms)
    print(rank_list)
    print(time.time() - begin)

    rank_list = []
    begin = time.time()
    score_list = [1, 1, 1, 1, 1]
    for i in range(5):
        ms = ModelScore(i, score_list[i])
        binary_insert_get_rank(rank_list, ms)
    print(rank_list)
    print(time.time() - begin)


