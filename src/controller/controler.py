import random
import time

from controller import RegularizedEASampler
from search_space import SpaceWrapper
from third_party.models import CellStructure


class ModelScore:
    def __init__(self, model_id, score):
        self.model_id = model_id
        self.score = score

    def __repr__(self):
        return "{}_{}".format(self.model_id, self.score)


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
        if rank_list_m[mid].score < new_item.score:
            left = mid
        else:
            right = mid
    if rank_list_m[left].score >= new_item.score:
        return left
    elif rank_list_m[right].score >= new_item.score:
        return right
    else:
        return right + 1


class Controller(object):

    def __init__(self, space: SpaceWrapper, args):
        # Current ea is better than others.
        self.search_strategy = RegularizedEASampler(space, args)

        # this is pair of (model, score )
        self.model_rank = {}
        self.vote_history = []

        # model vite_score dict
        self.vote_score = []
        self.vote_model_id = []

    def sample_next_arch(self, max_nodes: int) -> (str, CellStructure):
        """
        Return a generator
        :param max_nodes:
        :return:
        """
        return self.search_strategy.sample_next_arch(max_nodes)

    def fit_sampler(self, arch_id, alg_score):
        score = self._add_model_to_rank(arch_id, alg_score)
        self.search_strategy.fit_sampler(score)

    def _add_model_to_rank(self, model_id: str, alg_score: dict):
        for alg in alg_score:
            if alg not in self.model_rank:
                self.model_rank[alg] = []

        # add model and score to local list
        # rank = index + 1, since index can be 0
        rank_score = 0
        for alg, score in alg_score.items():
            alg_index = binary_insert_get_rank(self.model_rank[alg], ModelScore(model_id, score))
            alg_rank_ = alg_index + 1
            rank_score_ = alg_rank_ / len(self.model_rank[alg])
            rank_score += rank_score_

        vote_index = binary_insert_get_rank(self.vote_history, ModelScore(model_id, rank_score))
        final_vote_score = (vote_index + 1) / len(self.vote_history)

        model_score_index = binary_insert_get_rank(self.vote_score, ModelScore(model_id, final_vote_score))
        self.vote_model_id.insert(model_score_index, model_id)
        return final_vote_score

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
    for i in range(10):
        ms = ModelScore(i, random.randint(0, 10))
        binary_insert_get_rank(rank_list, ms)
    print(time.time() - begin)


