

from torch import autograd

from search_algorithm.core.evaluator import Evaluator
from search_algorithm.utils.autograd_hacks import *
from search_algorithm.utils.p_utils import get_layer_metric_array


class VoteEvaluator(Evaluator):

    def __init__(self):
        super().__init__()

    def evaluate(self, arch: nn.Module, device, batch_data: torch.tensor, batch_labels: torch.tensor) -> float:
        """
        This is simply sum over all weigth's norm to calculate models performance
        :param arch:
        :param device: CPU or GPU
        :param batch_data:
        :param batch_labels:
        :return:
        """

        pass



