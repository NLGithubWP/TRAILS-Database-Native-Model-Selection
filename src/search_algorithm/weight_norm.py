
from search_algorithm.core.evaluator import Evaluator
from search_algorithm.utils.autograd_hacks import *
from search_algorithm.utils.p_utils import get_layer_metric_array
 

class WeightNormEvaluator(Evaluator):

    def __init__(self):
        super().__init__()

    def evaluate(self, arch: nn.Module, pre_defined, batch_data: torch.tensor, batch_labels: torch.tensor) -> float:
        """
        This is simply sum over all weigth's norm to calculate models performance
        :param arch:
        :param pre_defined:
        :param batch_data:
        :param batch_labels:
        :return:
        """
        grad_norm_arr = get_layer_metric_array(arch, lambda l: l.weight.norm(), mode="param")

        # 3. Sum over all parameter's results to get the final score.
        score = 0.
        for i in range(len(grad_norm_arr)):
            score += grad_norm_arr[i].detach().cpu().numpy().sum()
        return score
