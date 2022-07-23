

from search_algorithm.core.evaluator import Evaluator
from search_algorithm.utils.autograd_hacks import *
from search_algorithm.utils.p_utils import get_layer_metric_array
from torch import nn


class GradPlainEvaluator(Evaluator):

    def __init__(self):
        super().__init__()

    def evaluate(self, arch: nn.Module, pre_defined, batch_data: torch.tensor, batch_labels: torch.tensor) -> float:
        """
        The score takes 3 steps:
            1. Run a forward & backward pass to calculate gradient of loss on weight, grad_w = d_loss/d_w
            2. Then calculate gradient, grad
            3. Sum up all weights' grad and get the overall architecture score.
        """

        split_data = 1
        loss_fn = F.cross_entropy

        N = batch_data.shape[0]
        for sp in range(split_data):
            st = sp * N // split_data
            en = (sp + 1) * N // split_data

            outputs = arch.forward(batch_data[st:en])
            loss = loss_fn(outputs, batch_labels[st:en])
            loss.backward()

        # select the gradients that we want to use for search/prune
        def plain(layer):
            if layer.weight.grad is not None:
                return layer.weight.grad * layer.weight
            else:
                return torch.zeros_like(layer.weight)

        grads_abs = get_layer_metric_array(arch, plain, "param")

        # Sum over all parameter's results to get the final score.
        score = 0.
        for i in range(len(grads_abs)):
            score += grads_abs[i].detach().cpu().numpy().sum()
        return score
