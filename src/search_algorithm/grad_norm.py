
from search_algorithm.core.evaluator import Evaluator
from search_algorithm.utils.autograd_hacks import *
from search_algorithm.utils.p_utils import get_layer_metric_array
from logger import logger
from torch import nn


class GradNormEvaluator(Evaluator):

    def __init__(self):
        super().__init__()

    def evaluate(self, arch: nn.Module, pre_defined, batch_data: torch.tensor, batch_labels: torch.tensor) -> float:
        """
        This is implementation of paper
        "Keep the Gradients Flowing: Using Gradient Flow to Study Sparse Network Optimization"
        The score takes 5 steps:
            1. Run a forward & backward pass to calculate gradient of loss on weight, grad_w = d_loss/d_w
            2. Then calculate norm for each gradient, grad.norm(p), default p = 2
            3. Sum up all weights' grad norm and get the overall architecture score.
        """

        split_data = 1
        loss_fn = F.cross_entropy

        # arch.zero_grad()
        N = batch_data.shape[0]

        grad_norm_arr = []
        for sp in range(split_data):
            st = sp * N // split_data
            en = (sp + 1) * N // split_data

            # 1. forward on mini-batch
            # logger.info("min-batch is in cuda2 = " + str(batch_data.is_cuda))
            outputs = arch.forward(batch_data[st:en])
            loss = loss_fn(outputs, batch_labels[st:en])
            loss.backward()

            # 2. lambda function as callback to calculate norm of gradient
            part_grad = get_layer_metric_array(
                arch,
                lambda l:
                    l.weight.grad.norm() if l.weight.grad is not None else torch.zeros_like(l.weight), mode='param')

            grad_norm_arr.extend(part_grad)

        # 3. Sum over all parameter's results to get the final score.
        score = 0.
        for i in range(len(grad_norm_arr)):
            score += grad_norm_arr[i].detach().cpu().numpy().sum()
        return score
