from torch import autograd

from search_algorithm.core.evaluator import Evaluator
from search_algorithm.utils.autograd_hacks import *
from search_algorithm.utils.p_utils import get_layer_metric_array


class GraspEvaluator(Evaluator):

    def __init__(self):
        super().__init__()

    def evaluate(self, arch: nn.Module, pre_defined, batch_data: torch.tensor, batch_labels: torch.tensor) -> float:
        """
        This is implementation of paper
        "PICKING WINNING TICKETS BEFORE TRAINING BY PRESERVING GRADIENT FLOW"
        The score takes 5 steps:
            1. Run a forward & backward pass to calculate gradient of loss on weight, grad_w = d_loss/d_w
            2. Run  forward & backward again, this is to calculate  H*g, it use hessian vector product to calculate it.
                eg, (d_loss / d_w)^2 * g = d_( (d_loss / d_w) * g) / d_w
                the code first calculate z = (d_loss / d_w) * g, and then calculate gradient on z with z.gradient()
            3. Then calculate Hg.0
            4. Sum up all weights' score and get the overall architecture score.
        """

        # alg cfgs
        T = 1
        num_iters = 1
        split_data = 1
        loss_fn = F.cross_entropy

        # get all applicable weights
        weights = []
        for layer in arch.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                weights.append(layer.weight)
                # TODO isn't this already true?
                layer.weight.requires_grad_(True)

        # NOTE original code had some input/target splitting into 2
        # I am guessing this was because of GPU mem limit
        # arch.zero_grad()
        N = batch_data.shape[0]
        for sp in range(split_data):
            st = sp * N // split_data
            en = (sp + 1) * N // split_data

            # forward/grad pass #1
            grad_w = None
            for _ in range(num_iters):
                # TODO get new data, otherwise num_iters is useless!
                outputs = arch.forward(batch_data[st:en]) / T
                loss = loss_fn(outputs, batch_labels[st:en])
                grad_w_p = autograd.grad(loss, weights, allow_unused=True)
                if grad_w is None:
                    grad_w = list(grad_w_p)
                else:
                    for idx in range(len(grad_w)):
                        grad_w[idx] += grad_w_p[idx]

        for sp in range(split_data):
            st = sp * N // split_data
            en = (sp + 1) * N // split_data

            # forward/grad pass #2
            outputs = arch.forward(batch_data[st:en]) / T
            loss = loss_fn(outputs, batch_labels[st:en])
            grad_f = autograd.grad(loss, weights, create_graph=True, allow_unused=True)

            # accumulate gradients computed in previous step and call backwards
            z, count = 0, 0
            for layer in arch.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    if grad_w[count] is not None:
                        z += (grad_w[count].data * grad_f[count]).sum()
                    count += 1
            z.backward()

        # compute final sensitivity metric and put in grads
        def grasp(layer):
            if layer.weight.grad is not None:
                return -layer.weight.data * layer.weight.grad  # -theta_q Hg
                # NOTE in the grasp code they take the *bottom* (1-p)% of values
                # but we take the *top* (1-p)%, therefore we remove the -ve sign
                # EDIT accuracy seems to be negatively correlated with this metric, so we add -ve sign here!
            else:
                return torch.zeros_like(layer.weight)

        grads = get_layer_metric_array(arch, grasp, "param")

        # Sum over all parameter's results to get the final score.
        score = 0.
        for i in range(len(grads)):
            score += grads[i].detach().cpu().numpy().sum()
        return score
