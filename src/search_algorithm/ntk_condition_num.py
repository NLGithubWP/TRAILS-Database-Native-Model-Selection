

import numpy as np
from search_algorithm.core.evaluator import Evaluator
from search_algorithm.utils.autograd_hacks import *
from search_space import Architecture


class NTKCondNumEvaluator(Evaluator):

    def __init__(self):
        super().__init__()

    def evaluate(self, arch: Architecture, pre_defined, batch_data: torch.tensor, batch_labels: torch.tensor) -> float:
        """
        This is implementation of paper TE-NAS,
        "NEURAL ARCHITECTURE SEARCH ON IMAGENET IN FOUR GPU HOURS: A THEORETICALLY INSPIRED PERSPECTIVE"
        The implementation only consider K in paper.
        The score takes 5 steps:
            1. run forward on a mini-batch
            2. output = sum( [ yi for i in mini-batch N ] ) and then run backward
            3. explicitly calculate gradient of f on each example, df/dxi,
                grads = [ df/ dxi for xi in [1, ..., N] ], dim = [N, number of parameters]
            4. calculate NTK = grads * grads_t
            5. calculate score = 1/K = eigenvalues_max / eigenvalues_min
        """

        arch.zero_grad()
        batch_size = batch_data.shape[0]
        add_hooks(arch)

        # 1. forward on mini-batch
        outputs = arch.forward(batch_data)

        # 2. run backward
        sum(outputs[torch.arange(batch_size), batch_labels]).backward()

        # 3. calculate gradient for each sample in the batch
        compute_grad1(arch, loss_type='sum')
        grads = [param.grad1.flatten(start_dim=1) for param in arch.parameters() if hasattr(param, 'grad1')]
        grads = torch.cat(grads, axis=1)

        # 4. ntk = ∇0 f(X) * Transpose( ∇0 f(X) ) [ batch_size * batch_size ]
        ntk = torch.matmul(grads, grads.t())

        # 5. sort eigenvalues and then calculate k = lambda_0 / lambda_m
        # since k is negatively correlated with the architecture’s test accuracy. So, it uses k = lambda_m / lambda_0
        eigenvalues, _ = torch.symeig(ntk)  # ascending
        # convert nan and inf into 0 and 10000
        score = np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0).item()
        return score

