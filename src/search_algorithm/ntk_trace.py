

import numpy as np
from search_algorithm.core.evaluator import Evaluator
from search_algorithm.utils.autograd_hacks import *
from search_space import Architecture


class NTKTraceEvaluator(Evaluator):

    def __init__(self):
        super().__init__()

    def evaluate(self, arch: Architecture, pre_defined, batch_data: torch.tensor, batch_labels: torch.tensor) -> float:
        """
        This is implementation of paper
        "NASI: Label- and Data-agnostic Neural Architecture Search at Initialization"
        The score takes 5 steps:
            1. run forward on a mini-batch
            2. output = sum( [ yi for i in mini-batch N ] ) and then run backward
            3. explicitly calculate gradient of f on each example, df/dxi,
                grads = [ df/ dxi for xi in [1, ..., N] ], dim = [N, number of parameters]
            4. calculate NTK = grads * grads_t
            5. calculate M_trace = traceNorm(NTK), score = np.sqrt(trace_norm / batch_size)
        """
        arch.zero_grad()
        batch_size = batch_data.shape[0]
        add_hooks(arch)

        # 1. forward on mini-batch
        outputs = arch.forward(batch_data)

        # 2. run backward
        # todo: why sum all sample's output ?
        output_f = sum(outputs[torch.arange(batch_size), batch_labels])
        output_f.backward()

        # 3. calculate gradient for each sample in the batch
        # grads = ∇0 f(X), it is N*P , N is number of sample, P is number of parameters,
        compute_grad1(arch, loss_type='sum')

        grads = [param.grad1.flatten(start_dim=1) for param in arch.parameters() if hasattr(param, 'grad1')]
        grads = torch.cat(grads, axis=1)

        # 4. ntk = ∇0 f(X) * Transpose( ∇0 f(X) ) [ batch_size * batch_size ]
        ntk = torch.matmul(grads, grads.t())

        # 5. calculate M_trace = sqrt ( |ntk|_tr * 1/m )

        # For a Hermitian matrix, like a density matrix,
        # the absolute value of the eigenvalues are exactly the singular values,
        # so the trace norm is the sum of the absolute value of the eigenvalues of the density matrix.
        eigenvalues, _ = torch.symeig(ntk)  # ascending
        trace_norm = eigenvalues.cpu().numpy().sum()
        return np.sqrt(trace_norm / batch_size)


