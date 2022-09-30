
import numpy as np
import torch
from torch import nn
from eva_engine.phase1.evaluator_base import Evaluator


class NWTEvaluator(Evaluator):

    def __init__(self):
        super().__init__()

    def evaluate(self, arch: nn.Module, device, batch_data: torch.tensor, batch_labels: torch.tensor) -> float:
        """
        This is implementation of paper "Neural Architecture Search without Training"
        The score takes 5 steps:
            1. for ech example, get the binary vector for each relu layer, where 1 means x > 0, 0 otherwise,
            2. calculate K = [Na - hamming_distance (ci, cj) for each ci, cj]
        """

        # add new attribute K
        arch.K = np.zeros((batch_data.shape[0], batch_data.shape[0]))

        def counting_forward_hook(module, inp, out):
            """
            :param module: module
            :param inp: input feature for this module
            :param out: out feature for this module
            :return: score
            """
            # if "visited_backwards" not in module.__dict__:
            #     return
            # if not module.visited_backwards:
            #     return

            # get the tensor = [batch_size, channel, size, size]
            if isinstance(inp, tuple):
                inp = inp[0]
            # the size -1 is inferred from other dimensions, eg,. [ batch_size, 16*32*32 ]
            inp = inp.view(inp.size(0), -1)
            # convert input to a binary code vector wth Relu, indicate whether the unit is active
            x = (inp > 0).float()
            # after summing up K+K2 over all modules,
            # at each position of index (i, j), the value = ( NA - dist(ci, cj) )
            K = x @ x.t()
            K2 = (1. - x) @ (1. - x.t())
            # sum up all relu module's result
            arch.K = arch.K + K.cpu().numpy() + K2.cpu().numpy()

        # def counting_backward_hook(module, inp, out):
        #     module.visited_backwards = True

        # for each relu, check how many active or inactive value in output
        for name, module in arch.named_modules():
            if 'ReLU' in str(type(module)):
                module.register_forward_hook(counting_forward_hook)
                # module.register_backward_hook(counting_backward_hook)

        batch_data2 = torch.clone(batch_data)

        # self.get_batch_jacobian(arch, batch_data, batch_labels)

        # run a forward computation
        arch(batch_data2.to(device))

        # calculate s = log|K|
        s, ld = np.linalg.slogdet(arch.K)
        return ld

    def get_batch_jacobian(self, arch, x, target):
        arch.zero_grad()
        x.requires_grad_(True)
        y = arch(x)
        y.backward(torch.ones_like(y))
        jacob = x.grad.detach()
        return jacob, target.detach(), y.detach()
