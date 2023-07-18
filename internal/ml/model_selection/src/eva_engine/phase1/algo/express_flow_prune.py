


from src.eva_engine.phase1.algo.alg_base import Evaluator
from src.eva_engine.phase1.utils.autograd_hacks import *
from torch import nn


class ExpressFlowEvaluator(Evaluator):

    def __init__(self):
        super().__init__()

    def evaluate(self, arch: nn.Module, device, batch_data: object, batch_labels: torch.Tensor) -> float:

        # # 1. Convert params to their abs. Record sign for converting it back.
        @torch.no_grad()
        def linearize(arch):
            signs = {}
            for name, param in arch.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        # convert to orig values with sign
        @torch.no_grad()
        def nonlinearize(arch, signs):
            for name, param in arch.state_dict().items():
                if 'weight_mask' not in name:
                    param.mul_(signs[name])

        def hook_fn(module, input, output):
            z = output  # activation

            # this method register_hook in PyTorch is used to register a backward hook on a tensor
            def grad_hook(grad):
                dz = grad  # gradient
                V = z * abs(dz)  # product
                Vs.append(V)
            z.register_hook(grad_hook)

        signs = linearize(arch)

        # Create a list to store the results for each neuron
        Vs = []
        hooks = []
        for module in arch.modules():
            if isinstance(module, nn.ReLU):
                hooks.append(module.register_forward_hook(hook_fn))

        if isinstance(batch_data, torch.Tensor):
            feature_dim = list(batch_data[0, :].shape)
            # add one dimension to feature dim, [1] + [3, 32, 32] = [1, 3, 32, 32]
            batch_data = torch.ones([1] + feature_dim).double().to(device)
            out = arch.forward(batch_data)
        else:
            # this is for the embedding data,
            batch_data = arch.generate_all_ones_embedding().to(device).float()
            out = arch.forward_wo_embedding(batch_data)

        # directly sum
        # out = arch(batch_data)
        torch.sum(out).backward()

        total_sum = 0.0*Vs[0].flatten().sum()*list(Vs[0].shape)[1]/10 \
                    + 0.0*Vs[1].flatten().sum()*list(Vs[1].shape)[1]/10 \
                    + Vs[2].flatten().sum()*list(Vs[2].shape)[1]/10 \
                    + Vs[3].flatten().sum()*list(Vs[3].shape)[1]/10

        total_sum = total_sum.item()

        # Remove the hooks
        for hook in hooks:
            hook.remove()

        return total_sum
