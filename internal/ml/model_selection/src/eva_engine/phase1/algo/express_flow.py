from src.eva_engine.phase1.algo.alg_base import Evaluator
from src.eva_engine.phase1.utils.autograd_hacks import *
from torch import nn
from src.common.constant import Config


class ExpressFlowEvaluator(Evaluator):

    def __init__(self):
        super().__init__()

    def evaluate(self, arch: nn.Module, device, batch_data: object, batch_labels: torch.Tensor,
                 space_name: str) -> float:

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

        arch.double()
        if space_name == Config.MLPSP:
            out = arch.forward_wo_embedding(batch_data.double())
        else:
            out = arch.forward(batch_data.double())

        # directly sum
        torch.sum(out).backward()

        trajectory_lengths = self.calculate_trajectory_length(arch, batch_data)
        total_sum = self.weighted_score(trajectory_lengths, Vs)
        # total_sum = self.compute_score_early_halflayers(out, Vs)

        # Remove the hooks
        for hook in hooks:
            hook.remove()

        return total_sum

    def calculate_trajectory_length(self, arch, batch_data):
        epsilon = 1e-5
        delta_x = torch.randn_like(batch_data) * epsilon

        originals, perturbations = [], []

        x, x_perturbed = batch_data, batch_data + delta_x

        for module in arch.mlp.mlp:
            x = module(x.double())
            x_perturbed = module(x_perturbed.double())

            if isinstance(module, nn.ReLU):
                originals.append(x)
                perturbations.append(x_perturbed)

        trajectory_lengths = [(x_perturbed - x).norm() / epsilon for x, x_perturbed in zip(originals, perturbations)]

        return trajectory_lengths

    def compute_score_early_halflayers(self, out, Vs):
        # Vs is a list of tensors, where each tensor corresponds to the product
        # V=z×∣dz∣ (where z is the activation and dz is the gradient) for every ReLU layer in your model.
        # Each tensor in Vs has the shape (batch_size, number_of_neurons)
        # 1. aggregates the importance of all neurons in that specific ReLU module.
        # 2. only use the first half layers.

        # Determine the half point
        half_point = len(Vs) // 2

        # Sum over the second half of the modules,
        # Vs[i].shape[1]: number of neuron in the layer i
        total_sum = sum(V.flatten().sum() * V.shape[1] for V in Vs[half_point:]) / 10
        total_sum = total_sum.item()
        return total_sum

    def weighted_score(self, trajectory_lengths, Vs):
        trajectory_lengths.reverse()
        # Modify trajectory_lengths to ensure that deeper layers have smaller weights
        # For example, by taking the inverse of each computed trajectory length.
        inverse_trajectory_lengths = [1.0 / (length + 1e-6) for length in trajectory_lengths]

        # Normalize trajectory lengths if needed (this ensures the weights aren't too large)
        normalized_lengths = [length / sum(inverse_trajectory_lengths) for length in inverse_trajectory_lengths]

        # Use the normalized trajectory lengths as weights for your total_sum
        total_sum = sum(
            normalized_length * V.flatten().sum() * V.shape[1] for normalized_length, V in zip(normalized_lengths, Vs))
        total_sum = total_sum.item()

        return total_sum



