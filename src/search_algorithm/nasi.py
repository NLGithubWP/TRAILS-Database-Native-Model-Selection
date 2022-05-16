
import torch
from torch.autograd import grad
import torch.nn.functional as F

class NASIEvaluator:

    def __init__(self):
        pass

    def score(self, arch, pre_defined, train_data) -> float:
        """
        Score each architecture with predefined architecture and data
        :param arch: architecture to be scored
        :param pre_defined: pre-defined evaluation args
        :param train_data: train data loader
        :return: score
        """

        # get all pre-defined variables.
        mu = pre_defined.reg_weight
        sparsity = pre_defined.sparsity
        v = pre_defined.gap
        is_adaptive = pre_defined.adaptive
        is_rand_data = pre_defined.rand_data
        is_rand_label = pre_defined.rand_label

        sum_gap = v

        # run in train mode
        arch.train()
        # get all weights from architecture
        model_params = [p for n, p in arch.named_parameters() if p is not None]

        # Sample data D_t ~ D of size b, only use one batch to evaluate.
        for step, (input, targets) in enumerate(train_data):
            print("\n step =", step)

            arch.reset_zero_grads()
            # process training data.
            if torch.cuda.is_available():
                input = input.cuda()
            if is_rand_data:
                input.normal_()
            if torch.cuda.is_available():
                targets = targets.cuda()
            if is_rand_label:
                idx = torch.randperm(targets.numel())
                targets = targets[idx]

            if torch.cuda.is_available():
                print("Model is loaded in GPU")
                arch.cuda()

            # this will run the forward,
            # Sample gt ∼ p(g) = Gumbel(0, 1) and determine
            logits = arch(input)

            task_loss = F.cross_entropy(logits, targets)
            # Evaluate gradient Gt = ∇Theta L(x) with data Dt
            grads = grad(task_loss, model_params, create_graph=True, allow_unused=True)

            if is_adaptive:
                gap = sum_gap / (step + 1)
            else:
                gap = v

            score, avg_eigen = self.trace_loss(grads, (1 - sparsity) * gap, mu)
            return score

    def trace_loss(self, grads, gap, reg_weight) -> (float, float):
        """
        Evaluate score of a given architecture with only one batch
        :param grads: y grads on weights
        :param gap: v in paper, v = r * n* 1/learning rate
        :param reg_weight: mu in paper.
        :return: score and NTK
        """

        grad_list = []

        for g in grads:
            try:
                ele = (g**2).sum()
                grad_list.append(ele)
            except:
                print("g is none")

        avg_eigen = sum(grad_list)

        # avg_eigen = sum([ (g ** 2).sum() for g in grads])

        score = avg_eigen - reg_weight * F.relu(avg_eigen - gap)
        return score, avg_eigen
