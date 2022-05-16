import torch
from torch.autograd import grad
import logging
import torch.nn.functional as F
from torchviz import make_dot

from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/fashion_mnist_experiment_1')

class NASISearch:

    def __init__(self):
        pass

    def search(self, model, args, train_queue):
        """
        Search an architecture from an give model.
        :param model:
        :param args:
        :param train_queue:
        :return:
        """
        # score for each alpha, store α∗ where each element corresponding to final learned value for each alpha.
        scores = [0 for _ in model._arch_parameters]
        avg_norm = 0
        max_norm = 0
        # momentum = 0.9
        momentum = 0

        sum_gap = args.gap

        model.train()
        # get all weights from architecture
        model_params = [p for n, p in model.named_parameters() if p is not None]

        # Sample data D_t ~ D of size b
        step = 0
        for step, (input, targets) in enumerate(train_queue):
            print("\n step =", step)
            model.reset_zero_grads()

            # process training data.
            if torch.cuda.is_available():
                input = input.cuda()
            if args.rand_data:
                input.normal_()
            if torch.cuda.is_available():
                targets = targets.cuda()
            if args.rand_label:
                idx = torch.randperm(targets.numel())
                targets = targets[idx]

            # this will run the forward,
            # Sample gt ∼ p(g) = Gumbel(0, 1) and determine
            logits = model(input)

            make_dot(logits, params=dict(model.named_parameters()))

            task_loss = F.cross_entropy(logits, targets)
            # Evaluate gradient Gt = ∇Theta L(x) with data Dt
            grads = grad(task_loss, model_params, create_graph=True)

            if args.adaptive:
                gap = sum_gap / (step + 1)
            else:
                gap = args.gap

            loss, avg_eigen = self.trace_loss(grads, (1 - args.sparsity) * gap, args.reg_weight)
            print("Score is ", loss)
            # calculate gradient on alpha
            # Evaluate gradient Gt = ∇α0 R(At) with data Dt
            loss.backward()

            # for adaptive only
            sum_gap += avg_eigen.item()

            # Evaluate gradient Gt = ∇α0 R(At) with data Dt
            grad_norm = model.arch_param_grad_norm()
            if avg_norm > 0:
                avg_norm = (1 - momentum) * grad_norm + momentum * avg_norm
            else:
                avg_norm = grad_norm
                max_norm = grad_norm

            # calculate max(∥G1∥2,...,∥Gt∥2)
            torch.max(avg_norm, max_norm, out=max_norm)

            # calculate Gt/max(∥G1∥2,...,∥Gt∥2)
            for i, alpha in enumerate(model._arch_parameters):
                if alpha.grad is not None:
                    # alpha.grad.div_(avg_norm)
                    # inplace update alpha.grad
                    alpha.grad.div_(max_norm)
                    scores[i] += alpha.grad
                else:
                    scores[i] = alpha

            if step % args.report_freq == 0:
                print("train %03d, tau %f, trace loss %f", step, model._tau, loss)
                print("architecture: %s", (model.genotype(scores),))
                print("avg gap: %f", sum_gap / (step + 2))

                logging.info("train %03d, tau %f, trace loss %f", step, model._tau, loss)
                logging.info("architecture: %s", (model.genotype(scores),))
                logging.info("avg gap: %f", sum_gap / (step + 2))
                for s in scores:
                    print(s)

            if step == args.steps - 1:
                break

        print("\n step =", step)
        arch = model.genotype(scores)
        return arch

    def trace_loss(self, grads, gap, reg_weight) -> (float, float):
        """
        Evaluate score of a given architecture with only one batch
        :param grads: y grads on weights
        :param gap: v in paper, v = r * n* 1/learning rate
        :param reg_weight: mu in paper.
        :return: score and NTK
        """
        avg_eigen = sum([(g ** 2).sum() for g in grads])
        score = avg_eigen - reg_weight * F.relu(avg_eigen - gap)
        return score, avg_eigen

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

            # this will run the forward,
            # Sample gt ∼ p(g) = Gumbel(0, 1) and determine
            logits = arch(input)

            # writer.add_graph(arch, input)
            # writer.close()

            task_loss = F.cross_entropy(logits, targets)
            # Evaluate gradient Gt = ∇Theta L(x) with data Dt
            grads = grad(task_loss, model_params, create_graph=True)

            if is_adaptive:
                gap = sum_gap / (step + 1)
            else:
                gap = v

            score, avg_eigen = self.trace_loss(grads, (1 - sparsity) * gap, mu)
            print("Score is ", score)
            return score




