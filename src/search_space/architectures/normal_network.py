
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from operations import *
from genotypes import PRIMITIVES
from genotypes import Genotype
from search_space.component.cell import Cell


class NormalNetwork(nn.Module):

    def __init__(self, input_channel,
                 num_classes: int,
                 layers,
                 steps=4, multiplier=4,
                 stem_multiplier=3,
                 init_alphas=0.01,
                 gumbel=False,
                 out_weight=False):
        """
        :param input_channel:
        :param num_classes:
        :param layers: how many cell inside this network.
        :param steps: how many nodes inside each cell
        :param multiplier:
        :param stem_multiplier:
        :param init_alphas:
        :param gumbel:
        :param out_weight:
        """
        super(NormalNetwork, self).__init__()
        self._cell_params_length = None
        self.alphas_inputs = None
        self.alphas_output = None
        self.alphas_mixed_op = None
        self._arch_parameters = None
        self.count_params = None
        self._C = input_channel
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        self._init_alphas = init_alphas
        self._gumbel = gumbel
        self._tau = 1
        self._out_weight = out_weight

        c_curr = stem_multiplier * input_channel #48
        self.stem = nn.Sequential(
            nn.Conv2d(3, c_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_curr, affine=False)
        )

        # init channel
        c_prev_prev, c_prev, c_curr = c_curr, c_curr, input_channel

        # stack multiple cells and form a Module later
        self.cells = nn.ModuleList()

        # if previous cell is a reduction cell.
        reduction_prev = False
        # if current cell is a reduction cell.
        reduction = False

        # if layers = 9, then 3 and 6 are the reduction cell.
        reduction_index = [layers // 3, 2 * layers // 3]

        for i in range(layers):
            if i in reduction_index:
                c_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, c_prev_prev, c_prev, c_curr, reduction, reduction_prev)
            # update information for next cell
            reduction_prev = reduction
            self.cells += [cell]
            c_prev_prev, c_prev = c_prev, multiplier * c_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev, num_classes)
        self.initialize_alphas()

    def forward(self, input):
        """
        Calculate the output for the network
        :param input: contains output of the previous two cells.
        :return:
        """
        s0 = s1 = self.stem(input)
        count_params = []

        for i, cell in enumerate(self.cells):

            # Sample gt ∼ p(g) = Gumbel(0, 1) and determine sampled architecture At based on α0 and gt
            normal_weights = self.sample(reduction=False)
            reduce_weights = self.sample(reduction=True)

            if not cell.reduction:
                weights, output_weights, new_input_weights = normal_weights
            else:
                weights, output_weights, new_input_weights = reduce_weights
            s0, s1 = s1, cell(s0, s1, weights, output_weights, new_input_weights)

            count_params += [cell.count_params]

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))

        self.count_params = sum(count_params)

        return logits

    def initialize_alphas(self):
        """
        Init the whole computation graph
        :return: None
        """
        NUM_CELL_INPUTS = 2
        TYPES_OF_CELL = 2
        OPS_PER_STEP = 2

        self._arch_parameters = []
        for _ in range(TYPES_OF_CELL):
            # alphas_mixed_op
            if torch.cuda.is_available():
                self.alphas_mixed_op = Variable(
                    self._init_alphas * torch.rand(self._steps * OPS_PER_STEP, len(PRIMITIVES)).cuda(), requires_grad=True)
            else:
                self.alphas_mixed_op = Variable(
                    self._init_alphas * torch.rand(self._steps * OPS_PER_STEP, len(PRIMITIVES)), requires_grad=True)

            # alphas_output
            if torch.cuda.is_available():
                self.alphas_output = Variable(self._init_alphas * torch.rand(self._steps).cuda(), requires_grad=True)
            else:
                self.alphas_output = Variable(self._init_alphas * torch.rand(self._steps), requires_grad=True)

            # alphas_inputs
            if torch.cuda.is_available():
                self.alphas_inputs = [
                    Variable(
                        self._init_alphas * torch.rand(OPS_PER_STEP, n_inputs).cuda(),
                        requires_grad=True)
                    for n_inputs in range(NUM_CELL_INPUTS, self._steps + NUM_CELL_INPUTS)]
            else:
                self.alphas_inputs = [
                    Variable(
                        self._init_alphas * torch.rand(OPS_PER_STEP, n_inputs),
                        requires_grad=True)
                    for n_inputs in range(NUM_CELL_INPUTS, self._steps + NUM_CELL_INPUTS)]

            # records all combinations
            self._arch_parameters += [self.alphas_mixed_op, self.alphas_output, *self.alphas_inputs]

        # how many params for each cell
        self._cell_params_length = len(self._arch_parameters) // TYPES_OF_CELL

    def get_cell_arch_params(self, start_idx, alphas=None):
        if alphas is None:
            alphas = self._arch_parameters
        weights = alphas[start_idx]
        output_weights = alphas[start_idx + 1]
        input_weights = alphas[start_idx + 2:]
        return weights, output_weights, input_weights

    def sample(self, reduction=False):
        """

        :param reduction:
        :return:
        """
        start_idx = self._cell_params_length if reduction else 0
        weights, output_weights, input_weights = self.get_cell_arch_params(start_idx)
        weights = self.normalize(weights, self._gumbel)
        output_weights = self.normalize(output_weights, self._gumbel) if self._out_weight else None
        new_input_weights = []
        for j in range(self._steps):
            new_input_weights += [self.normalize(input_weights[j], self._gumbel)]
        return weights, output_weights, new_input_weights

    def arch_param_grad_norm(self, grads=None):
        """
        Calculate ||Gt||_2 where Gt = ∇α0 R(At)
        :param grads: grads
        :return:
        """
        norm = 0
        eps = 1e-5
        if grads is None:
            for p in self._arch_parameters:
                if p.grad is not None:
                    norm += (p.grad ** 2).sum()
            return (norm + eps).sqrt()
        else:
            for g in grads:
                if g.grad is not None:
                    norm += (g ** 2).sum()
            return (norm + eps).sqrt()

    def reset_zero_grads(self):
        self.zero_grad()
        for p in self._arch_parameters:
            if p.grad is not None:
                p.grad.zero_()

    def normalize(self, x, gumbel=False):
        if gumbel:
            return F.gumbel_softmax(x, dim=-1, hard=True, tau=self._tau)
        else:
            return F.softmax(x, dim=-1)

    def genotype(self, alphas=None):
        def _parse(weights, input_weights, output_weights, num_outputs):
            ops_idx = np.argmax(weights, axis=-1)
            out_idx = np.argsort(output_weights, axis=-1)[-num_outputs:]
            inp_idx = []
            for i in range(self._steps):
                # inp_idx += np.argmax(input_weights[i], axis=-1).tolist()
                # w = np.max(input_weights[i], axis=0)
                # idx = np.argsort(w, axis=-1)[-2:].tolist()

                # # correct order
                # if np.max(w, axis=-1) == np.max(input_weights[i][0], axis=0):
                #     idx = idx[::-1]
                # inp_idx += idx

                w = input_weights[i]
                row_idx, col_idx = np.unravel_index(np.argmax(w, axis=None), w.shape)
                w_2 = w[int(1 - row_idx)]
                w_2[col_idx] = - 1e5
                col_idx_2 = np.argmax(w_2)
                idx = [col_idx, col_idx_2] if row_idx == 0 else [col_idx_2, col_idx]
                inp_idx += idx

            gene = []
            for i, op_idx in enumerate(ops_idx):
                gene += [(PRIMITIVES[op_idx], inp_idx[i])]
            return gene, ops_idx, out_idx

        normal_weights, normal_output_weights, normal_input_weights = self.get_cell_arch_params(0, alphas)
        reduce_weights, reduce_output_weights, reduce_input_weights = self.get_cell_arch_params(
            self._cell_params_length, alphas)

        gene_normal, ops_normal, out_normal = _parse(
            weights=normal_weights.data.cpu().numpy(),
            output_weights=normal_output_weights.data.cpu().numpy(),
            input_weights=[x.data.cpu().numpy() for x in normal_input_weights],
            num_outputs=self._multiplier
        )

        gene_reduce, ops_reduce, out_reduce = _parse(
            weights=reduce_weights.data.cpu().numpy(),
            output_weights=reduce_output_weights.data.cpu().numpy(),
            input_weights=[x.data.cpu().numpy() for x in reduce_input_weights],
            num_outputs=self._multiplier
        )

        if not self._out_weight:
            out_normal = list(range(2 + self._steps - self._multiplier, self._steps + 2))
            out_reduce = list(range(2 + self._steps - self._multiplier, self._steps + 2))

        genotype = Genotype(
            normal=gene_normal,
            normal_concat=out_normal,
            reduce=gene_reduce,
            reduce_concat=out_reduce,
        )

        return genotype









