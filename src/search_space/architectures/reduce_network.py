

import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from operations import *
from genotypes import PRIMITIVES
from genotypes import Genotype
from search_space.component.cell import Cell


class ReduceNetwork(nn.Module):
    def __init__(self ,C ,num_classes ,layers ,steps=4 ,multiplier=4 ,stem_multiplier=3,
                 init_alphas=0.01, gumbel=False, out_weight=False):
        super(ReduceNetwork, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        self._init_alphas = init_alphas
        self._gumbel = gumbel
        self._out_weight = out_weight

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps ,C_prev_prev ,C_prev ,C_curr ,reduction ,reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self.initialize_alphas()

    def forward(self, input):
        s0 = s1 = self.stem(input)

        for i, cell in enumerate(self.cells):
            start_idx = self._cell_params_length if cell.reduction else 0
            weights, output_weights, input_weights = self.get_cell_arch_params(start_idx)

            weights = self.normalize(weights, self._gumbel)
            output_weights = self.normalize(output_weights, self._gumbel) if self._out_weight else None
            new_input_weights = []
            for j in range(self._steps):
                new_input_weights += [self.normalize(input_weights[j], self._gumbel)]
            s0, s1 = s1, cell(s0, s1, weights, output_weights, new_input_weights)

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))

        return logits

    def initialize_alphas(self):
        NUM_CELL_INPUTS = 2
        TYPES_OF_CELL = 2
        OPS_PER_STEP = 1

        self._arch_parameters = []
        for _ in range(TYPES_OF_CELL):
            self.alphas_mixed_op = Variable(
                self._init_alphas * torch.rand(self._steps * OPS_PER_STEP, len(PRIMITIVES)).cuda(),
                requires_grad=True,
                )

            self.alphas_output = Variable(
                self._init_alphas * torch.rand(self._steps).cuda(),
                requires_grad=True,
                )

            self.alphas_inputs = [
                Variable(
                    self._init_alphas * torch.rand(n_inputs).cuda(),
                    requires_grad=True
                )
                for n_inputs in range(NUM_CELL_INPUTS,  self._steps + NUM_CELL_INPUTS)
            ]

            self._arch_parameters += [self.alphas_mixed_op, self.alphas_output, *self.alphas_inputs]
        self._cell_params_length = len(self._arch_parameters) // TYPES_OF_CELL

    def get_cell_arch_params(self, start_idx):
        weights = self._arch_parameters[start_idx]
        output_weights = self._arch_parameters[start_idx +1]
        input_weights = self._arch_parameters[start_idx +2:]
        return weights, output_weights, input_weights

    def reset_arch_trainable(self, train=False):
        for alpha in self._arch_parameters:
            alpha.requires_grad = train

    def reset_zero_grads(self):
        self.zero_grad()
        for p in self._arch_parameters:
            if p.grad is not None:
                p.grad.zero_()

    def arch_param_grad_norm(self, grads=None):
        norm = 0
        eps = 1e-5
        if grads is None:
            for p in self._arch_parameters:
                if p.grad is not None:
                    norm += (p.grad**2).sum()
            return (norm + eps).sqrt()
        else:
            for g in grads:
                if g.grad is not None:
                    norm += ( g**2).sum()
            return (norm + eps).sqrt()

    def normalize(self, x, gumbel=False):
        if gumbel:
            return F.gumbel_softmax(x, dim=-1, hard=True, tau=1)
        else:
            return F.softmax(x, dim=-1)

    def genotype(self):
        def _parse(weights, input_weights, output_weights, num_outputs):
            ops_idx = np.argmax(weights, axis=-1)
            out_idx = np.argsort(output_weights, axis=-1)[-num_outputs:]
            inp_idx = []
            for i in range(self._steps):
                inp_idx += np.argsort(input_weights[i], axis=-1).tolist()[-2:]

            gene = []
            for i, op_idx in enumerate(ops_idx):
                gene += [(PRIMITIVES[op_idx], (inp_idx[ 2 *i], inp_idx[ 2 * i +1]))]
            return gene, out_idx

        normal_weights, normal_output_weights, normal_input_weights = self.get_cell_arch_params(0)
        reduce_weights, reduce_output_weights, reduce_input_weights = self.get_cell_arch_params \
            (self._cell_params_length)

        gene_normal, out_normal = _parse(
            weights=normal_weights.data.cpu().numpy(),
            output_weights=normal_output_weights.data.cpu().numpy(),
            input_weights=[x.data.cpu().numpy() for x in normal_input_weights],
            num_outputs=self._multiplier
        )

        gene_reduce, out_reduce = _parse(
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