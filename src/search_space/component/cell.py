
from operations import *
from utils import utils
from search_space.component.operations import MixedInpOp


class Cell(nn.Module):
    def __init__(self, nodes, c_prev_prev, c_prev, current_channel, reduction, reduction_prev):
        """
        :param nodes: how many nodes inside this cell
        :param c_prev_prev: input, output of one of two preceding cells
        :param c_prev: input, output of one of two preceding cells
        :param current_channel:
        :param reduction:
        :param reduction_prev:
        """
        super(Cell, self).__init__()
        self.nodes = nodes
        self.count_params = None
        self.reduction = reduction
        self.ops_per_node = 2

        # the operation on output of the pre-pre cell
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(c_prev_prev, current_channel, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(c_prev_prev, current_channel, 1, 1, 0, affine=False)

        # the operation on output of the pre cell
        self.preprocess1 = ReLUConvBN(c_prev, current_channel, 1, 1, 0, affine=False)

        # if this is reduction cell, use pooling to reduce the size of previous normal cell's output.
        # a max-pooling operation in between normal and reduction cell is applied to
        # down-sampling intermediate features during the search process
        if reduction:
            self.preprocess0 = nn.Sequential(
                self.preprocess0,
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.preprocess1 = nn.Sequential(
                self.preprocess1,
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.ops = nn.ModuleList()

        # calculate the mixed operations for k times.
        for i in range(self.nodes * self.ops_per_node):
            op = MixedInpOp(current_channel, stride=1)
            self.ops.append(op)

    def forward(self, s0, s1, weights, output_weights, input_weights):
        """
        Cell run forward, calculate the output of each cell.
        :param s0: output of pre-pre cell,
        :param s1: output of pre cell,
        :param weights: weight for each operation between nodes.
        :param output_weights: weight used when Concatenation to output.
        :param input_weights: input weight vector
        :return:
        """

        # 1. process the output, which are inputs of current cell.
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        count_params = [
            utils.count_parameters(self.preprocess0),
            utils.count_parameters(self.preprocess1)
        ]

        # 2. compute and record output value for each node.
        states = [s0, s1]
        # each nodes' input is the output of all previous nodes in this cell
        for i in range(self.nodes):
            # following nodes will accept all proceeding nodes as inputs
            s = sum([
                self.ops[2 * i](states, input_weights[i][0], weights[2 * i]),
                self.ops[2 * i + 1](states, input_weights[i][1], weights[2 * i + 1])
            ])
            states.append(s)

        # 3. Add weight to output of each node inside each cell.
        if output_weights is not None:
            out_states = [w * t for w, t in zip(output_weights, states[2:])]
        else:
            out_states = states[2:]

        count_params += [op.count_params for op in self.ops]
        self.count_params = sum(count_params)

        # 4. Concatenation of all intermediate nodes
        return torch.cat(out_states, dim=1)
