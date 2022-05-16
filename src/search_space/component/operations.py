

from operations import *
from genotypes import PRIMITIVES
from utils import utils


class MixedOp(nn.Module):
    def __init__(self, input_channel, stride):
        """
        Mix all the operation in ops list, which is like darts search space
        :param input_channel:
        :param stride:
        """
        super(MixedOp, self).__init__()
        self.count_params = None
        self.ops = nn.ModuleList()

        # primitive: operation name
        # append all operations between two nodes.
        for primitive in PRIMITIVES:
            op = OPS[primitive](input_channel, stride, False)
            # if "pool" in primitive:
            #     op = nn.Sequential(op, nn.BatchNorm2d(input_channel, affine=False))
            self.ops.append(op)

    def forward(self, x, weights):
        """
        Calculate weighted operation for the final input for a node.
        :param x: input x
        :param weights: softmax in form of alpha
        :return: weighted sum result O_ij(x)
        """
        # count weighted parameter number
        self.count_params = sum(w * utils.count_parameters(op) for w, op in zip(weights, self.ops))
        return sum(w * op(x) for w, op in zip(weights, self.ops))


class MixedInpOp(nn.Module):
    def __init__(self, input_channel, stride):
        super(MixedInpOp, self).__init__()
        self.count_params = None
        self.mixed_op = MixedOp(input_channel, stride)

    def forward(self, inputs, input_weights, weights):
        """
        Gather all proceeding nodes' output as input, and calculate current node's output
        :param inputs: input vector for all previous nodes.
        :param input_weights: weight of each input to one node
        :param weights: weight for each operation between nodes.
        :return:
        """
        # before feeding previous nodes' output,
        # weighted their inputs and then get sum, which is the final input for each node
        node_inputs = sum([w * t for w, t in zip(input_weights, inputs)])
        # calculate by using mixed_ops
        output = self.mixed_op(node_inputs, weights=weights)
        self.count_params = self.mixed_op.count_params
        return output
