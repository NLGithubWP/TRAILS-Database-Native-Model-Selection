from common.constant import *
from search_algorithm.grad_norm import GradNormEvaluator
from search_algorithm.grad_plain import GradPlainEvaluator
from search_algorithm.jacob_cov import JacobConvEvaluator
from search_algorithm.nas_wot import NWTEvaluator
from search_algorithm.ntk_condition_num import NTKCondNumEvaluator
from search_algorithm.ntk_trace import NTKTraceEvaluator
from search_algorithm.ntk_trace_approx import NTKTraceApproxEvaluator
from search_algorithm.prune_fisher import FisherEvaluator
from search_algorithm.prune_grasp import GraspEvaluator
from search_algorithm.prune_snip import SnipEvaluator
from search_algorithm.prune_synflow import SynFlowEvaluator
from search_algorithm.weight_norm import WeightNormEvaluator

# evaluator mapper to register many existing evaluation algorithms
evaluator_register = {

    # # sum on gradient
    CommonVars.GRAD_NORM: GradNormEvaluator(),
    CommonVars.GRAD_PLAIN: GradPlainEvaluator(),
    #
    # # training free matrix
    # CommonVars.JACOB_CONV: JacobConvEvaluator(),
    CommonVars.NAS_WOT: NWTEvaluator(),

    # this is ntk based
    CommonVars.NTK_CONDNUM: NTKCondNumEvaluator(),
    CommonVars.NTK_TRACE: NTKTraceEvaluator(),

    CommonVars.NTK_TRACE_APPROX: NTKTraceApproxEvaluator(),

    # # prune based
    CommonVars.PRUNE_FISHER: FisherEvaluator(),
    CommonVars.PRUNE_GRASP: GraspEvaluator(),
    CommonVars.PRUNE_SNIP: SnipEvaluator(),
    CommonVars.PRUNE_SYNFLOW: SynFlowEvaluator(),

    # # sum of weight
    CommonVars.WEIGHT_NORM: WeightNormEvaluator(),

}

