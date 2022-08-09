

class CommonVars:

    # SAMPLER
    TEST_SAMPLER = "sequence"
    RANDOM_SAMPLER = "random"
    RL_SAMPLER = "rl"
    EA_SAMPLER = "ea"
    BOHB_SAMPLER = "bohb"

    # EVALUATOR
    GRAD_NORM = "grad_norm"
    GRAD_PLAIN = "grad_plain"

    JACOB_CONV = "jacob_conv"
    NAS_WOT = "nas_wot"

    NTK_CONDNUM = "ntk_cond_num"
    NTK_TRACE = "ntk_trace"
    NTK_TRACE_APPROX = "ntk_trace_approx"

    PRUNE_FISHER = "fisher"
    PRUNE_GRASP = "grasp"
    PRUNE_SNIP = "snip"
    PRUNE_SYNFLOW = "synflow"

    WEIGHT_NORM = "weight_norm"

    ALL_EVALUATOR = "all_matrix"

    # SEARCH SPACE
    NASBENCH101 = "nas-bench-101"
    NASBENCH201 = "nas-bench-201"

    # correlation coefficient metrics
    KendallTau = "KendallTau"
    Spearman = "Spearman"
    Pearson = "Pearson"
    AvgCorrelation = "average_correlation"
    AllCorrelation = "all_correlation"

