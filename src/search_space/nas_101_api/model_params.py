from search_space.core.model_params import ModelCfgs


class NasBench101Cfg(ModelCfgs):

    def __init__(self,
                 stem_out_channels,
                 num_stacks,
                 num_modules_per_stack,
                 num_labels):
        self.stem_out_channels = stem_out_channels
        self.num_stacks = num_stacks
        self.num_modules_per_stack = num_modules_per_stack
        self.num_labels = num_labels
