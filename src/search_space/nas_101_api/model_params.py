from src.search_space.core.model_params import ModelMacroCfg


class NB101MacroCfg(ModelMacroCfg):

    def __init__(self, init_channels, num_stacks, num_modules_per_stack, num_labels, bn):
        super().__init__(bn, init_channels, num_labels)
        self.num_stacks = num_stacks
        self.num_modules_per_stack = num_modules_per_stack
