from src.search_space.core.model_params import ModelMacroCfg


class NB201MacroCfg(ModelMacroCfg):

    def __init__(self, init_channels, init_b_type, init_w_type, num_labels, bn):
        super().__init__(bn, init_channels, num_labels)
        self.init_b_type = init_b_type
        self.init_w_type = init_w_type
