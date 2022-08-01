from search_space.core.model_params import ModelCfgs


class NasBench201Cfg(ModelCfgs):

    def __init__(self, init_b_type, init_w_type, num_labels, bn):
        super().__init__(bn)
        self.init_b_type = init_b_type
        self.init_w_type = init_w_type
        self.num_labels = num_labels
