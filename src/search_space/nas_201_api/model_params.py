from search_space.core.model_params import ModelCfgs


class NasBench201Cfg(ModelCfgs):

    def __init__(self,
                 dataset_name,
                 init_b_type,
                 init_w_type,
                 num_labels,
                 bn):
        self.dataset_name = dataset_name
        self.init_b_type = init_b_type
        self.init_w_type = init_w_type
        self.num_labels = num_labels
        self.bn = bn
