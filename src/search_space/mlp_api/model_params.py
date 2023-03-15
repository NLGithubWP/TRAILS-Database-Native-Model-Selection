from search_space.core.model_params import ModelMacroCfg


class MlpMacroCfg(ModelMacroCfg):

    def __init__(self, input_fea_dims, num_layers: int, num_labels: int, layer_choices: list, bn: bool):
        super().__init__(bn, input_fea_dims, num_labels)
        self.layer_choices = layer_choices
        self.num_layers = num_layers
