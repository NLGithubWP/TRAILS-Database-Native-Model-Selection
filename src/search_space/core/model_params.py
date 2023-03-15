

class ModelMacroCfg:
    """
    Macro search space config
    Search Space basic init,  use bn or not, input features, output labels, etc. 
    """

    def __init__(self, bn, init_channels, num_labels):
        """
        Args:
            bn: use bn or not
            init_channels: input feature dim,  
            num_labels: output labels. 
        """
        self.init_channels = init_channels
        self.bn = bn
        self.num_labels = num_labels


class ModelMicroCfg:

    """
    Micro space cfg
    Identifier for each model, connection patter, operations etc.
    """
    def __init__(self):
        pass

