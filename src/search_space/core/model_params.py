

class ModelCfgs:

    def __init__(self, bn, init_channels):
        """
        Each model cfg must have bn option.
        :param bn:
        """
        self.bn = bn
        self.init_channels = init_channels
