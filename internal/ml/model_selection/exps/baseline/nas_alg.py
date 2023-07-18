from src.common.constant import Config


def get_base_annotations(dataset_name):
    """
    This is from HNAS
    :param dataset_name:
    :return:
    """
    if dataset_name == Config.c10:
        annotations = [
            # ["REA (Training-based)", 93.92, 12000],
            # ["RS (Training-based)", 93.70, 12000],
            # ["REINFORCE (Training-based)", 93.85, 12000],
            # ["BOHB (Training-based)", 93.61, 12000],

            ["ENAS (Weight sharing)", 54.30, 13314.51],
            ["DARTS-V1 (Weight sharing)", 54.30, 16281],
            ["DARTS-V2 (Weight sharing)", 54.30, 43277],

            # ["NASWOT (Training-Free)", 92.96, 306],
            ["TE-NAS (Training-Free)", 93.90, 1558],
            ["KNAS (Training-Free)", 93.05, 4200],
        ]
    elif dataset_name == Config.c100:
        annotations = [
            # ["REA (Training-based)", 93.92, 12000],
            # ["RS (Training-based)", 93.70, 12000],
            # ["REINFORCE (Training-based)", 93.85, 12000],
            # ["BOHB (Training-based)", 93.61, 12000],

            ["ENAS (Weight sharing)", 15.61, 13314.51],
            ["DARTS-V1 (Weight sharing)", 15.61, 16281],
            ["DARTS-V2 (Weight sharing)", 15.61, 43277],

            # ["NASWOT (Training-Free)", 69.98, 306],
            ["TE-NAS (Training-Free)", 71.24, 1558],
            ["KNAS (Training-Free)", 68.91, 4200],
        ]
    elif dataset_name == Config.imgNet:
        annotations = [
            # ["REA (Training-based)", 93.92, 12000],
            # ["RS (Training-based)", 93.70, 12000],
            # ["REINFORCE (Training-based)", 93.85, 12000],
            # ["BOHB (Training-based)", 93.61, 12000],

            ["ENAS (Weight sharing)", 16.32, 13314.51],
            ["DARTS-V1 (Weight sharing)", 16.32, 16281],
            ["DARTS-V2 (Weight sharing)", 16.32, 43277],

            # ["NASWOT (Training-Free)", 44.44, 306],
            ["TE-NAS (Training-Free)", 42.38, 1558],
            ["KNAS (Training-Free)", 34.11, 4200],
        ]
    else:
        annotations = []
    return annotations
