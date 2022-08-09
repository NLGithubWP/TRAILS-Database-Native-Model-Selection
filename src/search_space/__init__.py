
import os
from .core.space import SpaceWrapper
from .nas_101_api.model_params import NasBench101Cfg
from .nas_101_api.space import NasBench101Space
from .nas_201_api.model_params import NasBench201Cfg
from .nas_201_api.space import NasBench201Space


def init_search_space(args) -> SpaceWrapper:

    if args.bn == 1:
        bn = True
    else:
        bn = False

    if args.search_space == 'nasbench101':

        model_cfg = NasBench101Cfg(
            args.init_channels,
            args.num_stacks,
            args.num_modules_per_stack,
            args.num_labels,
            bn)

        return NasBench101Space(os.path.join(args.base_dir, args.api_loc), model_cfg)

    elif args.search_space == 'nasbench201':

        model_cfg = NasBench201Cfg(
            args.init_channels,
            args.init_b_type,
            args.init_w_type,
            args.num_labels,
            bn)

        return NasBench201Space(os.path.join(args.base_dir, args.api_loc), model_cfg)

    # elif args.nasspace == 'nds_resnet':
    #     return NDS('ResNet')
    # elif args.nasspace == 'nds_amoeba':
    #     return NDS('Amoeba')
    # elif args.nasspace == 'nds_amoeba_in':
    #     return NDS('Amoeba_in')
    # elif args.nasspace == 'nds_darts_in':
    #     return NDS('DARTS_in')
    # elif args.nasspace == 'nds_darts':
    #     return NDS('DARTS')
    # elif args.nasspace == 'nds_darts_fix-w-d':
    #     return NDS('DARTS_fix-w-d')
    # elif args.nasspace == 'nds_darts_lr-wd':
    #     return NDS('DARTS_lr-wd')
    # elif args.nasspace == 'nds_enas':
    #     return NDS('ENAS')
    # elif args.nasspace == 'nds_enas_in':
    #     return NDS('ENAS_in')
    # elif args.nasspace == 'nds_enas_fix-w-d':
    #     return NDS('ENAS_fix-w-d')
    # elif args.nasspace == 'nds_pnas':
    #     return NDS('PNAS')
    # elif args.nasspace == 'nds_pnas_fix-w-d':
    #     return NDS('PNAS_fix-w-d')
    # elif args.nasspace == 'nds_pnas_in':
    #     return NDS('PNAS_in')
    # elif args.nasspace == 'nds_nasnet':
    #     return NDS('NASNet')
    # elif args.nasspace == 'nds_nasnet_in':
    #     return NDS('NASNet_in')
    # elif args.nasspace == 'nds_resnext-a':
    #     return NDS('ResNeXt-A')
    # elif args.nasspace == 'nds_resnext-a_in':
    #     return NDS('ResNeXt-A_in')
    # elif args.nasspace == 'nds_resnext-b':
    #     return NDS('ResNeXt-B')
    # elif args.nasspace == 'nds_resnext-b_in':
    #     return NDS('ResNeXt-B_in')
    # elif args.nasspace == 'nds_vanilla':
    #     return NDS('Vanilla')
    # elif args.nasspace == 'nds_vanilla_lr-wd':
    #     return NDS('Vanilla_lr-wd')
    # elif args.nasspace == 'nds_vanilla_lr-wd_in':
    #     return NDS('Vanilla_lr-wd_in')
