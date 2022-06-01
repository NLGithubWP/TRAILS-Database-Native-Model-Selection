
from search_space.core.space import SpaceWrapper


class NDS(SpaceWrapper):
    pass
#     def __init__(self, searchspace):
#         self.searchspace = searchspace
#         data = json.load(open(f'nds_data/{searchspace}.json', 'r'))
#         try:
#             data = data['top'] + data['mid']
#         except Exception as e:
#             pass
#         self.data = data
#
#     def __iter__(self):
#         for unique_hash in range(len(self)):
#             network = self.get_network(unique_hash)
#             yield unique_hash, network
#
#     def get_network_config(self, uid):
#         return self.data[uid]['net']
#
#     def get_network_optim_config(self, uid):
#         return self.data[uid]['optim']
#
#     def get_network(self, uid):
#         netinfo = self.data[uid]
#         config = netinfo['net']
#         #print(config)
#         if 'genotype' in config:
#             #print('geno')
#             gen = config['genotype']
#             genotype = Genotype(normal=gen['normal'], normal_concat=gen['normal_concat'], reduce=gen['reduce'], reduce_concat=gen['reduce_concat'])
#             if '_in' in self.searchspace:
#                 network = NetworkImageNet(config['width'], 1, config['depth'], config['aux'],  genotype)
#             else:
#                 network = NetworkCIFAR(config['width'], 1, config['depth'], config['aux'],  genotype)
#             network.drop_path_prob = 0.
#             #print(config)
#             #print('genotype')
#             L = config['depth']
#         else:
#             if 'bot_muls' in config and 'bms' not in config:
#                 config['bms'] = config['bot_muls']
#                 del config['bot_muls']
#             if 'num_gs' in config and 'gws' not in config:
#                 config['gws'] = config['num_gs']
#                 del config['num_gs']
#             config['nc'] = 1
#             config['se_r'] = None
#             config['stem_w'] = 12
#             L = sum(config['ds'])
#             if 'ResN' in self.searchspace:
#                 config['stem_type'] = 'res_stem_in'
#             else:
#                 config['stem_type'] = 'simple_stem_in'
#             #"res_stem_cifar": ResStemCifar,
#             #"res_stem_in": ResStemIN,
#             #"simple_stem_in": SimpleStemIN,
#             if config['block_type'] == 'double_plain_block':
#                 config['block_type'] = 'vanilla_block'
#             network = AnyNet(**config)
#         return_feature_layer(network)
#         return network
#
#     def __getitem__(self, index):
#         return index
#
#     def __len__(self):
#         return len(self.data)
#
#     def random_arch(self):
#         return random.randint(0, len(self.data)-1)
#
#     def get_final_accuracy(self, uid, acc_type, trainval):
#         return 100.-self.data[uid]['test_ep_top1'][-1]
#
#
# def return_feature_layer(network, prefix=''):
#     #for attr_str in dir(network):
#     #    target_attr = getattr(network, attr_str)
#     #    if isinstance(target_attr, torch.nn.Linear):
#     #        setattr(network, attr_str, ReturnFeatureLayer(target_attr))
#     for n, ch in list(network.named_children()):
#         if isinstance(ch, torch.nn.Linear):
#             setattr(network, n, ReturnFeatureLayer(ch))
#         else:
#             return_feature_layer(ch, prefix + '\t')
#
#
# class ReturnFeatureLayer(torch.nn.Module):
#     def __init__(self, mod):
#         super(ReturnFeatureLayer, self).__init__()
#         self.mod = mod
#     def forward(self, x):
#         return self.mod(x), x
#
