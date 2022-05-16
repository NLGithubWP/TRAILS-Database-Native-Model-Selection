

from third_party.AutoDL_Projects.xautodl.models import get_cell_based_tiny_net, get_cifar_models
from third_party.NATS_Bench.nats_bench import create

api = create(None, 'tss', fast_mode=True, verbose=True)

config = api.get_net_config(15620, 'cifar10')
print(api.arch(15620))

network = get_cell_based_tiny_net(config)

# net2 = get_cifar_models(config)
print(network)

info = api.get_more_info(15620, 'cifar10')
print(info)

params = api.get_net_param(15620, 'cifar10', None)
network.load_state_dict(next(iter(params.values())))

print(network)