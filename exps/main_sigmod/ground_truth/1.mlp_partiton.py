import argparse


from controller.sampler_all.seq_sampler import SequenceSampler
from search_space.init_search_space import init_search_space
from utilslibs.io_tools import write_json


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAutoNAS')

    # define search space,
    parser.add_argument('--search_space', type=str, default="mlp_sp", help='[nasbench101, nasbench201, mlp_sp]')

    parser.add_argument('--num_labels', type=int, default=1, help='[10, 100, 120, 2, 2, 2]')

    # those are for training
    parser.add_argument('--num_layers', default=4, type=int, help='# hidden layers')
    parser.add_argument('--hidden_choice_len', default=10, type=int, help='number of hidden layer choices, 10 or 20')

    parser.add_argument('--nfeat', type=int, default=5500, help='the number of features')
    parser.add_argument('--nfield', type=int, default=10, help='the number of fields')
    parser.add_argument('--nemb', type=int, default=10, help='embedding size')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    search_space_ins = init_search_space(args)

    sper = SequenceSampler(search_space_ins, args)
    gen = sper.sample_next_arch()

    all_models = {}
    index = 0
    for arch_id, _ in gen:
        all_models[index] = arch_id
        index += 1
    print(f"Partition done, total models = {index}")
    write_json(f"sampled_models_{index}_models.json", all_models)



