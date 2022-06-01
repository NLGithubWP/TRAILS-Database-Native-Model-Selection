
import traceback
from storage import dataset
from sampler import sampler_register
from search_algorithm import evaluator_register
import search_space

import argparse
import logging

logger = logging.getLogger('Log')


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAutoNAS')

    # define base dir, where it stores apis, datasets, logs, etc,
    parser.add_argument('--base_dir', type=str, default="./data", help='path of data folder')

    # dataLoader setting,
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--dataset', type=str, default='cifar10', help='in one of [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--num_data_workers', type=int, default=1, help='number of workers for dataLoader')
    parser.add_argument('--batch_sample_alg', type=str, default="grasp",
                        help='sample a mini batch, [random, grasp]')

    # define search space,
    parser.add_argument('--search_space', type=str, default="nasbench101",
                        help='which search space to use, [nasbench101, nasbench201, ... ]')

    # define architecture sampler,
    parser.add_argument('--arch_sampler', type=str, default="random",
                        help='which architecture sampler to use, [random, BOHB, ...]')

    # define evaluation method
    parser.add_argument('--evaluation', type=str, default="snip",
                        help='which evaluation algorithm to use, [nas_wot, ntk_trace, syn_flow, snip ]')

    # define device
    parser.add_argument('--device', type=str, default="cpu",
                        help='which device to use [cpu, gpu1, gpu2...]')

    # search space configs for nasBench101
    parser.add_argument('--stem_out_channels', default=128, type=int, help='output channels of stem convolution')
    parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules')
    parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()

    # define dataLoader, and sample a mini-batch
    train_loader, val_loader, class_num = dataset.get_dataloader(
        train_batch_size=1,
        test_batch_size=1,
        dataset=args.dataset,
        num_workers=args.num_data_workers,
        datadir=args.base_dir)
    args.num_labels = class_num

    # sample a batch with random or GRASP
    mini_batch, mini_batch_targets = dataset.get_mini_batch(
        dataloader=train_loader, sample_alg=args.batch_sample_alg, batch_size=args.batch_size, num_classes=class_num)

    mini_batch.to(args.device)
    mini_batch_targets.to(args.device)

    # define search space
    # 101
    args.nasspace = "nasbench101"
    args.api_loc = "./data/nasbench_only108.pkl"
    # 201
    # args.nasspace = "nasbench201"
    # args.api_loc = "./data/NAS-Bench-201-v1_0-e61699.pth"
    used_search_space = search_space.init_search_space(args)

    # 1. Sampler one architecture
    sampler = sampler_register[args.arch_sampler]

    # 2. Evaluator to evaluate one architecture
    evaluator = evaluator_register[args.evaluation]

    for i in range(40):
        print("--------------" + "Sample for the id = " + str(i) + "--------------")
        try:
            architecture, unique_hash = sampler.sample_one_arch(used_search_space, 7)

            architecture.to(args.device)

            # torch.autograd.set_detect_anomaly(True):
            score = evaluator.evaluate(arch=architecture,
                                       pre_defined=args,
                                       batch_data=mini_batch,
                                       batch_labels=mini_batch_targets)

            # 3. Query to query the real performance
            real_res, statics = used_search_space.query_result(unique_hash)

            print(real_res, statics)
            logger.info("Evaluation score is " + str(score))
            logger.info("Queried performance is " + str(statics))
            logger.info("Queried full-result is " + real_res)

        except Exception as e:
            logger.info(traceback.format_exc())
            logger.info("error: " + str(e))
            raise
