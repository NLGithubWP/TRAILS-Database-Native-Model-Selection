

import json
import traceback
from common.constant import CommonVars
from logger import logger
from search_algorithm.utils.gpu_util import showUtilization
from storage import dataset
from sampler import sampler_register
from search_algorithm import evaluator_register
import search_space
import torch
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAutoNAS')

    # job config
    parser.add_argument('--log_name', type=str, default="result.json", help="log_name")

    # job config
    parser.add_argument('--num_arch', type=int, default=10, help="how many architecture to evaluate")

    # define base dir, where it stores apis, datasets, logs, etc,
    parser.add_argument('--base_dir', type=str, default="/Users/kevin/project_python/Fast-AutoNAS/data",
                        help='path of data folder')

    # dataLoader setting,
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='in one of [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--num_data_workers', type=int, default=1,
                        help='number of workers for dataLoader')
    parser.add_argument('--batch_sample_alg', type=str, default="grasp",
                        help='sample a mini batch, [random, grasp]')

    # define search space,
    parser.add_argument('--search_space', type=str, default="nasbench201",
                        help='which search space to use, [nasbench101, nasbench201, ... ]')

    parser.add_argument('--api_loc', type=str, default="NAS-Bench-201-v1_0-e61699.pth",
                        help='which search space to use, ['
                             'nasbench101: nasbench_only108.pkl '
                             'nasbench201: NAS-Bench-201-v1_0-e61699.pth '
                             ' ... ]')

    # define architecture sampler,
    parser.add_argument('--arch_sampler', type=str, default="random",
                        help='which architecture sampler to use, [test, random, BOHB, ...]')

    # define evaluation method
    parser.add_argument('--evaluation', type=str, default="synflow",
                        help='which evaluation algorithm to use, support following options'
                             '{'
                             'grad_norm: '
                             'grad_plain: '
                             'jacob_conv: '
                             'nas_wot: '
                             'ntk_cond_num: '
                             'ntk_trace: '
                             'ntk_trace_approx: '
                             'fisher: '
                             'grasp: '
                             'snip: '
                             'synflow: '
                             'weight_norm: '
                             'all_matrix: use all of the above, one by one')

    # define device
    parser.add_argument('--device', type=str, default="cpu",
                        help='which device to use [cpu, cuda]')

    # search space configs for nasBench101
    parser.add_argument('--stem_out_channels', default=128, type=int, help='output channels of stem convolution')
    parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules')
    parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack')

    #  weight initialization settings, nasBench201 requires it.
    parser.add_argument('--init_w_type', type=str, default='none',
                        help='weight initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_b_type', type=str, default='none',
                        help='bias initialization (before pruning) type [none, xavier, kaiming, zero]')

    parser.add_argument('--bn', type=int, default=1,
                        help="If use batch norm in network 1 = true, 0 = false")

    parser.add_argument('--arch_size', type=int, default=1,
                        help='How many node the architecture has at least')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()
    logger.info("running with params: :" + json.dumps(args.__dict__, indent=2))

    logger.info("cuda available = " + str(torch.cuda.is_available()))

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

    mini_batch = mini_batch.to(args.device)
    mini_batch_targets = mini_batch_targets.to(args.device)

    logger.info("1. Loaded batch data into GPU")
    # Show the utilization of all GPUs in a nice table
    showUtilization()

    used_search_space = search_space.init_search_space(args)

    # 1. Sampler one architecture
    sampler = sampler_register[args.arch_sampler]
    arch_generator = sampler.sample_next_arch(used_search_space, args.arch_size)

    # 2. Evaluator to evaluate one architecture
    evaluator_dict = {}
    if args.evaluation == CommonVars.ALL_EVALUATOR:
        evaluator_dict = evaluator_register
    else:
        evaluator_dict.update({args.evaluation: evaluator_register[args.evaluation]})

    # store result, in form of { arch_id: { scores: {}, queried_res: ... }}
    result = dict()

    i = 1
    try:
        for arch_id, architecture in arch_generator:
            logger.info("arch id = :" + str(arch_id))
            try:
                if i > args.num_arch:
                    logger.info("Finish Job")
                    break
                logger.info("--------------" + "Evaluate architecture with id = " + str(i) + " --------------")
                result[arch_id] = dict()
                result[arch_id]["scores"] = dict()
                # torch.autograd.set_detect_anomaly(True):
                for evaluator_name, evaluator in evaluator_dict.items():
                    # clone the architecture
                    new_arch = used_search_space.copy_architecture(arch_id, architecture)
                    new_arch = new_arch.to(args.device)

                    score, time_usage, gpu_usage = evaluator.evaluate_wrapper(
                        arch=new_arch,
                        pre_defined=args,
                        batch_data=mini_batch,
                        batch_labels=mini_batch_targets)
                    logger.info(evaluator_name + ":" +  str(score))
                    # records result,
                    result[arch_id]["scores"].update(
                        {evaluator_name: {"score": score,
                                          "time_usage": time_usage,
                                          "gpu_usage": gpu_usage}
                         }
                    )

                    # clean old graph
                    del new_arch
                    torch.cuda.empty_cache()

                # 3. Query to query the real performance
                result[arch_id].update(used_search_space.query_performance(arch_id))
                i = i + 1
            except Exception as e:
                logger.info("======================================================")
                logger.error(traceback.format_exc())
                logger.error("error when evaluate architecture " + str(arch_id) + ", message = "+str(e))
                logger.info("======================================================")
                del new_arch
                torch.cuda.empty_cache()
                del result[arch_id]
                import gc
                gc.collect()

    except Exception as e:
        logger.info("================================================================================================")
        logger.error(traceback.format_exc())
        logger.error("error: " + str(e))
        logger.info("================================================================================================")

    logger.info(json.dumps(result, indent=2))

    with open('./Logs/'+args.log_name, 'w') as outfile:
        outfile.write(json.dumps(result))
