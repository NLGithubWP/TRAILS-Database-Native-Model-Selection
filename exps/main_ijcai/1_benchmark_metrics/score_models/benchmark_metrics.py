

import json
import random
import time
import traceback
import numpy as np
import os
import torch
import argparse
import calendar


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAutoNAS')

    # job config
    parser.add_argument('--pre_trained_result', type=str, default="201_15k_c10_bs32_ic16_unionBest.json",
                        help="the latest metrics result to read")
    parser.add_argument('--metrics_result', type=str, default="201_15k_c10_bs32_ic16_unionBest.json",
                        help="the latest metrics result to save")

    # job config
    parser.add_argument('--num_arch', type=int, default=2, help="how many architecture to evaluate")

    # define base dir, where it stores apis, datasets, logs, etc,
    parser.add_argument('--base_dir', type=str, default="/Users/kevin/project_python/FIRMEST/data",
                        help='path of data folder')

    # dataLoader setting,
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='in one of [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--num_data_workers', type=int, default=1,
                        help='number of workers for dataLoader')
    parser.add_argument('--batch_sample_alg', type=str, default="random",
                        help='sample a mini batch, [random, grasp]')

    # define search space,
    parser.add_argument('--search_space', type=str, default="nasbench101",
                        help='which search space to use, [nasbench101, nasbench201, nds ]')

    parser.add_argument('--api_loc', type=str, default="nasbench_only108.pkl",
                        help='which search space to use, ['
                             'nasbench101: nasbench_only108.pkl'
                             'nasbench201: NAS-Bench-201-v1_0-e61699.pth')

    # define evaluation method
    parser.add_argument('--evaluation', type=str, default="all_matrix",
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

    parser.add_argument('--init_channels', default=128, type=int,
                        help='16 for 101, 201, 36 for 301-CIFAR, 48 for 301-IMGNET')
    # search space configs for nasBench101
    parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules')
    parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack')

    # search space configs for nasBench311
    parser.add_argument("--layers", type=int, default=20, help="20 for cifar, 14 for imgnet")
    parser.add_argument("--auxiliary", action="store_true", default=False, help="use auxiliary tower")
    parser.add_argument("--auxiliary_weight", type=float, default=0.4, help="weight for auxiliary loss")

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
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    os.environ.setdefault("log_file_name", args.metrics_result[:-5]+"_"+str(ts)+".log" )

    from logger import logger
    from common.constant import CommonVars
    from eva_engine.phase1.utils.gpu_util import showUtilization
    from storage import dataset
    from controller import sampler_register
    from eva_engine import evaluator_register
    import search_space

    random.seed(20)
    np.random.seed(20)
    torch.manual_seed(20)

    if args.bn == 1:
        bn = True
    else:
        bn = False

    if args.dataset == "cifar10":
        base_folder = "./result_append/CIFAR10_15625/union/"

    elif args.dataset == "cifar100":
        base_folder = "./result_append/CIFAR100_15625/union/"

    elif args.dataset == "ImageNet16-120":
        base_folder = "./result_append/IMAGENET/union/"
    else:
        raise f"dataset {args.dataset} not supported"

    logger.info("running with params: :" + json.dumps(args.__dict__, indent=2))
    logger.info("cuda available = " + str(torch.cuda.is_available()))

    # define dataLoader, and sample a mini-batch
    train_loader, val_loader, class_num = dataset.get_dataloader(
        32,
        test_batch_size=1,
        dataset=args.dataset,
        num_workers=args.num_data_workers,
        datadir=args.base_dir)
    args.num_labels = class_num

    if args.batch_size // class_num == 0:
        logger.info("batch_size is smaller than class_num " + str(args.batch_size) + " " + str(class_num) )
        # exit(0)

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
    sampler = sampler_register["sequence"](used_search_space, args)
    arch_generator = sampler.sample_next_arch(args.arch_size)

    # 2. Evaluator to evaluate one architecture
    evaluator_dict = {}
    if args.evaluation == CommonVars.ALL_EVALUATOR:
        evaluator_dict = evaluator_register
        # del evaluator_dict[CommonVars.PRUNE_SYNFLOW]
    else:
        evaluator_dict.update({args.evaluation: evaluator_register[args.evaluation]})

    # store result, in form of { arch_id: { scores: {}, queried_res: ... }}
    result = dict()

    if os.path.exists(base_folder+args.pre_trained_result):
        with open(base_folder+args.pre_trained_result, 'r') as readfile:
            result = json.load(readfile)

    i = 1
    try:
        while True:
            begin_run_time = time.time()
            if i > args.num_arch:
                logger.info("Finish Job")
                break

            arch_id, _ = arch_generator.__next__()
            if arch_id in result:
                i += 1
                continue

            # architecture = used_search_space.new_architecture(int(arch_id))
            logger.info("---" + "Evaluate arch with id = " + str(i) + ", archiD = " + str(arch_id) + " ---")
            print("---" + "Evaluate arch with id = " + str(i) + ", archiD = " + str(arch_id) + " ---")

            result[arch_id] = dict()

            # torch.autograd.set_detect_anomaly(True):
            for evaluator_name, evaluator in evaluator_dict.items():
                # time_arch_create = time.time()
                new_arch = used_search_space.copy_architecture(arch_id, _)
                # print("time for copy arch = ", time.time() - time_arch_create)
                try:
                    eva_time = time.time()
                    new_arch = new_arch.to(args.device)

                    # from thop import clever_format,profile
                    # flops, params = profile(new_arch, inputs=(mini_batch,))
                    # flops2, params2 = clever_format([flops, params], "%.3f")
                    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
                    # print('Params = ' + str(params / 1000 ** 2) + 'M')
                    # print("origin: ", f"arch_id={3380}, B={args.batch_size},C={args.init_channels}, flops2={flops2}，
                    # params2={params2}")
                    # exit(0)

                    score, time_usage = evaluator.evaluate_wrapper(
                        arch=new_arch,
                        device=args.device,
                        batch_data=mini_batch,
                        batch_labels=mini_batch_targets)
                    result[arch_id].update({evaluator_name: '{:f}'.format(score)})
                    logger.info(evaluator_name + ":" + str(score) + ", eval_time = " + str(time.time() - eva_time))
                    print(evaluator_name + ":" + str(score) + ", eval_time = " + str(time.time() - eva_time))
                except Exception as e:
                    if "out of memory" in str(e):
                        logger.info("======================================================")
                        logger.error("architecture " + str(arch_id) + " will be evaluate in CPU, message = " + str(e))
                        logger.info("======================================================")
                        mini_batch_cpu = mini_batch.cpu()
                        mini_batch_targets_cpu = mini_batch_targets.cpu()
                        eva_time = time.time()
                        new_arch_cpu = used_search_space.copy_architecture(arch_id, new_arch.cpu()).cpu()
                        score, cpu_time_usage = evaluator.evaluate_wrapper(
                            arch=new_arch_cpu,
                            device="cpu",
                            batch_data=mini_batch_cpu,
                            batch_labels=mini_batch_targets_cpu)
                        result[arch_id].update({evaluator_name: '{:f}'.format(score)})
                        logger.info(evaluator_name + " on cpu:" + str(score) + ", eval_time = " + str(time.time() - eva_time))
                        print(evaluator_name + " on cpu:" + str(score) + ", eval_time = " + str(time.time() - eva_time))
                        del new_arch_cpu
                    else:
                        if arch_id in result:
                            del result[arch_id]
                        logger.info("======================================================")
                        logger.error(traceback.format_exc())
                        logger.error("error when evaluate architecture " + str(arch_id) + ", message = " + str(e))
                        logger.info("======================================================")
                finally:
                    # clean old graph
                    del new_arch
                    torch.cuda.empty_cache()

            # del architecture
            # architecture = None
            torch.cuda.empty_cache()
            # 3. Query to query the real performance
            # result[arch_id].update(used_search_space.query_performance(arch_id, args.dataset))
            i = i + 1
            logger.info("time takes " + str(time.time() - begin_run_time))
            print("time takes " + str(time.time() - begin_run_time))

    except Exception as e:
        logger.info("================================================================================================")
        logger.error(traceback.format_exc())
        logger.error("error: " + str(e))
        logger.info("================================================================================================")

    # logger.info(json.dumps(result, indent=2))

    with open(base_folder+args.metrics_result, 'w') as outfile:
        outfile.write(json.dumps(result))
