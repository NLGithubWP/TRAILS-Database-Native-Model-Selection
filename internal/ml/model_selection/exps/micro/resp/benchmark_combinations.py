from exps.shared_args import parse_arguments
import random
import calendar
import os
import time


def random_combination(iterable, sample_size):
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), sample_size))
    return tuple(pool[i] for i in indices)


def run_evolution_search(max_trained_models=1000,
                         pool_size=64,
                         tournament_size=10,
                         zero_cost_warmup=0,
                         zero_cost_move=False,
                         tfmem="express_flow",
                         dataset_name="frappe",
                         query_epoch=19,
                         args=None):
    best_tests = [0.0]
    pool = []  # (validation, spec) tuples
    time_usage_lst = [0]
    num_trained_models = 0

    search_space_ins = init_search_space(args)

    _evaluator = P2Evaluator(search_space_ins, dataset_name,
                             is_simulate=True,
                             train_loader=None,
                             val_loader=None)

    score_getter = SimulateScore(
        space_name=search_space_ins.name,
        dataset_name=dataset_name)

    # fill the initial pool
    cur_time_used = 0
    if zero_cost_warmup > 0:
        zero_cost_pool = []
        for _ in range(zero_cost_warmup):
            arch_id, arch_micro = search_space_ins.random_architecture_id()

            score = score_getter.query_all_tfmem_score(arch_id=arch_id)
            zero_cost_pool.append((score[tfmem], arch_id, arch_micro))
            zero_cost_pool = sorted(zero_cost_pool, key=lambda i: i[0], reverse=True)

        # already score 3k models, here
        cur_time_used += zero_cost_warmup * score_getter.api.get_score_one_model_time("gpu")

    for i in range(pool_size):
        if zero_cost_warmup > 0:
            arch_id = zero_cost_pool[i][1]
            arch_micro = zero_cost_pool[i][2]
        else:
            arch_id, arch_micro = search_space_ins.random_architecture_id()

        full_train_auc, time_usage = _evaluator.p2_evaluate(str(arch_id), query_epoch)
        # get accuracy by training.
        num_trained_models += 1
        pool.append((full_train_auc, arch_micro))

        cur_time_used += time_usage
        time_usage_lst.append(cur_time_used)

        if full_train_auc > best_tests[-1]:
            best_tests.append(full_train_auc)
        else:
            best_tests.append(best_tests[-1])

    # After the pool is seeded, proceed with evolving the population.
    while True:

        # sample some value from the pool
        sample = random_combination(pool, tournament_size)
        best_arch_spec = sorted(sample, key=lambda i: i[0])[-1][1]
        if zero_cost_move:

            _all_combs = search_space_ins.mutate_architecture_move_proposal(best_arch_spec)
            all_combs = []
            for ele in _all_combs:
                arch_id = ele[0]
                arch_micro = ele[1]
                score = score_getter.query_all_tfmem_score(arch_id=arch_id)
                all_combs.append((score[tfmem], arch_id, arch_micro))

            if random.random() > 0.75:
                best_arch_spec = random.choice(all_combs)
            else:
                best_arch_spec = sorted(list(all_combs), key=lambda i: i[0])[-1]

            _, arch_id, arch_micro = best_arch_spec

            cur_time_used += len(_all_combs) * score_getter.api.get_score_one_model_time("gpu")

        else:
            arch_id, arch_micro = search_space_ins.mutate_architecture(best_arch_spec)

        # train model to get auc
        full_train_auc, time_usage = _evaluator.p2_evaluate(str(arch_id), query_epoch)
        num_trained_models += 1

        cur_time_used += time_usage
        time_usage_lst.append(cur_time_used)

        # kill the oldest individual in the population.
        pool.append((full_train_auc, arch_micro))
        pool.pop(0)

        if full_train_auc > best_tests[-1]:
            best_tests.append(full_train_auc)
        else:
            best_tests.append(best_tests[-1])

        if num_trained_models >= max_trained_models:
            break

    best_tests.pop(0)
    return time_usage_lst, best_tests


def filter_refinment_fully_training():
    pass


if __name__ == "__main__":
    args = parse_arguments()

    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    os.environ.setdefault("log_logger_folder_name", f"{args.log_folder}")
    os.environ.setdefault("log_file_name", args.log_name + "_" + str(ts) + ".log")
    os.environ.setdefault("base_dir", args.base_dir)
    from src.query_api.interface import SimulateScore
    from src.eva_engine.phase2.evaluator import P2Evaluator
    from src.search_space.init_search_space import init_search_space

    ae_warmup_time_usage_lst, ae_warmup_best_tests = run_evolution_search(
        max_trained_models=1000,
        pool_size=64,
        tournament_size=10,
        zero_cost_warmup=3000,
        zero_cost_move=False,
        tfmem="express_flow",
        dataset_name="frappe",
        query_epoch=19,
        args=args)

    ae_move_time_usage_lst, ae_move_best_tests = run_evolution_search(
        max_trained_models=1000,
        pool_size=64,
        tournament_size=10,
        zero_cost_warmup=0,
        zero_cost_move=True,
        tfmem="express_flow",
        dataset_name="frappe",
        query_epoch=19,
        args=args)

    print("Done")
