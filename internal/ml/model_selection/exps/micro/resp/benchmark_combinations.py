from src.query_api.interface import SimulateScore
from src.eva_engine.phase2.evaluator import P2Evaluator
import random
from src.search_space.init_search_space import init_search_space
from exps.shared_args import parse_arguments


def random_combination(iterable, sample_size):
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), sample_size))
    return tuple(pool[i] for i in indices)


# multiple with move proposal
def mutate_spec_zero_cost(old_spec):
    possible_specs = []

    # each layer + each option => new model,  score it.
    for idx_to_change in range(len(old_spec)):
        entry_to_change = old_spec[idx_to_change]
        possible_entries = [x for x in range(5) if x != entry_to_change]
        for new_entry in possible_entries:
            new_spec = copy.copy(old_spec)
            new_spec[idx_to_change] = new_entry
            possible_specs.append((synflow_proxy[spec_to_idx[str(new_spec)]], new_spec))

    # get the large scored model
    best_new_spec = sorted(possible_specs, key=lambda i: i[0])[-1][1]
    if random.random() > 0.75:
        best_new_spec = random.choice(possible_specs)[1]
    return best_new_spec


def run_evolution_search(max_trained_models=1000,
                         pool_size=64,
                         tournament_size=10,
                         zero_cost_warmup=0,
                         zero_cost_move=False,
                         tfmem="express_flow",
                         dataset_name="frappe",
                         query_epoch=19,
                         args=None):
    best_valids, best_tests = [0.0], [0.0]
    pool = []  # (validation, spec) tuples
    num_trained_models = 0

    search_space_ins = init_search_space(args)

    _evaluator = P2Evaluator(search_space_ins, dataset_name,
                             is_simulate=True,
                             train_loader=None,
                             val_loader=None)

    # fill the initial pool
    if zero_cost_warmup > 0:
        zero_cost_pool = []
        for _ in range(zero_cost_warmup):
            arch_id, arch_micro = search_space_ins.random_architecture_id()

            score_getter = SimulateScore(
                space_name=search_space_ins.name,
                dataset_name=dataset_name)

            score = score_getter.query_all_tfmem_score(arch_id=arch_id)
            zero_cost_pool.append((score[tfmem], arch_id, arch_micro))
            zero_cost_pool = sorted(zero_cost_pool, key=lambda i: i[0], reverse=True)

    for i in range(pool_size):
        if zero_cost_warmup > 0:
            arch_id = zero_cost_pool[i][1]
            arch_micro = zero_cost_pool[i][2]
        else:
            raise "Use the framework to run our framework"

        full_train_auc, time_usage = _evaluator.p2_evaluate(str(arch_id), query_epoch)
        # get accuracy by training.
        num_trained_models += 1
        pool.append((full_train_auc, arch_micro))

        if full_train_auc > best_valids[-1]:
            best_valids.append(full_train_auc)
        else:
            best_valids.append(best_valids[-1])

        if full_train_auc > best_tests[-1]:
            best_tests.append(full_train_auc)
        else:
            best_tests.append(best_tests[-1])

    # After the pool is seeded, proceed with evolving the population.
    while True:

        # sample some value from the pool
        sample = random_combination(pool, tournament_size)
        best_arch_spec = sorted(sample, key=lambda i: i[0])[-1][2]
        if zero_cost_move:
            arch_id, arch_micro = mutate_spec_zero_cost(best_arch_spec)
        else:
            arch_id, arch_micro = search_space_ins.mutate_architecture(best_arch_spec)

        # train model to get auc
        full_train_auc, time_usage = _evaluator.p2_evaluate(str(arch_id), query_epoch)
        num_trained_models += 1

        # kill the oldest individual in the population.
        pool.append((full_train_auc, arch_micro))
        pool.pop(0)

        if full_train_auc > best_valids[-1]:
            best_valids.append(full_train_auc)
        else:
            best_valids.append(best_valids[-1])

        if full_train_auc > best_tests[-1]:
            best_tests.append(full_train_auc)
        else:
            best_tests.append(best_tests[-1])

        if num_trained_models >= max_trained_models:
            break

    best_tests.pop(0)
    best_valids.pop(0)
    return time_usage_lst, best_valids, best_tests


def filter_refinment_fully_training():
    pass


if __name__ == "__main__":
    args = parse_arguments()

    ae_warmup_time_usage_lst, ae_warmup_best_valids, ae_warmup_best_tests = run_evolution_search(
        max_trained_models=1000,
        pool_size=64,
        tournament_size=10,
        zero_cost_warmup=3000,
        zero_cost_move=False,
        tfmem="express_flow",
        dataset_name="frappe",
        query_epoch=19,
        args=args)

    ae_move_time_usage_lst, ae_move_best_valids, ae_move_best_tests = run_evolution_search(
        max_trained_models=1000,
        pool_size=64,
        tournament_size=10,
        zero_cost_warmup=0,
        zero_cost_move=True,
        tfmem="express_flow",
        dataset_name="frappe",
        query_epoch=19,
        args=args)
