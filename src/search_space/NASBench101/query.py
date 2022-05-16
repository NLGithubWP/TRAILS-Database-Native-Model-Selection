import json

import numpy as np
from nasbench import api


class Query101:

    def __init__(self):
        pass

    def query_result(self, query_apis, adjacency_matrix, node_list) -> (str, list):
        nasbench = query_apis["nb101_data"]
        model_spec = api.ModelSpec(matrix=adjacency_matrix, ops=node_list)
        res = nasbench.query(model_spec)
        static = []
        static.append(res["training_time"])
        static.append(res["train_accuracy"])
        static.append(res["validation_accuracy"])
        static.append(res["test_accuracy"])

        return str(res), static


class NasbenchWrapper(api.NASBench):
    """Small modification to the NASBench class, to return all three architecture evaluations at
    the same time, instead of samples."""

    def query(self, model_spec, epochs=108, stop_halfway=False):
        """Fetch one of the evaluations for this model spec.

        Each call will sample one of the config['num_repeats'] evaluations of the
        model. This means that repeated queries of the same model (or isomorphic
        models) may return identical metrics.

        This function will increment the budget counters for benchmarking purposes.
        See self.training_time_spent, and self.total_epochs_spent.

        This function also allows querying the evaluation metrics at the halfway
        point of training using stop_halfway. Using this option will increment the
        budget counters only up to the halfway point.

        Args:
          model_spec: ModelSpec object.
          epochs: number of epochs trained. Must be one of the evaluated number of
            epochs, [4, 12, 36, 108] for the full dataset.
          stop_halfway: if True, returned dict will only contain the training time
            and accuracies at the halfway point of training (num_epochs/2).
            Otherwise, returns the time and accuracies at the end of training
            (num_epochs).

        Returns:
          dict containing the evaluated darts for this object.

        Raises:
          OutOfDomainError: if model_spec or num_epochs is outside the search space.
        """
        if epochs not in self.valid_epochs:
            raise api.OutOfDomainError('invalid number of epochs, must be one of %s'
                                       % self.valid_epochs)

        fixed_stat, computed_stat = self.get_metrics_from_spec(model_spec)
        trainings = []
        for index in range(self.config['num_repeats']):
            computed_stat_at_epoch = computed_stat[epochs][index]

            data = {}
            data['module_adjacency'] = fixed_stat['module_adjacency']
            data['module_operations'] = fixed_stat['module_operations']
            data['trainable_parameters'] = fixed_stat['trainable_parameters']

            if stop_halfway:
                data['training_time'] = computed_stat_at_epoch['halfway_training_time']
                data['train_accuracy'] = computed_stat_at_epoch['halfway_train_accuracy']
                data['validation_accuracy'] = computed_stat_at_epoch['halfway_validation_accuracy']
                data['test_accuracy'] = computed_stat_at_epoch['halfway_test_accuracy']
            else:
                data['training_time'] = computed_stat_at_epoch['final_training_time']
                data['train_accuracy'] = computed_stat_at_epoch['final_train_accuracy']
                data['validation_accuracy'] = computed_stat_at_epoch['final_validation_accuracy']
                data['test_accuracy'] = computed_stat_at_epoch['final_test_accuracy']

            self.training_time_spent += data['training_time']
            if stop_halfway:
                self.total_epochs_spent += epochs // 2
            else:
                self.total_epochs_spent += epochs
            trainings.append(data)

        return trainings