import os
import random
import time
import warnings
import torch
import numpy as np
import openml
from sklearn.model_selection import train_test_split
import sklearn
from typing import Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# this corresponds to the number of threads
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_data(
        task_id: int,
        val_share: float = 0.25,
        test_size: float = 0.2,
        seed: int = 11):
    task = openml.tasks.get_task(task_id=task_id)
    dataset = task.get_dataset()
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format='dataframe',
        target=dataset.default_target_attribute)

    # Automatically identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['uint8', 'uint32', 'uint64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    print("dataset = ", X.shape, "column = ", X.shape[1])

    if numerical_cols.shape[0] + categorical_cols.shape[0] != X.shape[1]:
        print("Errored: Not all columns are detected")
        exit(0)

    # if task_id in [3, 31, 3945, 14965]:
    #     encoder = OneHotEncoder()
    #     X = encoder.fit_transform(X)

    if isinstance(y[1], bool):
        y = y.astype('bool')

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
        shuffle=True,
    )
    resampling_strategy_args = {'val_share': val_share}
    return X_train, X_test, y_train, y_test, resampling_strategy_args, categorical_indicator, numerical_cols, categorical_cols


# dilbert
task_id = 168909

# credit
# task_id = 31

# kc1
# task_id = 3917

# blood , "512-256-256-512" 30 iteration, 68.640
# task_id = 10101

# bank
# task_id = 14965, 70.9250094399

# christine, 512-384-512-256, 500 iteration, 0.724169741697417,
# task_id = 168908

# sylvine, 384-256-128-256, 500 iteration, 94.046
# task_id = 168912

# fabert, 256-384-512-512 500 iteration, 64.651
# task_id = 168910

# car 384-256-256-128, 500 iteration, 100.000
# task_id = 146821

# australian, "384-256-256-512", 15 iteration, 89.376
# task_id = 146818


seed = 11

if __name__ == '__main__':
    # Setting up reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    number_of_configurations_limit = 0

    ############################################################################
    # Data Loading
    # ============
    start_time = time.time()

    X_train, X_test, y_train, y_test, resampling_strategy_args, \
        categorical_indicator, numerical_cols, categorical_cols = get_data(
            task_id=task_id,
            seed=seed)

    # Building and training the MLP model
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    model = "512-384-512-256"
    corrected_tuple_value = tuple(int(num) for num in model.split('-'))
    print(corrected_tuple_value)
    mlp = MLPClassifier(hidden_layer_sizes=corrected_tuple_value,
                        max_iter=500,
                        activation='relu',
                        solver='adam',
                        random_state=60)

    # Define transformers for numerical and categorical columns
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    # Combine transformers in ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Create a pipeline with a scaler and the MLPClassifier
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('mlp', mlp)  # MLP model
    ])

    # Use the pipeline for training
    pipeline.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = pipeline.predict(X_test)

    test_balanced_accuracy = \
        sklearn.metrics.balanced_accuracy_score(y_test, y_pred.squeeze())

    print(f'Accuracy: {test_balanced_accuracy}')
