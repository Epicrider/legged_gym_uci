import os
import subprocess
import mlflow
import mlflow.pytorch
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR


import numpy as np

import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_tensorboard_metrics(log_dir):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()  # Load the TensorBoard events

    # Fetch the scalar metrics
    scalar_metrics = {}
    for tag in event_acc.Tags()["scalars"]:
        scalar_metrics[tag] = [x.value for x in event_acc.Scalars(tag)]
    
    return scalar_metrics


print( extract_tensorboard_metrics('../../mlFlowLogs/MLFlow_Test/0/4a19230080624c7fab0ed36c408a804b/artifacts/tensorboard_logs/') )
