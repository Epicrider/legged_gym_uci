
import os
import subprocess
#import mlflow
#import mlflow.pytorch
import torch
import numpy as np

import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import wandb

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def log_metrics(metrics):
    for i in range(0, len(list(metrics.items())[0][1])):
        step_dict = dict()
        for name in metrics.keys():
            step_dict[name] = metrics[name][i][1]
        wandb.log(step_dict, step=i)

def extract_tensorboard_metrics(log_dir):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    scalar_metrics = {}
    for tag in event_acc.Tags()["scalars"]:
        # This should create a list of tuples (step, value)
        scalar_metrics[tag] = [(x.step, x.value) for x in event_acc.Scalars(tag)]

    return scalar_metrics

def get_most_recent_folder(path):
    # List all entries in the given path
    all_entries = os.listdir(path)
    
    # Filter out only directories
    directories = [d for d in all_entries if os.path.isdir(os.path.join(path, d))]
    
    # If there are no directories, return None
    if not directories:
        return None
    
    # Get the most recent directory
    most_recent_dir = max(directories, key=lambda d: os.path.getmtime(os.path.join(path, d)))
    
    return os.path.join(path, most_recent_dir)


EXPERIMENT_NAME = "MLFlow_Test_4"
MAX_ITERATIONS = 1500

loadRange = np.array([0.5, 1, 1.5])
linkRange = np.array([0.1, 0.2, 0.3])


protoType = {
        "task": "a1_flat",
        "experiment_name": EXPERIMENT_NAME,
        "run_name": "",
        "seed": "1",
        "actor_hidden_dims": "512,256,128",
        "critic_hidden_dims": "512,256,128",
        
        
        "randomize_base_mass": "",
        "randomize_base_mass_range" : "",
        "randomize_base_mass_add_observation" : "",
        
        
        "randomize_link_mass" : "",
        "randomize_link_mass_range" : "",
        "randomize_link_mass_add_observation" : "",
        
        
        "break_joints" : "",
        "break_joints_add_observation" : "",
        
        
        "measure_heights": True, 
    
    
        "max_iterations" : MAX_ITERATIONS
        
    }


baseMassExperiments = []
for baseMass in [True, False]:
    if baseMass:
        for i in loadRange:
            for o in [True, False]:
                
                dct = {"randomize_base_mass": baseMass,
                "randomize_base_mass_range" : str(i),
                "randomize_base_mass_add_observation" : o}
                baseMassExperiments.append(dct)
    else:
        dct = {"randomize_base_mass": baseMass,
                "randomize_base_mass_range" : "0",
                "randomize_base_mass_add_observation" : False}
        baseMassExperiments.append(dct)
        
    print("\n\n")


linkMassExperiments = []


for linkMass in [True, False]:
    if linkMass:
        for i in linkRange:
            for o in [True, False]:
                dct = {"randomize_link_mass": linkMass,
                "randomize_link_mass_range" : str(1 - i)+","+str(1 + i),
                "randomize_link_mass_add_observation" : o}                
                linkMassExperiments.append(dct)        
    else:
        dct = {"randomize_link_mass": linkMass,
                "randomize_link_mass_range" : "1,1",
                "randomize_link_mass_add_observation" : False}
        linkMassExperiments.append(dct)
        
    print("\n\n")


breakJointsExperiments = []
for break_joints in [True, False]:
    if break_joints:
        for break_joints_add_observation in [True, False]:
            dct = {"break_joints": break_joints, "break_joints_add_observation": break_joints_add_observation}            
            breakJointsExperiments.append(dct)

    else:
        dct = {"break_joints": break_joints, "break_joints_add_observation": False}
        breakJointsExperiments.append(dct)



# Define your experiments
experiments = []

ctr = 0
for expList in baseMassExperiments + linkMassExperiments + breakJointsExperiments:    
    protoType2 = protoType.copy()
    
    for key in expList:
        protoType2[key] = expList[key]

    protoType2["run_name"] = "Run_"+str(ctr)    
    experiments.append(protoType2)

    ctr+=1

print("Experiment Example", experiments[0])

# Define the path to your train.py script
train_script_path = "train.py"
master_log_dir = f'{LEGGED_GYM_ROOT_DIR}'
print("Master Log Directory:", master_log_dir)

# Define the base directory where train.py saves the logs and models
base_log_dir = os.path.join(master_log_dir, "logs")

# Replace the MLflow logging in the experiments loop with wandb logging
for exp in experiments:
    # Initialize a new wandb run
    with wandb.init(project=EXPERIMENT_NAME, config=exp):
        # wandb.config is automatically populated with the values from exp
        
        # Build the command to run train.py
        command = ["python3", train_script_path]
        for param_name, param_value in exp.items():
            if param_value != False and param_value != '':
                command.append(f"--{param_name}")
                if param_value!= True:
                    command.append(str(param_value))
        
        command.append("--headless")

        # Run train.py
        subprocess.run(command)
        
        experiment_name = exp["experiment_name"]
        log_dir = get_most_recent_folder(os.path.join(base_log_dir, experiment_name))
        metrics = extract_tensorboard_metrics(log_dir)
        log_metrics(metrics)
        