import os
import subprocess
import mlflow
import mlflow.pytorch

# Define your experiments
experiments = [
    {
        "task": "TASK1",
        "experiment_name": "EXPERIMENT1",
        "run_name": "RUN1",
        "num_envs": 1,
        "seed": 123,
        "max_iterations": 1000,
    },
    {
        "task": "TASK2",
        "experiment_name": "EXPERIMENT2",
        "run_name": "RUN2",
        "num_envs": 2,
        "seed": 456,
        "max_iterations": 2000,
    },
    # ... add more experiments as needed
]

# Define the path to your train.py script
train_script_path = "/path/to/train.py"

# Define the base directory where train.py saves the logs and models
base_log_dir = "/path/to/issacgym_anymal/logs"

# Run each experiment
for exp in experiments:
    with mlflow.start_run():
        
        # Log parameters
        for param_name, param_value in exp.items():
            mlflow.log_param(param_name, param_value)
        
        # Build the command to run train.py
        command = ["python", train_script_path]
        for param_name, param_value in exp.items():
            command.append(f"--{param_name}")
            command.append(str(param_value))
        
        # Run train.py
        subprocess.run(command)
        
        # Log the model and TensorBoard logs
        experiment_name = exp["experiment_name"]
        run_name = exp["run_name"]
        log_dir = os.path.join(base_log_dir, experiment_name, run_name)
        model_path = os.path.join(log_dir, "model_<iteration>.pt")  # modify as needed
        tensorboard_log_dir = os.path.join(log_dir, "tensorboard_logs")  # modify as needed
        
        mlflow.pytorch.log_model(model_path, "model")
        mlflow.log_artifacts(tensorboard_log_dir, "tensorboard_logs")
