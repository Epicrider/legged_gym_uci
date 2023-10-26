import os
import subprocess
import mlflow
import mlflow.pytorch
import numpy as np

protoType = {
        "task": "TASK1",
        "experiment_name": "EXPERIMENT1",
        "run_name": "RUN1",
        "seed": 1,
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
        
        
        "measure_heights": ""
        
    }

loadRange = np.array([0.5, 1, 1.5])
linkRange = np.array([0.1, 0.2, 0.3])



baseMassExperiments = []


for baseMass in [True, False]:
    if baseMass:
        for i in loadRange:
            
            
            for o in [True, False]:
                
                dct = {"randomize_base_mass": baseMass,
                "randomize_base_mass_range" : str(-1*i)+","+str(i),
                "randomize_base_mass_add_observation" : str(o)}
                
                
                baseMassExperiments.append(dct)
                
                #print(baseMass, -1*i, i, o)
                #print(dct)
                
        
    else:
        
        
        for o in [True, False]:
            dct = {"randomize_base_mass": baseMass,
                "randomize_base_mass_range" : "0,0",
                "randomize_base_mass_add_observation" : str(o)}
            
            #print(baseMass, -1*i, i, o)
            
            baseMassExperiments.append(dct)
            #print(dct)
        
    print("\n\n")




linkMassExperiments = []


for linkMass in [True, False]:
    if linkMass:
        for i in linkRange:
            
            
            for o in [True, False]:
                
                dct = {"randomize_link_mass": linkMass,
                "randomize_link_mass_range" : str(1 - i)+","+str(1 + i),
                "randomize_link_mass_add_observation" : str(o)}
                
                
                linkMassExperiments.append(dct)
                
                #print(baseMass, -1*i, i, o)
                #print(dct)
                
        
    else:
        
        
        for o in [True, False]:
            dct = {"randomize_link_mass": linkMass,
                "randomize_link_mass_range" : "1,1",
                "randomize_link_mass_add_observation" : str(o)}
            
            #print(baseMass, -1*i, i, o)
            
            linkMassExperiments.append(dct)
            #print(dct)
        
    print("\n\n")







breakJointsExperiments = []

for break_joints in [True, False]:
    if break_joints:
        for break_joints_add_observation in [True, False]:
            
            dct = {"break_joints": break_joints, "break_joints_add_observation": break_joints_add_observation}
            
            breakJointsExperiments.append(dct)
            
            #break_joints_add_observation
    else:
        dct = {"break_joints": break_joints, "break_joints_add_observation": False}
        
        breakJointsExperiments.append(dct)


# Define your experiments
experiments = []


for expList in baseMassExperiments + linkMassExperiments + breakJointsExperiments:
    
    protoType2 = protoType.copy()
    
    for key in expList:
        
        protoType2[key] = expList[key]
        #print(baseExp[key])
    
    
    protoType2["measure_heights"] = False
    experiments.append(protoType2)
    
    protoType3 = protoType2.copy()
    protoType3["measure_heights"] = True
    experiments.append(protoType3)







# Define the path to your train.py script
train_script_path = "train.py"

# Define the base directory where train.py saves the logs and models
base_log_dir = "/"








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
