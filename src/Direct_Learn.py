
import os
import json
import pandas as pd
from dataclasses import  asdict

from OCS import AppointmentScheduler
from Learn_Module import IO_learning
from NN import SupervisedNN
from utility import parse_args, build_config, setup_logging, Config


def main():
    args = parse_args()
    config = build_config(args)

    # Folder structure
    components = [
        f"iter{config.learn_iter}",
        f"runs{config.runs}",
        f"um{config.update_model}",
        f"new{config.new}",
        f"stoch{config.stoch}",
        f"sc{config.scenarios}",
        f"van{config.vanilla}",
        f"gat{config.gated}",
        f"tl{config.time_limit}",
        f"nc{config.num_class}"
    ]
    components2 = [
        f"iter{config.learn_iter}",
        "runs4000",
        f"um{config.update_model}",
        f"new{config.new}",
        f"stoch{config.stoch}",
        f"sc{config.scenarios}",
        f"van{config.vanilla}",
        f"gat{config.gated}",
        f"tl{config.time_limit}",
        f"nc{config.num_class}"
    ]
    name = "_".join(components)
    folder = f"{config.policy}_{name}/"
    train_folder = f"../data/Data_Dagger_{config.lambda_}_{config.adapter}/Instances_N{config.ocs_param['N']}_s{config.ocs_param['sigma']}_P{config.ocs_param['P']}_I{config.ocs_param['I']}_L{config.ocs_param['L']}_{config.ocs_param['k_min']}_{config.ocs_param['k_max']}_{config.ocs_param['physician_weights']}_{config.ocs_param['c_miss_1']}_{config.ocs_param['c_miss_2']}_{config.ocs_param['co']}_{config.ocs_param['cnp']}"
    output_dir = os.path.join(train_folder, folder)
    os.makedirs(output_dir, exist_ok=True)
    name2 = "_".join(components2)
    data_folder = os.path.join(train_folder, f"{config.policy}_{name2}/")
    #data_folder = f"../data/Deterministic/Instances_N{config.ocs_param['N']}_s{config.ocs_param['sigma']}_P{config.ocs_param['P']}_I{config.ocs_param['I']}_L{config.ocs_param['L']}_{config.ocs_param['k_min']}_{config.ocs_param['k_max']}_{config.ocs_param['physician_weights']}_{config.ocs_param['c_miss_1']}_{config.ocs_param['c_miss_2']}_{config.ocs_param['co']}_{config.ocs_param['cnp']}/instances_solution"
    
    # Logging & seed
    setup_logging(output_dir)

    # Save config
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(asdict(config), f, indent=4)

    # Initialise simulators & learners
    sim = AppointmentScheduler(
        n_patients=config.ocs_param['N'],
        sigma = config.ocs_param['sigma'],
        n_phys=config.ocs_param['P'],
        n_slots=config.ocs_param['I'],
        # --- costs wired to your class fields used in build_model ---
        c_miss_1=config.ocs_param['c_miss_1'],
        c_miss_2=config.ocs_param['c_miss_2'],
        c_overtime=config.ocs_param['co'],
        c_np_per=config.ocs_param['cnp'],
        release_cap=config.ocs_param['Rmax'],
        cap_per_phys=config.ocs_param['L'],
        time_limit=config.time_limit,
        seed=config.seed,
        out_dir=output_dir,
        suffix=False
    )
    
    learner = IO_learning(sim, asdict(config), folder=output_dir) 
    
    if config.scenarios > 1 and config.stoch == 0:
        config.target_mode = "soft"
        config.target = "expert_distribution"
        print("Learning soft distribution")
    else:
        config.target_mode = "hard"
        config.target = "expert_action"
    
    policy = SupervisedNN( learner.feature_names, asdict(config), output_dir)
    
    #csv_path = os.path.join(data_folder, "offline_features_data.csv")
    csv_path = os.path.join(data_folder, "All_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find: {csv_path}")
    #read the data file
    df = pd.read_csv(csv_path)
    df = df[df.run < config.runs ]

    if config.num_class == 2:
        df["expert_action"] = df["label"] > 0
    else:
        df["expert_action"] = df["label"]
    policy.update_model(df, output_dir, 0, save_data=True)
    policy.feature_importance(
    df=df,
    repeats=5,
    metric="accuracy",        # or "loss" (returns negative CE)
    exclude_prefixes=("mask_",),  # keep masks out by default
    save_to=output_dir + "feature_importance.csv"
    )

if __name__ == '__main__':
    main()
