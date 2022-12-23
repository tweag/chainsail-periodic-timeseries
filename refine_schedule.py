import argparse
import pickle
import os

import numpy as np
import yaml

parser = argparse.ArgumentParser(
    description="Refine schedule in case of acceptance rate bottlenecks and set up rexfw simulation")
parser.add_argument(
    "simulation_run",
    type=str,
    help="Simulation run the schedule of which to refine")
parser.add_argument(
    "num_extra_replicas",
    type=int,
    help="Number of additional replicas to add between bottleneck temperatures")
parser.add_argument(
    "output_directory",
    type=str,
    help="Directory in which to write refined schedule and everything that's needed for a simulation using it")

args = parser.parse_args()

old_schedule = np.load(
    os.path.join(args.simulation_run, "schedule.pickle"),
    allow_pickle=True
)
acceptance_rates = np.loadtxt(
    os.path.join(args.simulation_run, "statistics", "re_pacc.txt")
)[:, 1:]
old_timesteps = np.load(
    os.path.join(args.simulation_run, "final_stepsizes.pickle"),
    allow_pickle=True
)
with open(os.path.join(args.simulation_run, "config.yml")) as f:
    old_config = yaml.safe_load(f)

replica_pairs = [(i, i + 1) for i in range(len(old_schedule))]
bottleneck_pair = replica_pairs[np.argmin(acceptance_rates[-1])]
bottleneck_lower_beta, bottleneck_upper_beta = old_schedule.take(bottleneck_pair)
bottleneck_interpolation = np.linspace(
    bottleneck_lower_beta,
    bottleneck_upper_beta,
    args.num_extra_replicas - 1,
    endpoint=False)
new_schedule = np.concatenate(
    (
        old_schedule[:bottleneck_pair[0]],
        bottleneck_interpolation,
        old_schedule[bottleneck_pair[1]:]
    )
)
new_schedule_path = os.path.join(args.output_directory, "schedule.pickle")
with open(new_schedule_path, "wb") as f:
    pickle.dump({"beta": new_schedule}, f)

bottleneck_lower_timestep, bottleneck_upper_timestep = old_timesteps.take(bottleneck_pair)
timestep_interpolation = np.linspace(
    bottleneck_lower_timestep,
    bottleneck_upper_timestep,
    args.num_extra_replicas - 1,
    endpoint=False)
new_timesteps = np.concatenate(
    (
        old_timesteps[:bottleneck_pair[0]],
        timestep_interpolation,
        old_timesteps[bottleneck_pair[1]:]
    )
)
new_timesteps_path = os.path.join(args.output_directory, "initial_stepsizes.pickle")
with open(new_timesteps_path, "wb") as f:
    pickle.dump(new_timesteps, f)

new_config = old_config.copy()
split_outputdir = os.path.split(args.output_directory)
new_config["general"].update(
    basename=os.path.join(*split_outputdir[:-1]),
    num_replicas=len(new_schedule),
    output_path=split_outputdir[-1]
)
new_config["local_sampling"].update(
    stepsizes=new_timesteps_path
)
new_config["re"].update(
    schedule="schedule.pickle"
)
new_config_path = os.path.join(args.output_directory, "config.yml")
with open(new_config_path, "w") as f:
    yaml.dump(new_config, f)

start_script = f'''
#!/usr/bin/env bash

mpirun \
  --oversubscribe \
  -n {len(new_schedule) + 1} \
  --storage {storage_config} \
  --basename {basename} \
  --path {split_outputdir[-1]} \
  --name refined_{len(new_schedule)}replicas \
  
'''
