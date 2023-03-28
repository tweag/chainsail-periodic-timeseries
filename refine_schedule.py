import argparse
from pathlib import Path
import pickle
import os
from typing import TypeAlias

import numpy as np
import yaml

CHAINSAIL_ROOT = os.environ["CHAINSAIL_ROOT"]

Schedule: TypeAlias = dict[str, np.ndarray]
ReplicaPair: TypeAlias = tuple[int, int]

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
    "output_dirname",
    type=str,
    help="Directory in which to write refined schedule simulations")

def find_bottleneck_replica_pair(old_schedule: Schedule) -> ReplicaPair:
    betas = old_schedule["beta"]
    acceptance_rates = np.loadtxt(
        simulation_run / Path(os.path.join("statistics", "re_stats.txt"))
    )[:, 1:]
    replica_pairs = [(i, i + 1) for i in range(len(betas))]
    return replica_pairs[np.argmin(acceptance_rates[-1])]

def refine_schedule(old_schedule: Schedule, bottleneck_pair: ReplicaPair, num_extra_replicas: int) -> Schedule:
    betas = old_schedule["beta"]
    bottleneck_lower_index, bottleneck_upper_index = bottleneck_pair
    bottleneck_lower_beta, bottleneck_upper_beta = betas.take(bottleneck_pair)
    bottleneck_interpolation = np.linspace(
        bottleneck_lower_beta,
        bottleneck_upper_beta,
        num_extra_replicas - 1,
        endpoint=False)
    new_schedule = np.concatenate(
        (
            betas[:bottleneck_lower_index],
            bottleneck_interpolation,
            betas[bottleneck_upper_index:]
        )
    )
    return {"beta": new_schedule}


def interpolate_stepsizes(simulation_run: Path, bottleneck_pair: ReplicaPair) -> np.ndarray:
    old_timesteps = np.load(
        os.path.join(simulation_run, "final_stepsizes.pickle"),
        allow_pickle=True
    )
    bottleneck_lower_timestep, bottleneck_upper_timestep = old_timesteps.take(bottleneck_pair)
    timestep_interpolation = np.linspace(
        bottleneck_lower_timestep,
    bottleneck_upper_timestep,
        args.num_extra_replicas - 1,
        endpoint=False)
    return np.concatenate(
        (
            old_timesteps[:bottleneck_pair[0]],
            timestep_interpolation,
            old_timesteps[bottleneck_pair[1]:]
        )
    )

def make_new_initial_states(old_config: dict, new_num_replicas: int, bottleneck_replica_pair: ReplicaPair) -> np.ndarray:
    dump_interval = old_config["re"]["dump_interval"]
    general = old_config["general"]
    num_samples = general["n_iterations"]
    old_num_replicas = general["num_replicas"]
    old_dirname = general["dirname"]
    old_output_path = general["output_path"]
    last_batch_filenames = [f"samples_replica{i}_{num_samples-dump_interval}-{num_samples}.pickle"
                            for i in range(1, old_num_replicas + 1)]
    old_final_states = np.array([np.load(os.path.join(old_dirname, old_output_path, "samples", fn), allow_pickle=True)[-1] for fn in last_batch_filenames])
    _, bottleneck_upper_index = bottleneck_replica_pair
    repeats = np.ones(old_num_replicas).astype(int)
    repeats[bottleneck_upper_index] = new_num_replicas - old_num_replicas + 1

    return np.repeat(old_final_states, repeats, axis=0)


def make_new_config(old_config: dict, new_num_replicas: int, new_simulation_path: Path, timestep_filename: Path = Path("initial_stepsizes.pickle"), schedule_filename: Path = Path("schedule.pickle"), initial_states_filename: Path = Path("initial_states.pickle")) -> dict:
    new_config = old_config.copy()
    new_config["general"].update(
        dirname=str(new_simulation_path.parent),
        num_replicas=new_num_replicas,
        output_path=new_simulation_path.name,
        initial_states=str(initial_states_filename)
    )
    new_config["local_sampling"].update(
        stepsizes=str(timestep_filename)
    )
    new_config["re"].update(
    schedule=str(schedule_filename)
    )

    return new_config

def write_storage_config(storage_config_path: Path):
    with open(storage_config_path, "w") as f:
        yaml.dump({"backend": "local", "backend_config": {"local": {}}}, f)


def write_launch_script(new_config: dict, storage_config_path: Path, new_simulation_path: Path):
    start_script = f'''#!/usr/bin/env bash

    set -uexo

    mpirun \\
      --oversubscribe \\
      -n {new_config["general"]["num_replicas"] + 1} \\
      python {os.path.join(CHAINSAIL_ROOT, "lib", "runners", "rexfw", "chainsail", "runners", "rexfw", "mpi.py")} \\
      --storage {storage_config_path} \\
      --dirname {new_config["general"]["dirname"]} \\
      --path {new_config["general"]["output_path"]} \\
      --name job-1.refined_{new_config["general"]["num_replicas"]}replicas \\
      --metrics-host 127.0.0.1 \\
      --metrics-port 1234 \\
      --user-code-host 127.0.0.1 \\
      --user-code-port 50001
    '''
    start_script_path = os.path.join(new_simulation_path, "launch_refined_simulation.sh")
    with open(start_script_path, "w") as f:
        f.write(start_script)


if __name__ == "__main__":
    args = parser.parse_args()
    simulation_run = Path(args.simulation_run)

    old_schedule = np.load(
        simulation_run / "schedule.pickle",
        allow_pickle=True
    )
    with open(simulation_run / "config.yml") as f:
        old_config = yaml.safe_load(f)

    bottleneck_replica_pair = find_bottleneck_replica_pair(old_schedule)

    new_schedule = refine_schedule(old_schedule, bottleneck_replica_pair, args.num_extra_replicas)
    new_num_replicas = len(new_schedule["beta"])
    new_simulation_path = Path(os.path.join(args.output_dirname, f"refined_{new_num_replicas}replicas"))
    os.makedirs(new_simulation_path, exist_ok=True)
    with open(new_simulation_path / Path("schedule.pickle"), "wb") as f:
        pickle.dump(new_schedule, f)

    new_stepsizes = interpolate_stepsizes(simulation_run, bottleneck_replica_pair)
    stepsizes_filename = Path("initial_stepsizes.pickle")
    with open(new_simulation_path / stepsizes_filename, "wb") as f:
        pickle.dump(new_stepsizes, f)

    new_initial_states = make_new_initial_states(old_config, new_num_replicas, bottleneck_replica_pair)
    initial_states_filename = Path("initial_states.pickle")
    with open(new_simulation_path / initial_states_filename, "wb") as f:
        pickle.dump(new_initial_states, f)

    new_config = make_new_config(old_config, new_num_replicas, new_simulation_path)
    with open(new_simulation_path / Path("config.yml"), "w") as f:
        yaml.dump(new_config, f)

    storage_config_path = new_simulation_path / Path("storage.yml")
    write_storage_config(storage_config_path)

    write_launch_script(new_config, storage_config_path, new_simulation_path)
