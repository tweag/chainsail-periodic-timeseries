import argparse
import pickle
import os
import sys

from dataclasses import dataclass, field
from tempfile import TemporaryDirectory
from typing import Callable

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from chainsail_helpers.scripts.concatenate_samples import main as concatenate_samples

parser = argparse.ArgumentParser(
    description="Plot Chainsail sampling results for periodic timeseries model"
)
parser.add_argument(
    "simulation_run",
    type=str,
    help='Simulation run, e.g. "/some/path/to/production_run"'
)
args = parser.parse_args()

results_dir = args.simulation_run
with TemporaryDirectory() as tmpdir:
    tmpfile = os.path.join(tmpdir, "samples.npy")
    sys.argv.append(tmpfile)
    concatenate_samples()
    samples = np.load(tmpfile)
    
re_acceptance_rates = np.loadtxt(os.path.join(results_dir, "statistics", "re_stats.txt"))
# dictionary with # of MCMC samples as keys and an array of acceptance rates (between replica 1 and 2, 2 and 3, ...)
# as values
re_acceptance_rates = dict(zip(re_acceptance_rates[:,0].astype(int), re_acceptance_rates[:,1:]))
single_replica_stats = np.loadtxt(os.path.join(results_dir, "statistics", "mcmc_stats.txt"))
# dictionary with # of MCMC samples as keys and single-replica (sr) HMC / local sampling acceptance rates as values
sr_paccs = dict(zip(single_replica_stats[:,0].astype(int), single_replica_stats[:,1::2]))
# same for stepsizes
sr_stepsizes = dict(zip(single_replica_stats[:,0].astype(int), single_replica_stats[:,2::2]))

with open(os.path.join(results_dir, "schedule.pickle"), "rb") as f:
    schedule = pickle.load(f)

burnin = 100
samples = samples[burnin:]

@dataclass
class Parameter:
    name: str
    extract_and_transform: Callable[np.ndarray, np.ndarray]
    unit: str | None = None
    num_bins: int = 75
    label: str = field(init=False)
    transformed_values: np.ndarray = field(init=False)

    def __post_init__(self):
        unit_string = f" [{self.unit}]" if self.unit else ""
        self.label = f"{self.name}{unit_string}"
        self.transformed_values = self.extract_and_transform(samples)


parameters = (
    Parameter(name='noise', extract_and_transform=lambda samples: np.exp(samples[:,0])),
    Parameter(name='frequency', unit='Hz', extract_and_transform=lambda samples: np.exp(samples[:,1])),
    Parameter(name='phase', unit='rad', extract_and_transform=lambda samples: np.arctan2(samples[:,2], samples[:,3])),
    Parameter(name='amplitude', extract_and_transform=lambda samples: np.sqrt(np.sum(samples[:,2:] ** 2, 1)))
)
parameter_dict = {p.name: p for p in parameters}


def clean_hist_axis(ax):
    for spine in ('top', 'left', 'right'):
        ax.spines[spine].set_visible(False)
    ax.set_yticks(())


def clean_trace_axis(ax):
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    

fig, axes = plt.subplots(7, 2, figsize=(10, 14))
axes_pair_iter = iter(axes)

for param in parameters:
    hist_ax, trace_ax = next(axes_pair_iter)
    hist_ax.hist(param.transformed_values, bins=param.num_bins)
    trace_ax.plot(param.transformed_values)
    hist_ax.set_xlabel(param.label)
    trace_ax.set_xlabel("MCMC samples")
    trace_ax.set_ylabel(param.label)
    clean_hist_axis(hist_ax)
    clean_trace_axis(trace_ax)
    
cax, re_pacc_ax = next(axes_pair_iter)

phase = parameter_dict['phase']
frequency = parameter_dict['frequency'] 
cax.hist2d(phase.transformed_values, frequency.transformed_values,
           bins=(phase.num_bins, frequency.num_bins),
           # norm=mpl.colors.LogNorm()
           )
cax.set_xlabel("phase [rad]")
cax.set_ylabel("frequency [hz]")

final_pacc_values = list(re_acceptance_rates.values())[-1]
num_pairs = len(final_pacc_values)
re_pacc_ax.plot(list(range(num_pairs)), final_pacc_values, marker="s", ls="")
re_pacc_ax.set_xticks(list(range(num_pairs)))
re_pacc_ax.set_xticklabels(((f"{i+1}-{i+2}" for i in range(num_pairs))))
re_pacc_ax.set_xlabel("replica pairs")
re_pacc_ax.set_ylabel("acceptance rate")
clean_trace_axis(re_pacc_ax)

sr_stepsizes_ax, sr_pacc_ax = next(axes_pair_iter)

final_stepsizes_values = list(sr_stepsizes.values())[-1]
num_replicas = len(final_stepsizes_values)
xticks = list(range(1, num_replicas + 1))
sr_stepsizes_ax.plot(xticks, final_stepsizes_values, marker="s", ls="")
sr_stepsizes_ax.set_xticks(xticks)
sr_stepsizes_ax.set_xlabel("replica")
sr_stepsizes_ax.set_ylabel("stepsize")
clean_trace_axis(sr_stepsizes_ax)

final_spacc_values = list(sr_paccs.values())[-1]
num_replicas = len(final_spacc_values)
sr_pacc_ax.plot(xticks, final_spacc_values, marker="s", ls="")
sr_pacc_ax.set_xticks(xticks)
sr_pacc_ax.set_xlabel("replica")
sr_pacc_ax.set_ylabel("acceptance rate")
clean_trace_axis(sr_pacc_ax)

schedule_ax, not_needed_ax = next(axes_pair_iter)
betas = schedule['beta']
xticks = list(range(1, len(betas) + 1))
schedule_ax.plot(xticks, betas, marker="s", ls="")
schedule_ax.set_xticks(xticks)
schedule_ax.set_xlabel("replica")
schedule_ax.set_ylabel("inverse\ntemperature")
clean_trace_axis(schedule_ax)
not_needed_ax.set_visible(False)

fig.tight_layout()
    
plt.show()
