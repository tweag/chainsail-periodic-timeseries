import sys

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

burnin = 100
samples = np.load(sys.argv[1])[burnin:]

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
    

fig, axes = plt.subplots(5, 2, figsize=(10, 10))

for param, (hist_ax, trace_ax) in zip(parameters, axes):
    hist_ax.hist(param.transformed_values, bins=param.num_bins)
    trace_ax.plot(param.transformed_values)
    hist_ax.set_xlabel(param.label)
    trace_ax.set_xlabel(param.label)
    clean_hist_axis(hist_ax)
    clean_trace_axis(trace_ax)
    
cax = axes[len(parameters),0]
phase = parameter_dict['phase']
frequency = parameter_dict['frequency'] 
cax.hist2d(phase.transformed_values, frequency.transformed_values,
           bins=(phase.num_bins, frequency.num_bins),
           # norm=mpl.colors.LogNorm()
           )
cax.set_xlabel("phase [rad]")
cax.set_ylabel("frequency [hz]")

axes[-1, -1].set_visible(False)

fig.tight_layout()
    
plt.show()
