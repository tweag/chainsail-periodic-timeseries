"""
Chainsail-compatible versions of the original model in
https://discourse.mc-stan.org/t/ideas-for-modelling-a-periodic-timeseries/22038
"""
from dataclasses import dataclass

import numpy as np
import sys
sys.path.insert(0, "/home/simeon/projects/tweag/chainsail-resources/chainsail_helpers/")

from chainsail_helpers.pdf import PDF

LOG_MIN = 1e-308
LOG_MAX = 1e+308

data = np.loadtxt("data.csv", delimiter=",", skiprows=1)

def stable_log(x, x_min=LOG_MIN, x_max=LOG_MAX):
    """
    Safe version of log, clips argument such that overflow does not occur.
    """

    x_min = max(x_min, LOG_MIN)
    x_max = min(x_max, LOG_MAX)

    return np.log(np.clip(x, x_min, x_max))


def weibull_log_prob(x, scale, shape):
    k = shape
    lammda = scale
    return (k - 1) * stable_log(x / lammda) - (x / lammda) ** k


@dataclass
class UnboundedParams:
    log_noise: float
    log_frequency: float
    phamp1: float
    phamp2: float


class BoundedParams:
    def __init__(self, unbounded_params: UnboundedParams) -> None:
        self._noise = np.exp(unbounded_params.log_noise)
        self._frequency = np.exp(unbounded_params.log_frequency)
        self._phamp = np.array([unbounded_params.phamp1, unbounded_params.phamp2])

    @property
    def noise(self) -> float:
        return self._noise

    @property
    def frequency(self) -> float:
        return self._frequency

    @property
    def phamp(self) -> np.array:
        return self._phamp
    

class TransformedParams:
    def __init__(self, bounded_params: BoundedParams, x: np.array) -> None:
        _frequency: float = bounded_params.frequency
        self._phase: float = np.arctan2(*bounded_params.phamp)
        self._amp: float = np.sqrt(np.sum(bounded_params.phamp ** 2))
        self._f: np.ndarray = self.amp * np.sin(x * _frequency - self.phase)

    @property
    def phase(self) -> float:
        return self._phase

    @property
    def amp(self) -> float:
        return self._amp

    @property
    def f(self) -> np.array:
        return self._f


class ManualPDF(PDF):
    def __init__(self, data: str) -> None:
        raw_data = np.loadtxt(data, delimiter=",", skiprows=1).T
        raw_x, raw_y = raw_data[0], raw_data[2]
        self._x, self._y = self._transform_data(raw_x, raw_y)

    def _transform_data(self, raw_x: np.array, raw_y: np.array):
        return raw_x, (raw_y - raw_y.mean()) / raw_y.std()
        
    def log_prior(self, x: np.ndarray) -> float:
        unbounded_params = UnboundedParams(*x)
        bounded_params = BoundedParams(unbounded_params)
        trans_params = TransformedParams(bounded_params, self._x)

        lp_noise = weibull_log_prob(bounded_params.noise, 2, 1)
        lp_freq = weibull_log_prob(bounded_params.frequency, 2, 2)
        lp_amp = weibull_log_prob(trans_params.amp, 2, 1)

        log_noise_jacobian = bounded_params.noise
        log_freq_jacobian = bounded_params.frequency
        log_amp_jacobian = stable_log(trans_params.amp)

        return lp_noise + lp_amp + lp_freq + log_amp_jacobian + log_noise_jacobian #+ log_freq_jacobian

    def log_likelihood(self, x: np.ndarray) -> float:
        unbounded_params = UnboundedParams(*x)
        bounded_params = BoundedParams(unbounded_params)
        trans_params = TransformedParams(bounded_params, self._x)
        
        return -0.5 / bounded_params.noise ** 2 * np.sum((self._y - trans_params.f) ** 2) - unbounded_params.log_noise

    def log_prob(self, x: np.ndarray) -> float:
        return self.log_likelihood(x) + self.log_prior(x)

    def log_prob_gradient(self, _: np.ndarray) -> None:
        pass


class BridgeStanPDF(PDF):
    def __init__(self, model: str, data: str) -> None:
        import json
        import sys
        sys.path.insert(0, "/home/simeon/src/bridgestan/python/")
        import bridgestan as bs

        bs.set_bridgestan_path("/home/simeon/src/bridgestan")

        basedir = "/home/simeon/projects/periodic_timeseries_chainsail/"
        data = np.loadtxt(data, delimiter=",", skiprows=1).T
        data_dict = dict(x=data[0].tolist(), y=data[2].tolist(), n=len(data[0]))
        import os
        # os.environ['LD_LIBRARY_PATH'] = os.getenv('LD_LIBRARY_PATH') + ":/nix/store/cynvahq5hc4g8hg99vajx6p1gw7sqhm7-glibc-2.34-210/lib/"
        self._model = bs.StanModel.from_stan_file(model, json.dumps(data_dict))
        
    def log_prob(self, x: np.ndarray) -> float:
        return self._model.log_densiity(x, jacobian=False)

    def log_prob_gradient(self, _: np.ndarray) -> None:
        return self._model.log_density_gradient(x, jacobian=False)[1]

if True:
    # Use httpstan to compile Stan model and evaluate log prob and its gradient
    from chainsail_helpers.pdf.stan import StanPDF

    with open("chainsail_compatible_model.stan") as f:
        model_code = f.read()
    data = np.loadtxt("data.csv", delimiter=",", skiprows=1)

    pdf = StanPDF(model_code,{"n": len(data), "x": data[:,0].tolist(), "y": data[:,2].tolist()})
else:
    # Use the above, likely incorrect from-scratch implementation of the model
    pdf = ManualPDF("data.csv")

# pdf = BridgeStanPDF("chainsail_compatible_model.stan", data="data.csv")
    
initial_states = np.array([-2.13, 1.738, 0.0142, 1.418])
