"""
Multi-objective optimization interface with discrete search space and SINR/MI metrics for ISAC-FAS.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from .channel import ChannelModel
from .env import (
    ACTIVE_RX,
    ACTIVE_TX,
    AZIMUTH_STEPS,
    ELEVATION_STEPS,
    NUM_PORTS,
    PHASE_BITS,
)


class VarType(Enum):
    CONTINUOUS = auto()
    DISCRETE = auto()


class OptVariable:
    def __init__(
        self,
        name: str,
        var_type: VarType,
        shape: Sequence[int] | Tuple[int, ...] = (),
        bounds: Sequence[int] | Sequence[float] = (0, 1),
    ):
        self.name = name
        self.var_type = var_type
        self.shape = tuple(shape) if len(shape) else ()
        self.bounds = tuple(bounds)

    def decode(self, value: Any) -> np.ndarray:
        arr = np.asarray(value)
        lo, hi = self.bounds
        if self.var_type == VarType.DISCRETE:
            arr = np.rint(arr).astype(int)
            arr = np.clip(arr, int(lo), int(hi))
            return arr
        arr = np.asarray(arr, dtype=float)
        arr = np.clip(arr, float(lo), float(hi))
        return arr


class SearchSpace:
    def __init__(self, variables: Iterable[OptVariable]):
        self.variables: List[OptVariable] = list(variables)

    def sample(self) -> Dict[str, np.ndarray]:
        sample: Dict[str, np.ndarray] = {}
        used_ports: set[int] = set()
        for var in self.variables:
            lo, hi = var.bounds
            size = var.shape if len(var.shape) else None
            if var.name in {"active_ports", "active_rx_ports"}:
                count = int(np.prod(var.shape)) if len(var.shape) else 1
                pool = np.arange(int(lo), int(hi) + 1)
                if used_ports:
                    pool = np.setdiff1d(pool, np.fromiter(used_ports, dtype=int), assume_unique=False)
                replace = False
                if pool.size < count:
                    pool = np.arange(int(lo), int(hi) + 1)
                    replace = pool.size < count
                val = np.random.choice(pool, size=count, replace=replace)
                if len(var.shape):
                    val = val.reshape(var.shape)
                sample[var.name] = var.decode(val)
                used_ports.update(np.asarray(sample[var.name]).ravel().tolist())
                continue
            if var.var_type == VarType.CONTINUOUS:
                val = np.random.uniform(float(lo), float(hi), size=size)
            else:
                val = np.random.randint(int(lo), int(hi) + 1, size=size)
            sample[var.name] = var.decode(val)
        return sample


class FAS_ISAC_Problem_MO:
    def __init__(self, scenario, config, channel_model: ChannelModel):
        self.scenario = scenario
        self.config = config
        self.channel_model = channel_model

        if isinstance(config, dict):
            self.alpha = config.get("alpha", 1.0)
        else:
            self.alpha = getattr(config, "alpha", 1.0)

        self.num_objectives = 2
        self.objective_names = ["rate", "mi"]

        self.num_tx = ACTIVE_TX
        self.num_rx = ACTIVE_RX
        self.total_ports = NUM_PORTS
        self.num_comm_users = len(getattr(self.scenario, "vehicles", []))
        self.phase_levels = 2 ** PHASE_BITS
        # Local import to avoid circular dependency with evaluate.py.
        from .evaluate import ISACEvaluator

        self.evaluator = ISACEvaluator(scenario, config, channel_model)

    def define_search_space(self) -> SearchSpace:
        phase_shape = (self.num_tx, self.num_comm_users)
        variables = [
            OptVariable(
                name="active_ports",
                var_type=VarType.DISCRETE,
                shape=(self.num_tx,),
                bounds=(0, self.total_ports - 1),
            ),
            OptVariable(
                name="active_rx_ports",
                var_type=VarType.DISCRETE,
                shape=(self.num_rx,),
                bounds=(0, self.total_ports - 1),
            ),
            OptVariable(
                name="phase_indices",
                var_type=VarType.DISCRETE,
                shape=phase_shape,
                bounds=(0, self.phase_levels - 1),
            ),
            OptVariable(
                name="az_idx",
                var_type=VarType.DISCRETE,
                shape=(1,),
                bounds=(0, AZIMUTH_STEPS - 1),
            ),
            OptVariable(
                name="el_idx",
                var_type=VarType.DISCRETE,
                shape=(1,),
                bounds=(0, ELEVATION_STEPS - 1),
            ),
        ]
        if isinstance(self.config, dict):
            optimize_alpha = self.config.get("OPTIMIZE_ALPHA", self.config.get("optimize_alpha", False))
            bounds = self.config.get("ALPHA_BOUNDS", self.config.get("alpha_bounds", (0.0, 1.0)))
        else:
            optimize_alpha = getattr(self.config, "OPTIMIZE_ALPHA", getattr(self.config, "optimize_alpha", False))
            bounds = getattr(self.config, "ALPHA_BOUNDS", getattr(self.config, "alpha_bounds", (0.0, 1.0)))
        if optimize_alpha:
            try:
                lo, hi = bounds
            except Exception:
                lo, hi = 0.0, 1.0
            variables.append(
                OptVariable(
                    name="alpha",
                    var_type=VarType.CONTINUOUS,
                    shape=(),
                    bounds=(float(lo), float(hi)),
                )
            )
        return SearchSpace(variables)

    def evaluate(self, solution_vector: dict, time_step: int, alpha: float = None) -> np.ndarray:
        return self.evaluator.evaluate(solution_vector, time_step)

    def evaluate_with_phase_greedy(
        self,
        solution_vector: dict,
        time_step: int,
        alpha: float | None = None,
        phase_passes: int | None = None,
        use_true_channel: bool | None = None,
    ) -> np.ndarray:
        return self.evaluator.evaluate_with_phase_greedy(
            solution_vector,
            time_step,
            alpha=alpha,
            phase_passes=phase_passes,
            use_true_channel=use_true_channel,
        )

    def evaluate_greybox(
        self,
        solution_vector: dict,
        time_step: int,
        alpha: float | None = None,
        phase_passes: int | None = None,
        use_true_channel: bool | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Return objectives plus intermediate variables for grey-box optimization."""
        return self.evaluator.evaluate_greybox(
            solution_vector,
            time_step,
            alpha=alpha,
            phase_passes=phase_passes,
            use_true_channel=use_true_channel,
        )


__all__ = ["VarType", "OptVariable", "SearchSpace", "FAS_ISAC_Problem_MO"]
