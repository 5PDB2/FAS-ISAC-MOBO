"""
MORBO-style multi-objective Bayesian optimization framework with trust regions.
Compatible with problem_mo.py interface (define_search_space, sample, evaluate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from abc import ABC, abstractmethod

import numpy as np
import numpy.random as npr
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


@dataclass
class MORBOConfig:
    n_trust_regions: int = 1
    surrogate_type: str = "rf"  # "rf" or "gp"
    acq_type: str = "mc"  # "mc" or "exact"
    budget: int = 100
    pop_size: int = 64
    num_mc_samples: int = 128
    candidate_source: str = "tr"  # "tr" or "feasible"
    repair_candidates: bool = True
    max_candidate_tries: int = 2000
    normalize_inputs: bool = True
    greedy_passes: int = 1
    continuous_steps: int = 11
    # If True, discrete vars are proposed on a dense continuous grid and then
    # projected back to valid discrete solutions via decode/repair.
    continuous_propose_discrete: bool = False
    discrete_relax_steps: int = 41
    use_phase_greedy: bool = True


@dataclass
class GreyboxConfig(MORBOConfig):
    greybox_num_samples: int = 128
    greybox_eps: float = 1e-8
    greybox_comm_only: bool = True


class SurrogateModel(ABC):
    def __init__(self, n_features: int, random_state: Optional[int] = None):
        self.n_features = int(n_features)
        self.random_state = random_state
        self.is_fit = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass


class RFSurrogate(SurrogateModel):
    def __init__(self, n_features: int, random_state: Optional[int] = None):
        super().__init__(n_features, random_state)
        self.model = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_fit = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fit or len(self.model.estimators_) == 0:
            mean = np.zeros(X.shape[0], dtype=float)
            std = np.ones(X.shape[0], dtype=float)
            return mean, std
        all_preds = np.vstack([tree.predict(X) for tree in self.model.estimators_])
        mean = all_preds.mean(axis=0)
        std = all_preds.std(axis=0)
        return mean, std

    def predict_trees(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fit or len(self.model.estimators_) == 0:
            return np.zeros((0, X.shape[0], 0), dtype=float)
        preds = []
        for tree in self.model.estimators_:
            pred = tree.predict(X)
            if pred.ndim == 1:
                pred = pred[:, None]
            preds.append(pred)
        return np.stack(preds, axis=0)


class GPSurrogate(SurrogateModel):
    def __init__(self, n_features: int, random_state: Optional[int] = None):
        super().__init__(n_features, random_state)
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(n_features), nu=2.5) + WhiteKernel(
            noise_level=1e-6, noise_level_bounds=(1e-8, 1e-1)
        )
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=random_state,
            n_restarts_optimizer=2,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_fit = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fit:
            mean = np.zeros(X.shape[0], dtype=float)
            std = np.ones(X.shape[0], dtype=float)
            return mean, std
        mean, std = self.model.predict(X, return_std=True)
        return mean, std


def _complex_to_vec(mat: Optional[np.ndarray], size: int) -> np.ndarray:
    if mat is None or size <= 0:
        return np.zeros(2 * size * size, dtype=float)
    arr = np.asarray(mat)
    return np.concatenate([arr.real.ravel(), arr.imag.ravel()]).astype(float)


def _vec_to_complex(vec: np.ndarray, size: int) -> Optional[np.ndarray]:
    if size <= 0:
        return None
    vec = np.asarray(vec, dtype=float)
    n = size * size
    if vec.size < 2 * n:
        return None
    real = vec[:n].reshape(size, size)
    imag = vec[n : 2 * n].reshape(size, size)
    return real + 1j * imag


def _make_psd(matrix: Optional[np.ndarray], eps: float = 1e-9) -> Optional[np.ndarray]:
    if matrix is None:
        return None
    herm = 0.5 * (matrix + matrix.conj().T)
    evals, evecs = np.linalg.eigh(herm)
    evals = np.maximum(evals, eps)
    return (evecs * evals) @ evecs.conj().T


def _rf_predict_trees(model: RandomForestRegressor, X: np.ndarray) -> np.ndarray:
    if model is None or not hasattr(model, "estimators_") or len(model.estimators_) == 0:
        return np.zeros((0, X.shape[0], 0), dtype=float)
    preds = []
    for tree in model.estimators_:
        pred = tree.predict(X)
        if pred.ndim == 1:
            pred = pred[:, None]
        preds.append(pred)
    return np.stack(preds, axis=0)


def _sample_rf_trees(tree_preds: np.ndarray, n_samples: int, rng: npr.RandomState) -> np.ndarray:
    if tree_preds.size == 0 or tree_preds.shape[0] == 0 or n_samples <= 0:
        return np.zeros((tree_preds.shape[1], 0, tree_preds.shape[2] if tree_preds.ndim == 3 else 0), dtype=float)
    n_trees = tree_preds.shape[0]
    replace = n_samples > n_trees
    idx = rng.choice(n_trees, size=n_samples, replace=replace)
    sampled = tree_preds[idx, :, :]
    return np.transpose(sampled, (1, 0, 2))


class GreyboxFeatureLayout:
    def __init__(self, num_users: int, num_rx: int):
        self.num_users = int(num_users)
        self.num_rx = int(num_rx)
        self._n_mat = 2 * self.num_rx * self.num_rx
        self.sig_slice = slice(0, self.num_users)
        self.int_slice = slice(self.num_users, 2 * self.num_users)
        self.r_in_slice = slice(2 * self.num_users, 2 * self.num_users + self._n_mat)
        self.r_sig_slice = slice(2 * self.num_users + self._n_mat, 2 * self.num_users + 2 * self._n_mat)
        self.dim = 2 * self.num_users + 2 * self._n_mat
        self.comm_slice = slice(0, 2 * self.num_users)
        self.sens_slice = slice(2 * self.num_users, self.dim)

    def encode(self, greybox: Dict[str, np.ndarray | float | None]) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=float)
        signal = np.asarray(greybox.get("signal", np.zeros(self.num_users)), dtype=float).reshape(-1)
        interference = np.asarray(
            greybox.get("interference", np.zeros(self.num_users)), dtype=float
        ).reshape(-1)
        vec[self.sig_slice] = signal[: self.num_users]
        vec[self.int_slice] = interference[: self.num_users]
        if self.num_rx > 0:
            vec[self.r_in_slice] = _complex_to_vec(greybox.get("R_in", None), self.num_rx)
            vec[self.r_sig_slice] = _complex_to_vec(greybox.get("R_sig", None), self.num_rx)
        return vec

    def decode(self, vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        vec = np.asarray(vec, dtype=float)
        signal = vec[self.sig_slice]
        interference = vec[self.int_slice]
        r_in = _vec_to_complex(vec[self.r_in_slice], self.num_rx) if self.num_rx > 0 else None
        r_sig = _vec_to_complex(vec[self.r_sig_slice], self.num_rx) if self.num_rx > 0 else None
        return signal, interference, r_in, r_sig

    def get_comm_slice(self) -> slice:
        return self.comm_slice

    def get_sens_slice(self) -> slice:
        return self.sens_slice


def pareto_front(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    points = np.asarray(points, dtype=float)
    keep = np.ones(points.shape[0], dtype=bool)
    for i, p in enumerate(points):
        if not keep[i]:
            continue
        dominated = np.all(p >= points, axis=1) & np.any(p > points, axis=1)
        keep &= ~dominated
        keep[i] = True
    return points[keep]


def hypervolume_2d(points: np.ndarray, ref_point: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    pts = pareto_front(points)
    pts = pts[np.argsort(pts[:, 0])]
    # Ensure decreasing second objective for proper 2D HV calculation
    filtered: List[Tuple[float, float]] = []
    best_y = -np.inf
    for x, y in pts[::-1]:
        if y > best_y:
            filtered.append((x, y))
            best_y = y
    filtered = filtered[::-1]

    hv = 0.0
    prev_x = ref_point[0]
    ref_y = ref_point[1]
    for x, y in filtered:
        width = max(0.0, x - prev_x)
        height = max(0.0, y - ref_y)
        hv += width * height
        prev_x = x
    return float(hv)


def ehvi_mc(
    mean: np.ndarray,
    std: np.ndarray,
    pareto_f: np.ndarray,
    ref_point: np.ndarray,
    num_mc_samples: int,
    rng: npr.RandomState,
) -> float:
    mean = np.asarray(mean, dtype=float)
    std = np.maximum(np.asarray(std, dtype=float), 1e-8)
    base_hv = hypervolume_2d(pareto_f, ref_point) if pareto_f.size else 0.0
    samples = rng.normal(loc=mean, scale=std, size=(num_mc_samples, mean.shape[0]))
    improvements = []
    for s in samples:
        combined = np.vstack([pareto_f, s]) if pareto_f.size else s[None, :]
        improvements.append(hypervolume_2d(combined, ref_point) - base_hv)
    return float(np.mean(improvements)) if improvements else 0.0


def ehvi_exact(
    mean: np.ndarray,
    std: np.ndarray,
    pareto_f: np.ndarray,
    ref_point: np.ndarray,
    grid_size: int = 20,
) -> float:
    # Approximate 2D EHVI via grid integration under independent Gaussian.
    mean = np.asarray(mean, dtype=float)
    std = np.maximum(np.asarray(std, dtype=float), 1e-8)
    if mean.shape[0] != 2:
        return 0.0
    base_hv = hypervolume_2d(pareto_f, ref_point) if pareto_f.size else 0.0

    x_lo = ref_point[0]
    y_lo = ref_point[1]
    x_hi = mean[0] + 3.0 * std[0]
    y_hi = mean[1] + 3.0 * std[1]

    xs = np.linspace(x_lo, x_hi, grid_size + 1)
    ys = np.linspace(y_lo, y_hi, grid_size + 1)
    cdf_x = norm.cdf(xs, loc=mean[0], scale=std[0])
    cdf_y = norm.cdf(ys, loc=mean[1], scale=std[1])

    ehvi = 0.0
    for i in range(grid_size):
        for j in range(grid_size):
            px = max(0.0, cdf_x[i + 1] - cdf_x[i])
            py = max(0.0, cdf_y[j + 1] - cdf_y[j])
            if px == 0.0 or py == 0.0:
                continue
            x_mid = 0.5 * (xs[i] + xs[i + 1])
            y_mid = 0.5 * (ys[j] + ys[j + 1])
            combined = np.vstack([pareto_f, [x_mid, y_mid]]) if pareto_f.size else np.array([[x_mid, y_mid]])
            hv = hypervolume_2d(combined, ref_point)
            ehvi += (hv - base_hv) * px * py
    return float(max(0.0, ehvi))


def compute_ehvi(
    mean: np.ndarray,
    std: np.ndarray,
    pareto_f: np.ndarray,
    ref_point: np.ndarray,
    method: str,
    num_mc_samples: int,
    rng: npr.RandomState,
) -> float:
    if method == "exact":
        return ehvi_exact(mean, std, pareto_f, ref_point)
    return ehvi_mc(mean, std, pareto_f, ref_point, num_mc_samples, rng)


def compute_ehvi_samples(obj_samples: np.ndarray, pareto_f: np.ndarray, ref_point: np.ndarray) -> np.ndarray:
    obj_samples = np.asarray(obj_samples, dtype=float)
    if obj_samples.size == 0:
        n_candidates = obj_samples.shape[0] if obj_samples.ndim > 0 else 0
        return np.zeros((n_candidates,), dtype=float)
    base_hv = hypervolume_2d(pareto_f, ref_point) if pareto_f.size else 0.0
    n_candidates, n_samples, _ = obj_samples.shape
    scores = np.zeros(n_candidates, dtype=float)
    for i in range(n_candidates):
        hvi_sum = 0.0
        for s in obj_samples[i]:
            combined = np.vstack([pareto_f, s]) if pareto_f.size else s[None, :]
            hvi = hypervolume_2d(combined, ref_point) - base_hv
            hvi_sum += max(0.0, hvi)
        scores[i] = hvi_sum / max(1, n_samples)
    return scores


class TrustRegion:
    def __init__(
        self,
        center: np.ndarray,
        bounds: Tuple[np.ndarray, np.ndarray],
        rng: npr.RandomState,
        init_radius: float = 0.5,
        min_radius: float = 0.05,
        max_radius: float = 1.0,
        success_tolerance: int = 3,
        failure_tolerance: int = 3,
    ):
        self.center = np.asarray(center, dtype=float)
        self.bounds = (np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float))
        self.rng = rng
        self.init_radius = float(init_radius)
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.radius = float(init_radius)
        self.success_tolerance = int(success_tolerance)
        self.failure_tolerance = int(failure_tolerance)
        self.success_counter = 0
        self.failure_counter = 0
        self.best_value = -np.inf

    def _within_region(self, points: np.ndarray) -> np.ndarray:
        lo, hi = self.bounds
        span = np.maximum(hi - lo, 1e-8)
        rel = np.abs(points - self.center) / span
        return np.all(rel <= self.radius + 1e-12, axis=1)

    def generate_candidates(self, n_candidates: int) -> np.ndarray:
        lo, hi = self.bounds
        span = np.maximum(hi - lo, 1e-8)
        perturb = self.rng.uniform(-self.radius, self.radius, size=(n_candidates, lo.shape[0]))
        candidates = self.center + perturb * span
        candidates = np.clip(candidates, lo, hi)
        return candidates

    def update_state(
        self, new_points: np.ndarray, new_values: np.ndarray, pareto_f: np.ndarray, ref_point: np.ndarray
    ) -> None:
        if new_points.size == 0:
            self.failure_counter += 1
            self._adjust_radius()
            return
        in_tr = self._within_region(new_points)
        if not np.any(in_tr):
            self.failure_counter += 1
            self._adjust_radius()
            return

        values = new_values[in_tr]
        base_hv = hypervolume_2d(pareto_f, ref_point) if pareto_f.size else 0.0
        best_idx = None
        best_hvi = -np.inf
        for i, v in enumerate(values):
            combined = np.vstack([pareto_f, v]) if pareto_f.size else v[None, :]
            hvi = hypervolume_2d(combined, ref_point) - base_hv
            if hvi > best_hvi:
                best_hvi = hvi
                best_idx = i
        if best_idx is not None and best_hvi > 1e-6:
            self.best_value = float(best_hvi)
            self.center = new_points[in_tr][best_idx]
            self.success_counter += 1
        else:
            self.failure_counter += 1
        self._adjust_radius()

    def _adjust_radius(self) -> None:
        if self.success_counter >= self.success_tolerance:
            self.radius = min(self.max_radius, self.radius * 1.5)
            self.success_counter = 0
            self.failure_counter = 0
        elif self.failure_counter >= self.failure_tolerance:
            self.radius = max(self.min_radius, self.radius / 2.0)
            self.success_counter = 0
            self.failure_counter = 0

    def restart(self, new_center: np.ndarray) -> None:
        self.center = np.asarray(new_center, dtype=float)
        self.radius = self.init_radius
        self.success_counter = 0
        self.failure_counter = 0
        self.best_value = -np.inf


class MORBOOptimizer:
    def __init__(self, problem, config: MORBOConfig):
        self.problem = problem
        self.config = config
        self.search_space = problem.define_search_space()
        self.variables = self.search_space.variables
        self.var_map = {var.name: var for var in self.variables}
        self.lower, self.upper = self._build_bounds(self.variables)
        self.dim = self.lower.shape[0]
        self.rng = npr.RandomState(0)
        self.X_history: List[np.ndarray] = []
        self.Y_history: List[np.ndarray] = []
        self.pareto_f = np.zeros((0, self._num_objectives()), dtype=float)
        self.trust_regions: List[TrustRegion] = []

    def _num_objectives(self) -> int:
        n_obj = getattr(self.problem, "num_objectives", None)
        if n_obj is None:
            return 2
        return int(n_obj)

    def _build_bounds(self, variables) -> Tuple[np.ndarray, np.ndarray]:
        lows = []
        highs = []
        for var in variables:
            size = int(np.prod(var.shape)) if len(var.shape) else 1
            lo, hi = var.bounds
            lows.extend([float(lo)] * size)
            highs.extend([float(hi)] * size)
        return np.array(lows, dtype=float), np.array(highs, dtype=float)

    def _normalize_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if not getattr(self.config, "normalize_inputs", False):
            return X
        span = np.maximum(self.upper - self.lower, 1e-12)
        return (X - self.lower) / span

    def _is_continuous(self, var) -> bool:
        vt = getattr(var, "var_type", None)
        name = getattr(vt, "name", None)
        return name == "CONTINUOUS"

    def _candidate_values(
        self,
        var,
        lo: float,
        hi: float,
        current: float | None = None,
    ) -> np.ndarray:
        if self._is_continuous(var):
            steps = max(2, int(getattr(self.config, "continuous_steps", 11)))
            vals = np.linspace(float(lo), float(hi), steps, dtype=float)
            if current is not None:
                vals = np.concatenate([vals, [float(current)]])
            return np.unique(np.clip(vals, float(lo), float(hi)))
        if bool(getattr(self.config, "continuous_propose_discrete", False)):
            steps = max(2, int(getattr(self.config, "discrete_relax_steps", 41)))
            vals = np.linspace(float(lo), float(hi), steps, dtype=float)
            if current is not None:
                vals = np.concatenate([vals, [float(current)]])
            return np.unique(np.clip(vals, float(lo), float(hi)))
        return np.arange(int(lo), int(hi) + 1, dtype=int)

    def _encode(self, sample: Dict[str, np.ndarray]) -> np.ndarray:
        parts = []
        for var in self.variables:
            val = np.asarray(sample[var.name], dtype=float).reshape(-1)
            parts.append(val)
        return np.concatenate(parts, axis=0)

    def _decode(self, vector: np.ndarray) -> Dict[str, np.ndarray]:
        result = {}
        idx = 0
        for var in self.variables:
            size = int(np.prod(var.shape)) if len(var.shape) else 1
            segment = vector[idx : idx + size]
            idx += size
            if len(var.shape):
                arr = segment.reshape(var.shape)
            else:
                arr = segment[0]
            result[var.name] = var.decode(arr)
        return result

    def _make_unique(self, arr: np.ndarray, lo: int, hi: int, forbidden: set[int] | None = None) -> np.ndarray:
        arr = np.asarray(arr, dtype=int).ravel().tolist()
        forbidden = set() if forbidden is None else set(forbidden)
        pool = [i for i in range(int(lo), int(hi) + 1) if i not in forbidden]
        seen = set(forbidden)
        for i, val in enumerate(arr):
            if val in seen:
                replacement = None
                for cand in pool:
                    if cand not in seen:
                        replacement = cand
                        break
                if replacement is None:
                    replacement = val
                arr[i] = replacement
            seen.add(arr[i])
        return np.asarray(arr, dtype=int)

    def _repair_solution(self, sol: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if "active_ports" not in sol or "active_rx_ports" not in sol:
            return sol
        tx_var = self.var_map.get("active_ports")
        rx_var = self.var_map.get("active_rx_ports")
        if tx_var is None or rx_var is None:
            return sol
        lo, hi = tx_var.bounds
        tx = np.clip(np.asarray(sol["active_ports"], dtype=int), lo, hi)
        rx = np.clip(np.asarray(sol["active_rx_ports"], dtype=int), lo, hi)
        tx_flat = self._make_unique(tx, lo, hi)
        rx_flat = self._make_unique(rx, lo, hi, forbidden=set(tx_flat.tolist()))
        sol["active_ports"] = tx_flat.reshape(tx_var.shape)
        sol["active_rx_ports"] = rx_flat.reshape(rx_var.shape)
        return sol

    def _vector_to_solution(self, x_vec: np.ndarray, repair: bool) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        sol = self._decode(x_vec)
        if repair:
            sol = self._repair_solution(sol)
            x_vec = self._encode(sol)
        return x_vec, sol

    def _sample_feasible_candidates(self, tr: TrustRegion, n_candidates: int) -> np.ndarray:
        candidates = []
        tries = 0
        max_tries = max(n_candidates, int(self.config.max_candidate_tries))
        while len(candidates) < n_candidates and tries < max_tries:
            sample = self.search_space.sample()
            x_vec = self._encode(sample)
            if tr._within_region(x_vec[None, :])[0]:
                candidates.append(x_vec)
            tries += 1
        if not candidates:
            return tr.generate_candidates(n_candidates)
        return np.asarray(candidates, dtype=float)

    def _sample_random(self) -> np.ndarray:
        sample = self.search_space.sample()
        return self._encode(sample)

    def _ensure_trust_regions(self) -> None:
        if self.trust_regions:
            return
        for _ in range(self.config.n_trust_regions):
            center = self._sample_random()
            tr = TrustRegion(center=center, bounds=(self.lower, self.upper), rng=self.rng)
            self.trust_regions.append(tr)

    def _update_pareto(self) -> None:
        if not self.Y_history:
            self.pareto_f = np.zeros((0, self._num_objectives()), dtype=float)
            return
        y_arr = np.vstack(self.Y_history)
        self.pareto_f = pareto_front(y_arr)

    def _default_ref_point(self) -> np.ndarray:
        if not self.Y_history:
            return np.zeros(self._num_objectives(), dtype=float)
        y_arr = np.vstack(self.Y_history)
        mins = y_arr.min(axis=0)
        margin = 0.1 * np.maximum(1.0, np.abs(mins))
        return mins - margin

    def _train_surrogates(self, X: np.ndarray, Y: np.ndarray) -> List[SurrogateModel]:
        surrogates: List[SurrogateModel] = []
        Xn = self._normalize_X(X)
        for obj_idx in range(Y.shape[1]):
            if self.config.surrogate_type == "gp":
                surrogate = GPSurrogate(self.dim, random_state=0)
            else:
                surrogate = RFSurrogate(self.dim, random_state=0)
            surrogate.fit(Xn, Y[:, obj_idx])
            surrogates.append(surrogate)
        return surrogates

    def _predict_objectives(self, X: np.ndarray, surrogates: Sequence[SurrogateModel]) -> Tuple[np.ndarray, np.ndarray]:
        Xn = self._normalize_X(X)
        means = []
        stds = []
        for surrogate in surrogates:
            mean, std = surrogate.predict(Xn)
            means.append(mean)
            stds.append(std)
        return np.stack(means, axis=1), np.stack(stds, axis=1)

    def _acq_scores(
        self,
        X: np.ndarray,
        surrogates: Sequence[SurrogateModel],
        pareto_f: np.ndarray,
        ref_point: np.ndarray,
    ) -> np.ndarray:
        mean, std = self._predict_objectives(X, surrogates)
        scores = np.zeros(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            scores[i] = compute_ehvi(
                mean[i],
                std[i],
                pareto_f,
                ref_point,
                method=self.config.acq_type,
                num_mc_samples=self.config.num_mc_samples,
                rng=self.rng,
            )
        return scores

    def _pick_start_x(self, X: np.ndarray, Y: np.ndarray, tr: TrustRegion | None) -> np.ndarray:
        scores = Y[:, 0] + Y[:, 1]
        if tr is None:
            return X[int(np.argmax(scores))]
        mask = tr._within_region(X)
        if np.any(mask):
            return X[mask][int(np.argmax(scores[mask]))]
        return tr.center.copy()

    def _greedy_acq_candidate(
        self,
        surrogates: Sequence[SurrogateModel],
        pareto_f: np.ndarray,
        ref_point: np.ndarray,
        start_x: np.ndarray,
        tr: TrustRegion | None = None,
    ) -> np.ndarray:
        x = np.asarray(start_x, dtype=float).copy()
        passes = max(1, int(getattr(self.config, "greedy_passes", 1)))
        idx = 0
        for _ in range(passes):
            idx = 0
            for var in self.variables:
                size = int(np.prod(var.shape)) if len(var.shape) else 1
                for offset in range(size):
                    lo = self.lower[idx + offset]
                    hi = self.upper[idx + offset]
                    values = self._candidate_values(var, lo, hi, current=x[idx + offset])
                    if values.size == 0:
                        continue
                    cand_list = []
                    for val in values:
                        cand = x.copy()
                        cand[idx + offset] = val
                        cand, _ = self._vector_to_solution(cand, self.config.repair_candidates)
                        cand_list.append(cand)
                    cand_X = np.vstack(cand_list)
                    # Continuous-propose may collapse to same discrete point.
                    cand_X = np.unique(cand_X, axis=0)
                    if tr is not None:
                        mask = tr._within_region(cand_X)
                        if not np.any(mask):
                            continue
                        cand_X = cand_X[mask]
                    if cand_X.shape[0] == 0:
                        continue
                    scores = self._acq_scores(cand_X, surrogates, pareto_f, ref_point)
                    best_idx = int(np.argmax(scores))
                    x = cand_X[best_idx]
                idx += size
        return x

    def _evaluate_solution(self, sol: Dict[str, np.ndarray], time_step: int) -> np.ndarray:
        if getattr(self.config, "use_phase_greedy", False) and hasattr(self.problem, "evaluate_with_phase_greedy"):
            try:
                return np.asarray(self.problem.evaluate_with_phase_greedy(sol, time_step), dtype=float)
            except TypeError:
                pass
        return np.asarray(self.problem.evaluate(sol, time_step=time_step), dtype=float)

    def _initial_design(self) -> None:
        n_init = min(self.config.budget, max(2, self.config.pop_size))
        for _ in range(n_init):
            x_vec = self._sample_random()
            x_vec, sol = self._vector_to_solution(x_vec, self.config.repair_candidates)
            y = self._evaluate_solution(sol, time_step=0)
            self.X_history.append(x_vec)
            self.Y_history.append(y)
        self._update_pareto()

    def run(self) -> Dict[str, np.ndarray]:
        self._ensure_trust_regions()
        if not self.X_history:
            self._initial_design()

        while len(self.X_history) < self.config.budget:
            X = np.vstack(self.X_history)
            Y = np.vstack(self.Y_history)
            surrogates = self._train_surrogates(X, Y)

            ref_point = self._default_ref_point()
            pareto_before = self.pareto_f.copy()
            best_x = None
            best_score = -np.inf
            for tr in self.trust_regions:
                start_x = self._pick_start_x(X, Y, tr)
                candidate = self._greedy_acq_candidate(surrogates, self.pareto_f, ref_point, start_x, tr=tr)
                score = float(self._acq_scores(candidate[None, :], surrogates, self.pareto_f, ref_point)[0])
                if score > best_score:
                    best_score = score
                    best_x = candidate

            if best_x is None:
                break

            x_vec, sol = self._vector_to_solution(best_x, self.config.repair_candidates)
            y = self._evaluate_solution(sol, time_step=0)
            self.X_history.append(x_vec)
            self.Y_history.append(y)

            new_points = np.asarray([x_vec], dtype=float)
            new_values = np.asarray([y], dtype=float)

            for tr in self.trust_regions:
                tr.update_state(new_points, new_values, pareto_before, ref_point)
                if tr.radius <= tr.min_radius + 1e-12:
                    tr.restart(self._sample_random())
            self._update_pareto()

        return {
            "X": np.vstack(self.X_history) if self.X_history else np.zeros((0, self.dim), dtype=float),
            "Y": np.vstack(self.Y_history) if self.Y_history else np.zeros((0, self._num_objectives()), dtype=float),
            "pareto_front": self.pareto_f,
        }


class GreyboxMORBOOptimizer(MORBOOptimizer):
    def __init__(self, problem, config: GreyboxConfig):
        super().__init__(problem, config)
        self.config: GreyboxConfig = config
        self._has_greybox = hasattr(self.problem, "evaluate_greybox")
        self.num_comm_users = int(getattr(self.problem, "num_comm_users", 0))
        self.num_rx = int(getattr(self.problem, "num_rx", 0))
        self.greybox_layout = GreyboxFeatureLayout(self.num_comm_users, self.num_rx)
        self.H_history: List[np.ndarray] = []
        self.grey_surrogates: Dict[str, Optional[RandomForestRegressor]] = {}
        self.obj_surrogates: List[SurrogateModel] = []
        evaluator = getattr(self.problem, "evaluator", None)
        self.comm_noise = float(getattr(evaluator, "comm_noise", 1.0))
        self._using_greybox = False

    def _evaluate_greybox(self, sol: Dict[str, np.ndarray], time_step: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._has_greybox:
            y, greybox = self.problem.evaluate_greybox(sol, time_step=time_step)
            h_vec = self.greybox_layout.encode(greybox)
        else:
            y = self._evaluate_solution(sol, time_step=time_step)
            h_vec = np.zeros(self.greybox_layout.dim, dtype=float)
        return np.asarray(y, dtype=float), h_vec

    def _train_grey_surrogates(self, X: np.ndarray, H: np.ndarray) -> None:
        self.grey_surrogates = {"comm": None, "sens": None}
        if H.size == 0:
            return
        Xn = self._normalize_X(X)
        comm_slice = self.greybox_layout.get_comm_slice()
        sens_slice = self.greybox_layout.get_sens_slice()
        comm_targets = H[:, comm_slice] if H.shape[1] else np.zeros((H.shape[0], 0), dtype=float)
        sens_targets = H[:, sens_slice] if H.shape[1] else np.zeros((H.shape[0], 0), dtype=float)

        if comm_targets.shape[1] > 0:
            comm_model = RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=2,
                random_state=0,
            )
            comm_model.fit(Xn, comm_targets)
            self.grey_surrogates["comm"] = comm_model

        if not self.config.greybox_comm_only and sens_targets.shape[1] > 0:
            sens_model = RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=2,
                random_state=0,
            )
            sens_model.fit(Xn, sens_targets)
            self.grey_surrogates["sens"] = sens_model

    def _greybox_objectives_from_h(self, h_vec: np.ndarray) -> np.ndarray:
        signal, interference, r_in, r_sig = self.greybox_layout.decode(h_vec)
        signal = np.maximum(signal, 0.0)
        interference = np.maximum(interference, 0.0)
        if self.num_comm_users > 0:
            sinr = signal / (interference + self.comm_noise)
            rate = float(np.sum(np.log2(1.0 + sinr)))
        else:
            rate = 0.0

        mi = 0.0
        if self.num_rx > 0 and r_in is not None and r_sig is not None:
            r_in = _make_psd(r_in)
            r_sig = _make_psd(r_sig)
            try:
                rin = r_in + self.config.greybox_eps * np.eye(self.num_rx)
                chol = np.linalg.cholesky(rin)
                z = np.linalg.solve(chol, r_sig.conj().T)
                y = np.linalg.solve(chol.conj().T, z)
                info_mat = np.eye(self.num_rx) + y.conj().T
                sign, logdet = np.linalg.slogdet(info_mat)
                mi = float(logdet / np.log(2.0)) if sign > 0 else 0.0
            except np.linalg.LinAlgError:
                mi = 0.0
        return np.array([rate, mi], dtype=float)

    def _rate_from_comm_samples(self, comm_samples: np.ndarray) -> np.ndarray:
        if self.num_comm_users <= 0 or comm_samples.size == 0:
            return np.zeros((comm_samples.shape[0], comm_samples.shape[1]), dtype=float)
        signal = np.maximum(comm_samples[..., : self.num_comm_users], 0.0)
        interference = np.maximum(comm_samples[..., self.num_comm_users : 2 * self.num_comm_users], 0.0)
        sinr = signal / (interference + self.comm_noise)
        rate = np.sum(np.log2(1.0 + sinr), axis=-1)
        return np.asarray(rate, dtype=float)

    def _sample_objective_from_surrogate(
        self, X: np.ndarray, obj_index: int, n_samples: int
    ) -> np.ndarray:
        n_candidates = X.shape[0]
        if not self.obj_surrogates or obj_index >= len(self.obj_surrogates):
            return np.zeros((n_candidates, n_samples), dtype=float)
        surrogate = self.obj_surrogates[obj_index]
        Xn = self._normalize_X(X)
        if isinstance(surrogate, RFSurrogate):
            tree_preds = surrogate.predict_trees(Xn)
            if tree_preds.ndim == 2:
                tree_preds = tree_preds[:, :, None]
            if tree_preds.shape[0] == 0:
                return np.zeros((n_candidates, n_samples), dtype=float)
            samples = _sample_rf_trees(tree_preds, n_samples, self.rng)
            return samples[:, :, 0]
        mean, std = surrogate.predict(Xn)
        mean = np.asarray(mean, dtype=float).reshape(-1, 1)
        std = np.asarray(std, dtype=float).reshape(-1, 1)
        noise = self.rng.normal(size=(n_candidates, n_samples))
        return mean + std * noise

    def _sample_objectives_from_grey(self, X: np.ndarray) -> np.ndarray:
        n_candidates = X.shape[0]
        comm_model = self.grey_surrogates.get("comm")
        sens_model = self.grey_surrogates.get("sens")
        Xn = self._normalize_X(X)
        comm_preds = _rf_predict_trees(comm_model, Xn) if comm_model else np.zeros((0, n_candidates, 0))
        sens_preds = _rf_predict_trees(sens_model, Xn) if sens_model else np.zeros((0, n_candidates, 0))

        comm_trees = comm_preds.shape[0]
        sens_trees = sens_preds.shape[0]
        available = [t for t in [comm_trees, sens_trees] if t > 0]
        if self.config.greybox_comm_only:
            available = [t for t in [comm_trees] if t > 0]
        if not available:
            return np.zeros((n_candidates, 0, self._num_objectives()), dtype=float)

        if self.config.greybox_num_samples > 0:
            n_samples = int(self.config.greybox_num_samples)
        else:
            n_samples = min(available)
        n_samples = max(1, n_samples)

        comm_slice = self.greybox_layout.get_comm_slice()
        sens_slice = self.greybox_layout.get_sens_slice()
        comm_dim = comm_slice.stop - comm_slice.start
        sens_dim = sens_slice.stop - sens_slice.start

        if self.config.greybox_comm_only:
            if comm_trees > 0 and comm_dim > 0:
                comm_samples = _sample_rf_trees(comm_preds, n_samples, self.rng)
                rate_samples = self._rate_from_comm_samples(comm_samples)
            else:
                rate_samples = self._sample_objective_from_surrogate(X, 0, n_samples)
            mi_samples = self._sample_objective_from_surrogate(X, 1, n_samples)
            mi_samples = np.maximum(mi_samples, 0.0)
            rate_samples = np.maximum(rate_samples, 0.0)
            obj_samples = np.zeros((n_candidates, n_samples, self._num_objectives()), dtype=float)
            obj_samples[:, :, 0] = rate_samples
            if self._num_objectives() > 1:
                obj_samples[:, :, 1] = mi_samples
            return obj_samples

        comm_samples = _sample_rf_trees(comm_preds, n_samples, self.rng) if comm_trees > 0 else None
        sens_samples = _sample_rf_trees(sens_preds, n_samples, self.rng) if sens_trees > 0 else None
        if comm_samples is None:
            comm_samples = np.zeros((n_candidates, n_samples, comm_dim), dtype=float)
        if sens_samples is None:
            sens_samples = np.zeros((n_candidates, n_samples, sens_dim), dtype=float)

        h_samples = np.zeros((n_candidates, n_samples, self.greybox_layout.dim), dtype=float)
        h_samples[:, :, comm_slice] = comm_samples
        h_samples[:, :, sens_slice] = sens_samples

        obj_samples = np.zeros((n_candidates, n_samples, self._num_objectives()), dtype=float)
        for i in range(n_candidates):
            for j in range(n_samples):
                obj_samples[i, j] = self._greybox_objectives_from_h(h_samples[i, j])
        return obj_samples

    def _acq_scores(
        self,
        X: np.ndarray,
        surrogates: Sequence[SurrogateModel],
        pareto_f: np.ndarray,
        ref_point: np.ndarray,
    ) -> np.ndarray:
        if self._using_greybox:
            obj_samples = self._sample_objectives_from_grey(X)
            return compute_ehvi_samples(obj_samples, pareto_f, ref_point)
        return super()._acq_scores(X, surrogates, pareto_f, ref_point)

    def _predict_objectives_from_grey(
        self, X: np.ndarray, surrogates: Optional[Sequence[SurrogateModel]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        obj_samples = self._sample_objectives_from_grey(X)
        if obj_samples.size == 0:
            means = np.zeros((X.shape[0], self._num_objectives()), dtype=float)
            stds = np.zeros_like(means)
            return means, stds
        means = obj_samples.mean(axis=1)
        stds = obj_samples.std(axis=1)
        return means, stds

    def _initial_design(self) -> None:
        n_init = min(self.config.budget, max(2, self.config.pop_size))
        for _ in range(n_init):
            x_vec = self._sample_random()
            x_vec, sol = self._vector_to_solution(x_vec, self.config.repair_candidates)
            y, h = self._evaluate_greybox(sol, time_step=0)
            self.X_history.append(x_vec)
            self.Y_history.append(y)
            self.H_history.append(h)
        self._update_pareto()

    def run(self) -> Dict[str, np.ndarray]:
        self._ensure_trust_regions()
        if not self.X_history:
            self._initial_design()

        while len(self.X_history) < self.config.budget:
            X = np.vstack(self.X_history)
            Y = np.vstack(self.Y_history)
            H = np.vstack(self.H_history) if self.H_history else np.zeros((X.shape[0], 0), dtype=float)

            use_greybox = self._has_greybox and H.shape[1] > 0
            self._using_greybox = use_greybox
            if use_greybox:
                self._train_grey_surrogates(X, H)
                if self.config.greybox_comm_only:
                    self.obj_surrogates = self._train_surrogates(X, Y)
                    surrogates = self.obj_surrogates
                else:
                    self.obj_surrogates = []
                    surrogates = []
            else:
                surrogates = self._train_surrogates(X, Y)
                self.obj_surrogates = surrogates

            ref_point = self._default_ref_point()
            pareto_before = self.pareto_f.copy()
            best_x = None
            best_score = -np.inf
            for tr in self.trust_regions:
                start_x = self._pick_start_x(X, Y, tr)
                candidate = self._greedy_acq_candidate(surrogates, self.pareto_f, ref_point, start_x, tr=tr)
                score = float(self._acq_scores(candidate[None, :], surrogates, self.pareto_f, ref_point)[0])
                if score > best_score:
                    best_score = score
                    best_x = candidate

            if best_x is None:
                break

            x_vec, sol = self._vector_to_solution(best_x, self.config.repair_candidates)
            y, h = self._evaluate_greybox(sol, time_step=0)
            self.X_history.append(x_vec)
            self.Y_history.append(y)
            self.H_history.append(h)

            new_points = np.asarray([x_vec], dtype=float)
            new_values = np.asarray([y], dtype=float)

            for tr in self.trust_regions:
                tr.update_state(new_points, new_values, pareto_before, ref_point)
                if tr.radius <= tr.min_radius + 1e-12:
                    tr.restart(self._sample_random())
            self._update_pareto()

        return {
            "X": np.vstack(self.X_history) if self.X_history else np.zeros((0, self.dim), dtype=float),
            "Y": np.vstack(self.Y_history) if self.Y_history else np.zeros((0, self._num_objectives()), dtype=float),
            "pareto_front": self.pareto_f,
        }


__all__ = [
    "MORBOConfig",
    "GreyboxConfig",
    "SurrogateModel",
    "RFSurrogate",
    "GPSurrogate",
    "compute_ehvi",
    "compute_ehvi_samples",
    "TrustRegion",
    "MORBOOptimizer",
    "GreyboxFeatureLayout",
    "GreyboxMORBOOptimizer",
]
