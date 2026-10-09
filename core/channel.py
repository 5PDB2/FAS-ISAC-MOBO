"""
Channel models for comm, radar targets/clutter, and self-interference.
"""

from __future__ import annotations

import numpy as np

from .env import (
    AZIMUTH_STEPS,
    CHANNEL_SCALING_FACTOR,
    ELEVATION_STEPS,
    NLOS_PATHS,
    NUM_PORTS,
    RICIAN_K_DB,
    SENSING_PROCESSING_GAIN_DB,
    TIME_STEP,
    WAVELENGTH,
)

EPS = 1e-12
ANGLE_SPREAD_RAD = np.deg2rad(20.0)


class ChannelModel:
    """
    Comm channels follow a Rician model; radar channels use a two-way path loss model.
    """

    def __init__(self, scenario, config):
        self.scenario = scenario
        self.config = config
        if isinstance(config, dict):
            self.dt = float(getattr(scenario, "dt", config.get("TIME_STEP", TIME_STEP)))
            self.k_db = float(config.get("RICIAN_K_DB", RICIAN_K_DB))
            self.nlos_paths = int(config.get("NLOS_PATHS", NLOS_PATHS))
            self.si_k_db = float(config.get("SI_K_DB", 10.0))
        else:
            self.dt = float(getattr(scenario, "dt", getattr(config, "TIME_STEP", TIME_STEP)))
            self.k_db = float(getattr(config, "RICIAN_K_DB", RICIAN_K_DB))
            self.nlos_paths = int(getattr(config, "NLOS_PATHS", NLOS_PATHS))
            self.si_k_db = float(getattr(config, "SI_K_DB", 10.0))

        self.processing_gain = float(np.sqrt(10.0 ** (SENSING_PROCESSING_GAIN_DB / 10.0)))
        self.num_ports = int(NUM_PORTS)
        self.H_scatter_global = (
            np.random.normal(size=(self.num_ports, self.num_ports))
            + 1j * np.random.normal(size=(self.num_ports, self.num_ports))
        ) / np.sqrt(2.0)

    @staticmethod
    def _geometry(bs_pos: np.ndarray, entity_pos: np.ndarray) -> tuple[float, float, float]:
        diff = entity_pos - bs_pos
        distance = float(np.linalg.norm(diff) + EPS)
        theta = float(np.arcsin(np.clip(diff[2] / distance, -1.0, 1.0)))
        phi = float(np.arctan2(diff[1], diff[0]))
        return distance, theta, phi

    def _get_entity_position(self, entity, time_step: int) -> np.ndarray:
        if hasattr(entity, "trajectory") and len(entity.trajectory) > 0:
            idx = int(np.clip(time_step, 0, len(entity.trajectory) - 1))
            return np.asarray(entity.trajectory[idx], dtype=float)
        return np.asarray(entity.position, dtype=float)

    @staticmethod
    def _orientation_radians(orientation_indices: tuple[int, int]) -> tuple[float, float]:
        """Map decoded orientation indices to the array's azimuth/elevation."""
        az_idx, el_idx = (int(orientation_indices[0]), int(orientation_indices[1]))
        az_idx = int(np.clip(az_idx, 0, AZIMUTH_STEPS - 1))
        el_idx = int(np.clip(el_idx, 0, ELEVATION_STEPS - 1))
        # ``TAS_AZ_IDX``/``TAS_EL_IDX`` are documented as the unrotated
        # +Y-facing baseline.  Keep that physical reference fixed at (3, 3)
        # while quantizing rotation uniformly in each angular coordinate.
        azimuth = 2.0 * np.pi * (az_idx - 3) / max(1, AZIMUTH_STEPS)
        elevation = 0.5 * np.pi + np.pi * (el_idx - 3) / max(1, ELEVATION_STEPS)
        return float(azimuth), float(elevation)

    def _steering(
        self,
        theta: float,
        phi: float,
        active_ports: list[int],
        orientation_indices: tuple[int, int] | None = None,
    ) -> np.ndarray:
        if orientation_indices is not None:
            azimuth, elevation = self._orientation_radians(orientation_indices)
            theta_arr, phi_arr = self.scenario.fas.global_to_local_angles(
                np.array([theta], dtype=float), np.array([phi], dtype=float), azimuth, elevation
            )
            theta, phi = float(theta_arr[0]), float(phi_arr[0])
        steer = self.scenario.fas.compute_steering_vector_vectorized(
            np.array([theta], dtype=float), np.array([phi], dtype=float), active_ports
        )
        return steer[0]

    def compute_comm_channel(
        self,
        time_step: int,
        vehicle_idx: int,
        active_tx: list[int],
        orientation_indices: tuple[int, int] | None = None,
    ) -> np.ndarray:
        vehicles = getattr(self.scenario, "vehicles", [])
        if vehicle_idx < 0 or vehicle_idx >= len(vehicles):
            raise IndexError("vehicle_idx out of range for scenario vehicles.")

        bs_pos = np.asarray(self.scenario.bs_position, dtype=float)
        vehicle = vehicles[vehicle_idx]
        pos = self._get_entity_position(vehicle, time_step)
        dist, theta, phi = self._geometry(bs_pos, pos)
        
        A_los = WAVELENGTH / (4.0 * np.pi * dist)
        psi_los = -2.0 * np.pi * dist / WAVELENGTH
        a_los = self._steering(theta, phi, active_tx, orientation_indices)
        h_los = A_los * np.exp(1j * psi_los) * a_los

        n_paths = max(1, int(self.nlos_paths))
        nlos_sum = np.zeros_like(h_los, dtype=complex)
        for _ in range(n_paths):
            th = np.clip(theta + np.random.uniform(-ANGLE_SPREAD_RAD, ANGLE_SPREAD_RAD), -0.5 * np.pi, 0.5 * np.pi)
            ph = phi + np.random.uniform(-ANGLE_SPREAD_RAD, ANGLE_SPREAD_RAD)
            if ph > np.pi:
                ph -= 2.0 * np.pi
            elif ph < -np.pi:
                ph += 2.0 * np.pi
            a_n = self._steering(th, ph, active_tx, orientation_indices)
            amp = np.abs(np.random.normal(loc=A_los, scale=0.5 * A_los))
            psi_n = np.random.uniform(0.0, 2.0 * np.pi)
            nlos_sum += amp * np.exp(1j * psi_n) * a_n
        nlos_sum /= np.sqrt(n_paths)

        K_lin = 10.0 ** (self.k_db / 10.0)
        w_los = np.sqrt(K_lin / (K_lin + 1.0))
        w_nlos = np.sqrt(1.0 / (K_lin + 1.0))
        h_total = w_los * h_los + w_nlos * nlos_sum
        return h_total * CHANNEL_SCALING_FACTOR

    def compute_radar_channel(
        self,
        entity,
        active_tx: list[int],
        active_rx: list[int],
        time_step: int = 0,
        orientation_indices: tuple[int, int] | None = None,
    ) -> np.ndarray:
        if isinstance(entity, (int, np.integer)):
            entities = getattr(self.scenario, "entities", [])
            if entity < 0 or entity >= len(entities):
                raise IndexError("entity_idx out of range for scenario entities.")
            entity_obj = entities[int(entity)]
        else:
            entity_obj = entity

        bs_pos = np.asarray(self.scenario.bs_position, dtype=float)
        pos = self._get_entity_position(entity_obj, time_step)
        dist, theta, phi = self._geometry(bs_pos, pos)
        A_radar = (WAVELENGTH / (4.0 * np.pi * dist)) ** 2

        a_tx = self._steering(theta, phi, active_tx, orientation_indices)
        b_rx = self._steering(theta, phi, active_rx, orientation_indices)
        H = A_radar * np.outer(b_rx, a_tx.conj())
        return H * self.processing_gain * CHANNEL_SCALING_FACTOR

    def compute_si_channel(
        self,
        active_tx: list[int],
        active_rx: list[int],
        si_cancellation_db: float,
        K_db: float | None = None,
        si_override: np.ndarray | None = None,
    ) -> np.ndarray:
        if si_override is not None:
            override = np.asarray(si_override, dtype=np.complex128)
            selected_shape = (len(active_rx), len(active_tx))
            if override.shape == selected_shape:
                return override.copy()
            if override.ndim != 2:
                raise ValueError("si_override must be a selected Rx-by-Tx or full-port matrix.")
            rx_idx = np.asarray(active_rx, dtype=int)
            tx_idx = np.asarray(active_tx, dtype=int)
            if rx_idx.size == 0 or tx_idx.size == 0:
                return np.zeros(selected_shape, dtype=np.complex128)
            if np.any(rx_idx < 0) or np.any(tx_idx < 0) or np.max(rx_idx) >= override.shape[0] or np.max(tx_idx) >= override.shape[1]:
                raise ValueError("si_override is too small for the selected port indices.")
            return override[np.ix_(rx_idx, tx_idx)].copy()
        if K_db is None:
            K_db = float(self.si_k_db)
        tx_pos = np.vstack([self.scenario.fas.get_port_position(int(i)) for i in active_tx])  # (Nt,3)
        rx_pos = np.vstack([self.scenario.fas.get_port_position(int(i)) for i in active_rx])  # (Nr,3)

        diff = rx_pos[:, None, :] - tx_pos[None, :, :]  # (Nr,Nt,3)
        dist = np.linalg.norm(diff, axis=2)
        dist = np.maximum(dist, 1e-6)

        # LoS component based on physical path loss.
        phase = -1j * 2.0 * np.pi * dist / WAVELENGTH
        A_los = WAVELENGTH / (4.0 * np.pi * dist)
        H_los = A_los * np.exp(phase)

        # Scattering component based on a static global field and average path loss.
        H_sub = self.H_scatter_global[np.ix_(active_rx, active_tx)]
        d_avg = float(np.mean(dist))
        d_avg = max(d_avg, 1e-6)
        A_scat = WAVELENGTH / (4.0 * np.pi * d_avg)
        H_scatter = A_scat * H_sub

        K_lin = 10.0 ** (K_db / 10.0)
        w_los = np.sqrt(K_lin / (K_lin + 1.0))
        w_scat = np.sqrt(1.0 / (K_lin + 1.0))
        H_total = w_los * H_los + w_scat * H_scatter

        beta = CHANNEL_SCALING_FACTOR * np.sqrt(10.0 ** (-si_cancellation_db / 10.0))
        return beta * H_total


__all__ = ["ChannelModel"]
