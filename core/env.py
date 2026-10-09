"""
Environment and configuration constants for the ISAC-FAS simulation stack.
"""

from __future__ import annotations

import numpy as np

# Core constants
SPEED_OF_LIGHT = 3.0e8
CARRIER_FREQUENCY = 28.0e9  # Hz
WAVELENGTH = SPEED_OF_LIGHT / CARRIER_FREQUENCY

# Reference-SNR normalization
NOISE_POWER = 1.0
REF_DISTANCE = 50.0  # meters
REF_SNR_DB = 20.0
REF_TX_POWER_DBM = 30.0

pl_amp_ref = WAVELENGTH / (4.0 * np.pi * REF_DISTANCE)
p_tx_ref = 10.0 ** ((REF_TX_POWER_DBM - 30.0) / 10.0)
snr_target = 10.0 ** (REF_SNR_DB / 10.0)
CHANNEL_SCALING_FACTOR = np.sqrt(snr_target) / (pl_amp_ref * np.sqrt(p_tx_ref))

GRID_SHAPE = (9, 9)
NUM_PORTS = GRID_SHAPE[0] * GRID_SHAPE[1]
ACTIVE_TX = 4
ACTIVE_RX = 4
PORT_SPACING = WAVELENGTH / 2

# Quantization / discrete controls
PHASE_BITS = 3
AZIMUTH_STEPS = 8
ELEVATION_STEPS = 8
TAS_INDICES = [30, 31, 39, 40]  # Tx fixed 2x2 center square
TAS_RX_INDICES = [32, 33, 41, 42]  # Rx fixed square disjoint from Tx
TAS_AZ_IDX = (AZIMUTH_STEPS - 1) // 2  # Fixed +Y facing (no rotation)
TAS_EL_IDX = (ELEVATION_STEPS - 1) // 2

# Self-interference / sensing parameters
SI_GAMMA = 2.0
TARGET_REF_INR_DB = 10.0
NUM_SI_PATHS = 8
SENSING_PROCESSING_GAIN_DB = 95.0
RICIAN_K_DB = 10.0
NLOS_PATHS = 4

# Simulation defaults
TIME_STEP = 0.02  # seconds
USER_VELOCITY = 10.0  # m/s
RCS_DB = 0.0
RCS_LINEAR = 10.0 ** (RCS_DB / 10.0)

# Motion bounds (half-space, BS faces +Y)
X_MIN, X_MAX = -100.0, 100.0
Y_MIN, Y_MAX = 10.0, 100.0  # start 10 m away from BS to avoid singularities
Z_MIN_UAV, Z_MAX_UAV = 20.0, 80.0


class FAS_Geometry:
    """
    Fluid antenna geometry on an X-Z plane (y = 0), normal pointing +Y.
    """

    def __init__(self, grid_shape: tuple[int, int] = GRID_SHAPE, spacing: float = PORT_SPACING, wavelength: float = WAVELENGTH):
        if len(grid_shape) != 2:
            raise ValueError("grid_shape must be (rows, cols)")
        self.grid_shape = tuple(int(x) for x in grid_shape)
        self.spacing = float(spacing)
        self.wavelength = float(wavelength)
        self.k = 2 * np.pi / self.wavelength

    def get_port_position(self, index: int) -> np.ndarray:
        rows, cols = self.grid_shape
        num_ports = rows * cols
        if index < 0 or index >= num_ports:
            raise IndexError(f"Port index {index} out of range for grid with {num_ports} ports.")

        row = index // cols
        col = index % cols

        x = (col - (cols - 1) / 2.0) * self.spacing
        z = (row - (rows - 1) / 2.0) * self.spacing
        y = 0.0
        return np.array([x, y, z], dtype=float)

    @staticmethod
    def _sph_to_cart(thetas: np.ndarray, phis: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                np.cos(thetas) * np.cos(phis),
                np.cos(thetas) * np.sin(phis),
                np.sin(thetas),
            ]
        )

    @staticmethod
    def _cart_to_sph(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
        r = np.linalg.norm(vectors, axis=1) + 1e-12
        thetas = np.arcsin(np.clip(z / r, -1.0, 1.0))
        phis = np.arctan2(y, x)
        return thetas, phis

    @staticmethod
    def _rotation_matrix(az_rad: float, el_rad: float) -> np.ndarray:
        beta = el_rad - np.pi / 2.0
        ca, sa = np.cos(az_rad), np.sin(az_rad)
        cb, sb = np.cos(beta), np.sin(beta)

        Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cb, -sb], [0.0, sb, cb]])
        return Rz @ Rx

    def global_to_local_angles(
        self, thetas: np.ndarray, phis: np.ndarray, az_rot_rad: float, el_rot_rad: float
    ) -> tuple[np.ndarray, np.ndarray]:
        thetas = np.asarray(thetas, dtype=float).ravel()
        phis = np.asarray(phis, dtype=float).ravel()

        v_global = self._sph_to_cart(thetas, phis)  # (N,3)
        R = self._rotation_matrix(az_rot_rad, el_rot_rad)
        v_local = (R.T @ v_global.T).T
        return self._cart_to_sph(v_local)

    def compute_steering_vector_vectorized(
        self, thetas: np.ndarray, phis: np.ndarray, port_indices: list[int]
    ) -> np.ndarray:
        thetas = np.asarray(thetas, dtype=float).ravel()
        phis = np.asarray(phis, dtype=float).ravel()

        positions = np.vstack([self.get_port_position(int(idx)) for idx in port_indices])  # (P,3)
        dirs = self._sph_to_cart(thetas, phis)  # (N,3)
        phases = self.k * (positions @ dirs.T)  # (P,N)
        return np.exp(1j * phases.T)


class Entity:
    def __init__(self, position: np.ndarray, velocity: np.ndarray):
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)
        self.trajectory: list[np.ndarray] = []

    def _apply_boundaries(self) -> None:
        # X bounds
        if self.position[0] < X_MIN:
            self.position[0] = X_MIN + (X_MIN - self.position[0])
            self.velocity[0] *= -1.0
        elif self.position[0] > X_MAX:
            self.position[0] = X_MAX - (self.position[0] - X_MAX)
            self.velocity[0] *= -1.0
        # Y bounds (half-space, y > 0)
        if self.position[1] < Y_MIN:
            self.position[1] = Y_MIN + (Y_MIN - self.position[1])
            self.velocity[1] *= -1.0
        elif self.position[1] > Y_MAX:
            self.position[1] = Y_MAX - (self.position[1] - Y_MAX)
            self.velocity[1] *= -1.0

    def step(self, dt: float) -> None:
        raise NotImplementedError

    def precompute_trajectory(self, total_steps: int, dt: float) -> list[np.ndarray]:
        self.trajectory = [self.position.copy()]
        for _ in range(int(total_steps)):
            self.step(dt)
            self.trajectory.append(self.position.copy())
        return self.trajectory


class Vehicle(Entity):
    def __init__(self):
        x = np.random.uniform(X_MIN, X_MAX)
        y = np.random.uniform(Y_MIN, Y_MAX)
        z = 0.0

        speed = np.random.uniform(10.0, 20.0)
        heading = np.random.uniform(0.0, 2.0 * np.pi)
        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)
        vz = 0.0
        super().__init__(position=np.array([x, y, z]), velocity=np.array([vx, vy, vz]))

    def step(self, dt: float) -> None:
        self.position += self.velocity * dt
        self.position[2] = 0.0
        self._apply_boundaries()

        speed = np.linalg.norm(self.velocity)
        if speed < 1e-6:
            heading = np.random.uniform(0.0, 2.0 * np.pi)
            direction = np.array([np.cos(heading), np.sin(heading), 0.0])
        else:
            direction = self.velocity / speed

        scalar_noise = np.random.normal(loc=0.0, scale=2.0)
        self.velocity += (scalar_noise * direction) 
        self.velocity[2] = 0.0


class UAV(Entity):
    def __init__(self):
        x = np.random.uniform(X_MIN, X_MAX)
        y = np.random.uniform(Y_MIN, Y_MAX)
        z = np.random.uniform(Z_MIN_UAV, Z_MAX_UAV)

        speed = np.random.uniform(5.0, 15.0)
        direction = np.random.normal(size=3)
        while np.linalg.norm(direction) < 1e-6:
            direction = np.random.normal(size=3)
        direction = direction / np.linalg.norm(direction)
        velocity = speed * direction

        super().__init__(position=np.array([x, y, z]), velocity=velocity)

    def step(self, dt: float) -> None:
        self.position += self.velocity * dt
        self._apply_boundaries()

        noise = np.random.normal(loc=0.0, scale=1.0, size=3)
        self.velocity += noise

        if self.position[2] < 1.0:
            self.position[2] = 1.0
            if self.velocity[2] < 0:
                self.velocity[2] *= 0


class Clutter(Entity):
    def __init__(self):
        x = np.random.uniform(X_MIN, X_MAX)
        y = np.random.uniform(Y_MIN, Y_MAX)
        z = 0.0
        super().__init__(position=np.array([x, y, z]), velocity=np.zeros(3))

    def step(self, dt: float) -> None:
        # Static scatterers do not move.
        return None


class ISAC_Scenario:
    def __init__(self, num_vehicles: int = 1, num_uavs: int = 1, num_clutter: int = 0, dt: float = TIME_STEP):
        self.dt = float(dt)
        self.bs_position = np.array([0.0, 0.0, 30.0], dtype=float)
        self.fas = FAS_Geometry(grid_shape=GRID_SHAPE, spacing=PORT_SPACING)

        self.vehicles = [Vehicle() for _ in range(int(num_vehicles))]
        self.uavs = [UAV() for _ in range(int(num_uavs))]
        self.clutter = [Clutter() for _ in range(int(num_clutter))]
        self.entities = self.vehicles + self.uavs + self.clutter

    def generate_trajectories(self, total_time: float) -> int:
        steps = int(np.ceil(total_time / self.dt))
        for entity in self.entities:
            entity.precompute_trajectory(steps, self.dt)
        return steps


__all__ = [
    "SPEED_OF_LIGHT",
    "CARRIER_FREQUENCY",
    "WAVELENGTH",
    "GRID_SHAPE",
    "NUM_PORTS",
    "ACTIVE_TX",
    "ACTIVE_RX",
    "PORT_SPACING",
    "PHASE_BITS",
    "AZIMUTH_STEPS",
    "ELEVATION_STEPS",
    "TAS_INDICES",
    "TAS_RX_INDICES",
    "TAS_AZ_IDX",
    "TAS_EL_IDX",
    "SI_GAMMA",
    "TARGET_REF_INR_DB",
    "NUM_SI_PATHS",
    "SENSING_PROCESSING_GAIN_DB",
    "RICIAN_K_DB",
    "NLOS_PATHS",
    "TIME_STEP",
    "USER_VELOCITY",
    "RCS_DB",
    "RCS_LINEAR",
    "X_MIN",
    "X_MAX",
    "Y_MIN",
    "Y_MAX",
    "Z_MIN_UAV",
    "Z_MAX_UAV",
    "FAS_Geometry",
    "Entity",
    "Vehicle",
    "UAV",
    "Clutter",
    "ISAC_Scenario",
    "NOISE_POWER",
    "REF_DISTANCE",
    "REF_SNR_DB",
    "REF_TX_POWER_DBM",
    "CHANNEL_SCALING_FACTOR",
]
