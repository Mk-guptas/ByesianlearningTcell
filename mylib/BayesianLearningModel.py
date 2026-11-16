import numpy as np
import matplotlib.pyplot as plt






class BayesianDensityEvolution1D:
    """
    Absolute-density Bayesian evolution model for non-exhausted and exhausted
    T-cell subpopulations on a one-dimensional phenotype axis.

    The model evolves two absolute densities:

        n_m(x,t): non-exhausted cells
        n_e(x,t): exhausted cells

    with shared transport, shared Bayesian adaptation, shared fitness growth,
    and one-way exhaustion transfer

        r(x) = kappa(x) f(x).

    The equations are

        dn_m/dt = L_x[n_m] + B_q[n_m] + (f(x) - r(x)) n_m
        dn_e/dt = L_x[n_e] + B_q[n_e] + f(x) n_e + r(x) n_m.

    The Bayesian operator is applied separately to each compartment:

        B_q[n] = tau * ((N * p(y|x) / integral p(y|x)n(x)dx) - 1) * n(x),

    where N = integral n(x)dx. This preserves each compartment's mass when
    transport, growth, and transfer are absent.
    """

    def __init__(
        self,
        likelihood_function,
        potential_function,
        fitness_function=None,
        exhaustion_rate_function=None,
        k3_nonexhausted_function=None,
        exhausted_k3_value=0.001 * 60.0,
        D=0.05,
        tau=1.0,
        tau_nonexhausted=None,
        tau_exhausted=None,
        v=0.0,
        exhausted_fitness_mode="same",
        exhausted_fixed_fitness=None,
        x_range=(-5.0, 5.0),
        Nx=400,
        dt=1e-4,
        Total_T=1.0,
        eps=1e-14,
    ):
        if not callable(likelihood_function):
            raise TypeError("likelihood_function must be callable")
        if not callable(potential_function):
            raise TypeError("potential_function must be callable")
        if fitness_function is not None and not callable(fitness_function):
            raise TypeError("fitness_function must be callable when provided")
        if exhaustion_rate_function is None or not callable(exhaustion_rate_function):
            raise TypeError("exhaustion_rate_function must be callable")
        if k3_nonexhausted_function is not None and not callable(k3_nonexhausted_function):
            raise TypeError("k3_nonexhausted_function must be callable when provided")

        self.likelihood_function = likelihood_function
        self.potential_function = potential_function
        self.fitness_function = fitness_function
        self.exhaustion_rate_function = exhaustion_rate_function
        self.k3_nonexhausted_function = k3_nonexhausted_function
        self.exhausted_k3_value = float(exhausted_k3_value)

        self.D = float(D)
        self.tau = float(tau)
        self.tau_nonexhausted = self.tau if tau_nonexhausted is None else float(tau_nonexhausted)
        self.tau_exhausted = self.tau if tau_exhausted is None else float(tau_exhausted)
        self.v = float(v)
        self.exhausted_fitness_mode = str(exhausted_fitness_mode).lower()
        self.exhausted_fixed_fitness = None if exhausted_fixed_fitness is None else float(exhausted_fixed_fitness)
        self.x_min, self.x_max = self._validate_range(x_range, "x_range")
        self.Nx = int(Nx)
        self.dt = float(dt)
        self.T = float(Total_T)
        self.eps = float(eps)
        self.Nt = int(round(self.T / self.dt))

        if self.D < 0:
            raise ValueError("D must be non-negative")
        if self.tau < 0 or self.tau_nonexhausted < 0 or self.tau_exhausted < 0:
            raise ValueError("tau, tau_nonexhausted, and tau_exhausted must be non-negative")
        if self.exhausted_fitness_mode not in {"same", "fixed"}:
            raise ValueError("exhausted_fitness_mode must be 'same' or 'fixed'")
        if self.exhausted_fitness_mode == "fixed" and self.exhausted_fixed_fitness is None:
            raise ValueError("exhausted_fixed_fitness must be provided when exhausted_fitness_mode='fixed'")
        if self.exhausted_fixed_fitness is not None and not np.isfinite(self.exhausted_fixed_fitness):
            raise ValueError("exhausted_fixed_fitness must be finite")
        if self.exhausted_k3_value < 0:
            raise ValueError("exhausted_k3_value must be non-negative")
        if self.Nx < 3:
            raise ValueError("Nx must be at least 3")
        if self.dt <= 0 or self.T <= 0:
            raise ValueError("dt and Total_T must be positive")

        self.x = np.linspace(self.x_min, self.x_max, self.Nx)
        self.dx = self.x[1] - self.x[0]
        self.x_edge = 0.5 * (self.x[:-1] + self.x[1:])

        self.likelihood_grid = self._evaluate_likelihood_function(self.x, sigma_x=1.0)
        self.potential_grid = self._evaluate_potential_function(self.x)
        self.fitness_grid = self._evaluate_fitness_function(self.x)
        self.exhausted_fitness_grid = self._evaluate_exhausted_fitness_grid(self.x)
        self.kappa_grid = self._evaluate_exhaustion_rate_function(self.x)
        self.transfer_rate_grid = self.kappa_grid * self.fitness_grid
        self.k3_nonexhausted_grid = self._evaluate_k3_nonexhausted_function(self.x)
        self.k3_exhausted_grid = np.full_like(self.x, self.exhausted_k3_value, dtype=float)

        self.force_edge = -(self.potential_grid[1:] - self.potential_grid[:-1]) / self.dx
        self.times = np.linspace(0.0, self.Nt * self.dt, self.Nt + 1)

        self.N_nonexhausted_history = np.zeros(self.Nt + 1)
        self.N_exhausted_history = np.zeros(self.Nt + 1)
        self.N_total_history = np.zeros(self.Nt + 1)

        self.mean_x_nonexhausted_history = np.zeros(self.Nt + 1)
        self.mean_x_exhausted_history = np.zeros(self.Nt + 1)
        self.mean_x_total_history = np.zeros(self.Nt + 1)

        self.mean_fitness_nonexhausted_history = np.zeros(self.Nt + 1)
        self.mean_fitness_exhausted_history = np.zeros(self.Nt + 1)
        self.mean_fitness_total_history = np.zeros(self.Nt + 1)

        self.mean_k3_nonexhausted_history = np.zeros(self.Nt + 1)
        self.mean_k3_exhausted_history = np.zeros(self.Nt + 1)
        self.mean_k3_total_history = np.zeros(self.Nt + 1)

        self.transfer_total_history = np.zeros(self.Nt + 1)
        self.snapshots_nonexhausted = {}
        self.snapshots_exhausted = {}
        self.snapshots_total = {}

        self.final_nonexhausted = None
        self.final_exhausted = None

    @staticmethod
    def _validate_range(value_range, name):
        value_range = np.asarray(value_range, dtype=float)
        if value_range.shape != (2,):
            raise ValueError(f"{name} must be a two-number array/list: [min, max]")
        if value_range[0] >= value_range[1]:
            raise ValueError(f"{name}[0] must be smaller than {name}[1]")
        return float(value_range[0]), float(value_range[1])

    @staticmethod
    def _as_grid_values(values, x, name):
        values = np.asarray(values, dtype=float)
        if values.shape == ():
            values = np.full_like(x, float(values), dtype=float)
        if values.shape != x.shape:
            raise ValueError(f"{name} must return a scalar or an array with the same shape as x")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} returned non-finite values")
        return values

    @staticmethod
    def _gaussian_1d(x, mean, sigma):
        return np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))

    def _call_likelihood_function(self, x, sigma_x):
        try:
            return self.likelihood_function(x, sigma_x)
        except TypeError:
            return self.likelihood_function(x)

    def _evaluate_likelihood_function(self, x, sigma_x):
        values = self._as_grid_values(
            self._call_likelihood_function(x, sigma_x),
            x,
            "likelihood_function",
        )
        if np.any(values < 0):
            raise ValueError("likelihood_function must return non-negative values")
        if np.max(values) <= self.eps:
            raise ValueError("likelihood_function is zero everywhere on the grid")
        return values

    def _evaluate_potential_function(self, x):
        return self._as_grid_values(self.potential_function(x), x, "potential_function")

    def _evaluate_fitness_function(self, x):
        if self.fitness_function is None:
            return np.ones_like(x, dtype=float)
        return self._as_grid_values(self.fitness_function(x), x, "fitness_function")

    def _evaluate_exhausted_fitness_grid(self, x):
        if self.exhausted_fitness_mode == "same":
            return self.fitness_grid.copy()
        return np.full_like(x, self.exhausted_fixed_fitness, dtype=float)

    def _evaluate_exhaustion_rate_function(self, x):
        values = self._as_grid_values(self.exhaustion_rate_function(x), x, "exhaustion_rate_function")
        if np.any(values < 0):
            raise ValueError("exhaustion_rate_function must return non-negative values")
        return values

    def _evaluate_k3_nonexhausted_function(self, x):
        if self.k3_nonexhausted_function is None:
            return np.zeros_like(x, dtype=float)
        values = self._as_grid_values(self.k3_nonexhausted_function(x), x, "k3_nonexhausted_function")
        if np.any(values < 0):
            raise ValueError("k3_nonexhausted_function must return non-negative values")
        return values

    def _initial_density(self, total_number, initial_mean, initial_sigma):
        total_number = float(total_number)
        initial_sigma = float(initial_sigma)
        if total_number < 0:
            raise ValueError("initial cell numbers must be non-negative")
        if initial_sigma <= 0:
            raise ValueError("initial_sigma must be positive")
        if total_number == 0:
            return np.zeros_like(self.x, dtype=float)
        density = self._gaussian_1d(self.x, float(initial_mean), initial_sigma)
        mass = np.sum(density) * self.dx
        if mass <= 0:
            raise ValueError("initial density mass is non-positive")
        return total_number * density / mass

    def _density_std(self, density):
        mass = np.sum(density) * self.dx
        if mass <= self.eps:
            return self.eps
        density_norm = density / mass
        mean = np.sum(self.x * density_norm) * self.dx
        variance = np.sum((self.x - mean) ** 2 * density_norm) * self.dx
        return max(np.sqrt(max(variance, 0.0)), self.eps)

    def _transport(self, density):
        density_edge = 0.5 * (density[:-1] + density[1:])
        fitness_density_edge = 0.5 * (
            self.fitness_grid[:-1] * density[:-1]
            + self.fitness_grid[1:] * density[1:]
        )
        flux = (
            self.force_edge * density_edge
            - self.v * fitness_density_edge
            - self.D * (density[1:] - density[:-1]) / self.dx
        )

        ddt = np.zeros_like(density)
        ddt[0] = -flux[0] / self.dx
        ddt[1:-1] = -(flux[1:] - flux[:-1]) / self.dx
        ddt[-1] = flux[-1] / self.dx
        return ddt

    def _bayesian_source(self, density, tau):
        if tau == 0:
            return np.zeros_like(density)
        total_number = np.sum(density) * self.dx
        if total_number <= self.eps:
            return np.zeros_like(density)

        sigma_x = self._density_std(density)
        likelihood_grid = self._evaluate_likelihood_function(self.x, sigma_x)
        self.likelihood_grid = likelihood_grid.copy()
        evidence = np.sum(likelihood_grid * density) * self.dx
        if evidence <= self.eps:
            return np.zeros_like(density)

        multiplier = tau * (total_number * likelihood_grid / evidence - 1.0)
        return multiplier * density

    def _weighted_mean(self, grid, density):
        mass = np.sum(density) * self.dx
        if mass <= self.eps:
            return np.nan
        return np.sum(grid * density) * self.dx / mass

    def _store_history(self, n_nonexhausted, n_exhausted, step):
        n_total_density = n_nonexhausted + n_exhausted
        N_nonexhausted = np.sum(n_nonexhausted) * self.dx
        N_exhausted = np.sum(n_exhausted) * self.dx
        N_total = N_nonexhausted + N_exhausted

        self.N_nonexhausted_history[step] = N_nonexhausted
        self.N_exhausted_history[step] = N_exhausted
        self.N_total_history[step] = N_total

        self.mean_x_nonexhausted_history[step] = self._weighted_mean(self.x, n_nonexhausted)
        self.mean_x_exhausted_history[step] = self._weighted_mean(self.x, n_exhausted)
        self.mean_x_total_history[step] = self._weighted_mean(self.x, n_total_density)

        self.mean_fitness_nonexhausted_history[step] = self._weighted_mean(self.fitness_grid, n_nonexhausted)
        self.mean_fitness_exhausted_history[step] = self._weighted_mean(self.exhausted_fitness_grid, n_exhausted)
        if N_total > self.eps:
            total_growth = (
                np.sum(self.fitness_grid * n_nonexhausted) * self.dx
                + np.sum(self.exhausted_fitness_grid * n_exhausted) * self.dx
            )
            self.mean_fitness_total_history[step] = total_growth / N_total
        else:
            self.mean_fitness_total_history[step] = np.nan

        self.mean_k3_nonexhausted_history[step] = self._weighted_mean(
            self.k3_nonexhausted_grid,
            n_nonexhausted,
        )
        self.mean_k3_exhausted_history[step] = self.exhausted_k3_value if N_exhausted > self.eps else np.nan

        if N_total > self.eps:
            total_k3 = (
                np.sum(self.k3_nonexhausted_grid * n_nonexhausted) * self.dx
                + self.exhausted_k3_value * N_exhausted
            )
            self.mean_k3_total_history[step] = total_k3 / N_total
        else:
            self.mean_k3_total_history[step] = np.nan

        self.transfer_total_history[step] = np.sum(self.transfer_rate_grid * n_nonexhausted) * self.dx

    def solve(
        self,
        initial_nonexhausted_total=1.0,
        initial_exhausted_total=0.0,
        initial_mean=1.0,
        initial_sigma=0.2,
        initial_exhausted_mean=None,
        initial_exhausted_sigma=None,
        snapshot_steps=None,
        snapshot_times=None,
    ):
        if initial_exhausted_mean is None:
            initial_exhausted_mean = initial_mean
        if initial_exhausted_sigma is None:
            initial_exhausted_sigma = initial_sigma

        n_nonexhausted = self._initial_density(
            initial_nonexhausted_total,
            initial_mean,
            initial_sigma,
        )
        n_exhausted = self._initial_density(
            initial_exhausted_total,
            initial_exhausted_mean,
            initial_exhausted_sigma,
        )

        if snapshot_steps is None:
            snapshot_steps = {} if snapshot_times is not None else {
                0: "Initial",
                self.Nt // 2: "Middle",
                self.Nt: "Final",
            }
        else:
            snapshot_steps = dict(snapshot_steps)

        if snapshot_times is not None:
            for snapshot_time, label in snapshot_times.items():
                step = int(round(float(snapshot_time) / self.dt))
                if step < 0 or step > self.Nt:
                    raise ValueError("snapshot_times must fall within [0, Total_T]")
                snapshot_steps[step] = label

        self.snapshots_nonexhausted = {}
        self.snapshots_exhausted = {}
        self.snapshots_total = {}

        for step in range(self.Nt + 1):
            if step in snapshot_steps:
                label = snapshot_steps[step]
                self.snapshots_nonexhausted[label] = n_nonexhausted.copy()
                self.snapshots_exhausted[label] = n_exhausted.copy()
                self.snapshots_total[label] = n_nonexhausted.copy() + n_exhausted.copy()

            self._store_history(n_nonexhausted, n_exhausted, step)

            if step == self.Nt:
                break

            transfer = self.transfer_rate_grid * n_nonexhausted
            dn_nonexhausted_dt = (
                self._transport(n_nonexhausted)
                + self._bayesian_source(n_nonexhausted, self.tau_nonexhausted)
                + self.fitness_grid * n_nonexhausted
                - transfer
            )
            dn_exhausted_dt = (
                self._transport(n_exhausted)
                + self._bayesian_source(n_exhausted, self.tau_exhausted)
                + self.exhausted_fitness_grid * n_exhausted
                + transfer
            )

            n_nonexhausted = np.clip(n_nonexhausted + self.dt * dn_nonexhausted_dt, 0.0, None)
            n_exhausted = np.clip(n_exhausted + self.dt * dn_exhausted_dt, 0.0, None)

        self.final_nonexhausted = n_nonexhausted.copy()
        self.final_exhausted = n_exhausted.copy()
        return n_nonexhausted, n_exhausted

    def get_history(self):
        return {
            "time": self.times.copy(),
            "N_nonexhausted": self.N_nonexhausted_history.copy(),
            "N_exhausted": self.N_exhausted_history.copy(),
            "N_total": self.N_total_history.copy(),
            "mean_x_nonexhausted": self.mean_x_nonexhausted_history.copy(),
            "mean_x_exhausted": self.mean_x_exhausted_history.copy(),
            "mean_x_total": self.mean_x_total_history.copy(),
            "mean_fitness_nonexhausted": self.mean_fitness_nonexhausted_history.copy(),
            "mean_fitness_exhausted": self.mean_fitness_exhausted_history.copy(),
            "mean_fitness_total": self.mean_fitness_total_history.copy(),
            "mean_k3_nonexhausted": self.mean_k3_nonexhausted_history.copy(),
            "mean_k3_exhausted": self.mean_k3_exhausted_history.copy(),
            "mean_k3_total": self.mean_k3_total_history.copy(),
            "transfer_total": self.transfer_total_history.copy(),
        }

    def p_y_given_x(self):
        return self.likelihood_grid.copy()

    def potential(self):
        return self.potential_grid.copy()

    def fitness(self):
        return self.fitness_grid.copy()

    def exhaustion_rate(self):
        return self.kappa_grid.copy()

    def exhausted_fitness(self):
        return self.exhausted_fitness_grid.copy()

    def transfer_rate(self):
        return self.transfer_rate_grid.copy()

    def k3_nonexhausted(self):
        return self.k3_nonexhausted_grid.copy()

    def k3_exhausted(self):
        return self.k3_exhausted_grid.copy()

    def force(self):
        return -np.gradient(self.potential_grid, self.dx, edge_order=2)




### Below are equation to evolve probablity density #######################################

class BayesianFokkerPlanckSolver1D:
    """
    One-dimensional Bayesian Fokker-Planck solver with reflecting boundaries.

    This version is fully functional: the user directly provides

        likelihood_function(x) = p(y|x)
        or likelihood_function(x, sigma_x) = p(y|x, sigma_x)
        potential_function(x) = U(x)
        fitness_function(x) = f(x)  [optional]
        k3_function(x) = k3(x)      [optional]
        decay_rate_function(x) = d_k(x)  [optional]

    The solver computes the force from the supplied potential:

        F(x) = -dU/dx

    The transport flux is:

        J(x, t) = F(x) p(x, t) + v f(x) p(x, t) - D dp/dx

    so v is a prefactor for the fitness-weighted drift flux.

    The Bayesian source is multiplied by tau, so tau=0 turns it off:

        S_Bayes = tau * ((p(y|x) / p(y)) - 1) * p(x, t)

    An optional replicator source can also be added:

        S_rep = replication_rate * (f(x) - f_0) * p(x, t)

    If fitness_baseline is None, f_0 is the current mean fitness under p.

    If decay_rate_function is provided, k3 evolves during solve as:

        dK3(x, t)/dt = -d_k(x) f(x) K3(x, t)
    """

    def __init__(
        self,
        likelihood_function,
        potential_function,
        D=0.05,
        tau=1.0,
        v=0.0,
        fitness_function=None,
        replication_rate=1.0,
        fitness_baseline=None,
        k3_function=None,
        decay_rate_function=None,
        x_range=(-5.0, 5.0),
        Nx=400,
        dt=1e-4,
        Total_T=1.0,
        eps=1e-14,
    ):
        if not callable(likelihood_function):
            raise TypeError("likelihood_function must be callable")
        if not callable(potential_function):
            raise TypeError("potential_function must be callable")
        if fitness_function is not None and not callable(fitness_function):
            raise TypeError("fitness_function must be callable when provided")
        if k3_function is not None and not callable(k3_function):
            raise TypeError("k3_function must be callable when provided")
        if decay_rate_function is not None and not callable(decay_rate_function):
            raise TypeError("decay_rate_function must be callable when provided")

        self.likelihood_function = likelihood_function
        self.potential_function = potential_function
        self.fitness_function = fitness_function
        self.k3_function = k3_function
        self.decay_rate_function = decay_rate_function
        self.D = float(D)
        self.tau = float(tau)
        self.v = float(v)
        self.replication_rate = float(replication_rate)
        self.fitness_baseline = None if fitness_baseline is None else float(fitness_baseline)
        self.x_min, self.x_max = self._validate_range(x_range, "x_range")
        self.Nx = int(Nx)
        self.dt = float(dt)
        self.T = float(Total_T)
        self.eps = float(eps)
        self.Nt = int(round(self.T / self.dt))

        if self.D < 0:
            raise ValueError("D must be non-negative")
        if self.tau < 0:
            raise ValueError("tau must be non-negative; use tau=0 to turn Bayesian term off")
        if self.replication_rate < 0:
            raise ValueError("replication_rate must be non-negative")
        if self.Nx < 3:
            raise ValueError("Nx must be at least 3")
        if self.dt <= 0 or self.T <= 0:
            raise ValueError("dt and Total_T must be positive")

        self.x = np.linspace(self.x_min, self.x_max, self.Nx)
        self.dx = self.x[1] - self.x[0]
        self.x_edge = 0.5 * (self.x[:-1] + self.x[1:])

        self.likelihood_grid = self._evaluate_likelihood_function(self.x, sigma_x=1.0)
        self.potential_grid = self._evaluate_potential_function(self.x)
        self.fitness_grid = self._evaluate_fitness_function(self.x)
        self.decay_rate_grid = self._evaluate_decay_rate_function(self.x)
        self.initial_k3_grid = self._evaluate_k3_function(self.x)
        self.k3_grid = self.initial_k3_grid.copy()
        self.force_edge = -(self.potential_grid[1:] - self.potential_grid[:-1]) / self.dx

        self.times = np.linspace(0.0, self.Nt * self.dt, self.Nt + 1)
        self.mean_history = np.zeros(self.Nt + 1)
        self.variance_history = np.zeros(self.Nt + 1)
        self.mass_history = np.zeros(self.Nt + 1)
        self.mean_fitness_history = np.zeros(self.Nt + 1)
        self.k3_history = np.zeros(self.Nt + 1)
        self.snapshots = {}

    @staticmethod
    def _validate_range(value_range, name):
        value_range = np.asarray(value_range, dtype=float)
        if value_range.shape != (2,):
            raise ValueError(f"{name} must be a two-number array/list: [min, max]")
        if value_range[0] >= value_range[1]:
            raise ValueError(f"{name}[0] must be smaller than {name}[1]")
        return float(value_range[0]), float(value_range[1])

    @staticmethod
    def _as_grid_values(values, x, name):
        values = np.asarray(values, dtype=float)
        if values.shape == ():
            values = np.full_like(x, float(values), dtype=float)
        if values.shape != x.shape:
            raise ValueError(f"{name} must return a scalar or an array with the same shape as x")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} returned non-finite values")
        return values

    @staticmethod
    def _gaussian_1d(x, mean, sigma):
        return np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))

    def _call_likelihood_function(self, x, sigma_x):
        try:
            return self.likelihood_function(x, sigma_x)
        except TypeError:
            return self.likelihood_function(x)

    def _evaluate_likelihood_function(self, x, sigma_x):
        values = self._as_grid_values(
            self._call_likelihood_function(x, sigma_x),
            x,
            "likelihood_function",
        )
        if np.any(values < 0):
            raise ValueError("likelihood_function must return non-negative values")
        if np.max(values) <= self.eps:
            raise ValueError("likelihood_function is zero everywhere on the grid")
        return values

    def _evaluate_potential_function(self, x):
        return self._as_grid_values(self.potential_function(x), x, "potential_function")

    def _evaluate_fitness_function(self, x):
        if self.fitness_function is None:
            return np.ones_like(x, dtype=float)
        return self._as_grid_values(self.fitness_function(x), x, "fitness_function")

    def _evaluate_k3_function(self, x):
        if self.k3_function is None:
            return np.zeros_like(x, dtype=float)
        return self._as_grid_values(self.k3_function(x), x, "k3_function")

    def _evaluate_decay_rate_function(self, x):
        if self.decay_rate_function is None:
            return np.zeros_like(x, dtype=float)
        values = self._as_grid_values(self.decay_rate_function(x), x, "decay_rate_function")
        if np.any(values < 0):
            raise ValueError("decay_rate_function must return non-negative values")
        return values

    def p_y_given_x(self):
        """Return the user-supplied p(y|x) evaluated on the x grid."""
        return self.likelihood_grid.copy()

    def potential(self):
        """Return the user-supplied U(x) evaluated on the x grid."""
        return self.potential_grid.copy()

    def fitness(self):
        """Return the user-supplied f(x) evaluated on the x grid."""
        return self.fitness_grid.copy()

    def k3(self):
        """Return the current k3(x, t) values on the x grid."""
        return self.k3_grid.copy()

    def decay_rate(self):
        """Return d_k(x), the k3 decay-rate function evaluated on the x grid."""
        return self.decay_rate_grid.copy()

    def force(self):
        """Return F(x) = -dU/dx at cell centers."""
        return -np.gradient(self.potential_grid, self.dx, edge_order=2)

    def _normalize(self, p):
        mass = np.sum(p) * self.dx
        if mass <= 0:
            raise ValueError("density mass became non-positive")
        return p / mass

    def _initial_density(self, initial_mean, initial_sigma):
        initial_sigma = float(initial_sigma)
        if initial_sigma <= 0:
            raise ValueError("initial_sigma must be positive")
        p = self._gaussian_1d(self.x, float(initial_mean), initial_sigma)
        return self._normalize(p)

    def _transport(self, p):
        p_edge = 0.5 * (p[:-1] + p[1:])
        fitness_p_edge = 0.5 * (
            self.fitness_grid[:-1] * p[:-1] + self.fitness_grid[1:] * p[1:]
        )
        J = (
            self.force_edge * p_edge
            - self.v* fitness_p_edge
            - self.D * (p[1:] - p[:-1]) / self.dx
        )

        dpdt = np.zeros_like(p)
        dpdt[0] = -J[0] / self.dx
        dpdt[1:-1] = -(J[1:] - J[:-1]) / self.dx
        dpdt[-1] = J[-1] / self.dx
        return dpdt

    def _density_std(self, p):
        mass = np.sum(p) * self.dx
        if mass <= 0:
            return self.eps
        p_norm = p / mass
        mean = np.sum(self.x * p_norm) * self.dx
        variance = np.sum((self.x - mean) ** 2 * p_norm) * self.dx
        return max(np.sqrt(max(variance, 0.0)), self.eps)

    def _bayes_source(self, p):
        if self.tau == 0:
            return np.zeros_like(p)
        sigma_x = self._density_std(p)
        likelihood_grid = self._evaluate_likelihood_function(self.x, sigma_x)
        self.likelihood_grid = likelihood_grid.copy()
        evidence = np.sum(likelihood_grid * p) * self.dx
        if evidence < self.eps:
            return np.zeros_like(p)
        return self.tau * (likelihood_grid / evidence - 1.0) * p

    def _replicator_source(self, p):
        if self.fitness_function is None or self.replication_rate == 0:
            return np.zeros_like(p)
        if self.fitness_baseline is None:
            f0 = np.sum(self.fitness_grid * p) * self.dx
        else:
            f0 = self.fitness_baseline
        return self.replication_rate * (self.fitness_grid - f0) * p

    def _update_k3_grid(self):
        if self.k3_function is None or self.decay_rate_function is None:
            return
        decay_exponent = -self.dt * self.decay_rate_grid * self.fitness_grid
        self.k3_grid = self.k3_grid * np.exp(decay_exponent)
        self.k3_grid = np.clip(self.k3_grid, 0.0, None)

    def _store_moments(self, p, step):
        mass = np.sum(p) * self.dx
        p_norm = p / mass if mass > 0 else p
        mean = np.sum(self.x * p_norm) * self.dx
        variance = np.sum((self.x - mean) ** 2 * p_norm) * self.dx
        mean_fitness = np.sum(self.fitness_grid * p_norm) * self.dx
        mean_k3 = np.sum(self.k3_grid * p_norm) * self.dx
        self.mass_history[step] = mass
        self.mean_history[step] = mean
        self.variance_history[step] = variance
        self.mean_fitness_history[step] = mean_fitness
        self.k3_history[step] = mean_k3

    def solve(
        self,
        initial_mean=1.0,
        initial_sigma=0.2,
        renorm=True,
        snapshot_steps=None,
        snapshot_times=None,
    ):
        p = self._initial_density(initial_mean, initial_sigma)
        self.k3_grid = self.initial_k3_grid.copy()

        if snapshot_steps is None:
            snapshot_steps = {} if snapshot_times is not None else {
                    0: "Initial",
                    self.Nt // 2: "Middle",
                    self.Nt: "Final",
                }
        else:
            snapshot_steps = dict(snapshot_steps)

        if snapshot_times is not None:
            for snapshot_time, label in snapshot_times.items():
                step = int(round(float(snapshot_time) / self.dt))
                if step < 0 or step > self.Nt:
                    raise ValueError("snapshot_times must fall within [0, Total_T]")
                snapshot_steps[step] = label

        self.snapshots = {}

        for step in range(self.Nt + 1):
            if step in snapshot_steps:
                self.snapshots[snapshot_steps[step]] = p.copy()
            self._store_moments(p, step)

            if step == self.Nt:
                break

            dpdt = self._transport(p) + self._bayes_source(p) + self._replicator_source(p)
            p = p + self.dt * dpdt
            p = np.clip(p, 0.0, None)
            if renorm:
                p = self._normalize(p)
            self._update_k3_grid()

        return p

    def get_moment_history(self):
        return {
            "time": self.times.copy(),
            "mass": self.mass_history.copy(),
            "mean": self.mean_history.copy(),
            "variance": self.variance_history.copy(),
            "mean_fitness": self.mean_fitness_history.copy(),
            "k3": self.k3_history.copy(),
        }

    def plot_model_fields(self):
        nrows = 3
        if self.fitness_function is not None:
            nrows += 1
        if self.k3_function is not None:
            nrows += 1
        if self.decay_rate_function is not None:
            nrows += 1
        fig, axes = plt.subplots(nrows, 1, figsize=(8, 2.6 * nrows), sharex=True, constrained_layout=True)
        axes[0].plot(self.x, self.likelihood_grid, lw=2)
        axes[0].set_ylabel("p(y|x)")
        axes[0].grid(True)

        axes[1].plot(self.x, self.potential_grid, lw=2)
        axes[1].set_ylabel("U(x)")
        axes[1].grid(True)

        axes[2].plot(self.x, self.force(), lw=2)
        axes[2].set_ylabel("F(x)")
        axes[2].set_xlabel("x")
        axes[2].grid(True)

        next_axis = 3
        if self.fitness_function is not None:
            axes[2].set_xlabel("")
            axes[next_axis].plot(self.x, self.fitness_grid, lw=2)
            axes[next_axis].set_ylabel("f(x)")
            axes[next_axis].set_xlabel("x")
            axes[next_axis].grid(True)
            next_axis += 1

        if self.k3_function is not None:
            axes[next_axis - 1].set_xlabel("")
            axes[next_axis].plot(self.x, self.k3_grid, lw=2)
            axes[next_axis].set_ylabel("k3(x)")
            axes[next_axis].set_xlabel("x")
            axes[next_axis].grid(True)
            next_axis += 1

        if self.decay_rate_function is not None:
            axes[next_axis - 1].set_xlabel("")
            axes[next_axis].plot(self.x, self.decay_rate_grid, lw=2)
            axes[next_axis].set_ylabel("d_k(x)")
            axes[next_axis].set_xlabel("x")
            axes[next_axis].grid(True)
        plt.show()

    def plot_distributions(self):
        plt.figure(figsize=(8, 4))
        colors = {"Initial": "gray", "Middle": "steelblue", "Final": "crimson"}
        for key, color in colors.items():
            if key in self.snapshots:
                p = self.snapshots[key]
                plt.plot(self.x, p, color=color, lw=2, label=key)
        plt.xlabel("x")
        plt.ylabel("p(x, t)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_moments(self):
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
        axes[0].plot(self.times, self.mean_history)
        axes[0].set_ylabel("mean")
        axes[0].grid(True)

        axes[1].plot(self.times, self.variance_history)
        axes[1].set_ylabel("variance")
        axes[1].grid(True)

        axes[2].plot(self.times, self.mass_history)
        axes[2].set_ylabel("mass")
        axes[2].set_xlabel("time")
        axes[2].grid(True)
        plt.show()

    def suggest_stable_dt(self, safety=0.2):
        max_speed = np.max(np.abs(self.force() + self.v * self.fitness_grid))

        candidates = []
        if self.D > 0:
            candidates.append(0.5 * self.dx ** 2 / self.D)
        if max_speed > 0:
            candidates.append(self.dx / max_speed)
        if not candidates:
            return np.inf
        return float(safety * min(candidates))


class BayesianFokkerPlanckSolver1DExhaustion(BayesianFokkerPlanckSolver1D):
    """
    One-dimensional Bayesian Fokker-Planck solver with a non-exhausted
    functional sub-density m(x, t).

    The total phenotype density phi(x, t) follows the same dynamics as
    BayesianFokkerPlanckSolver1D. In parallel, m(x, t) follows the same
    transport, Bayesian, and replicator terms, but also loses functional
    potency through

        dm/dt = ... - kappa(x) f(x) m.

    The model keeps phi normalized. The sub-density m is not normalized:
    integral m dx is the non-exhausted fraction of the population.

    The reported mean k3 is

        <k3>(t) = integral k3_function(x) m(x,t) dx
                  + integral exhausted_k3_value * (phi(x,t)-m(x,t)) dx.

    Thus phenotype differentiation changes k3 through k3_function(x), while
    functional exhaustion changes k3 through loss of m(x,t).
    """

    def __init__(
        self,
        likelihood_function,
        potential_function,
        D=0.05,
        tau=1.0,
        v=0.0,
        fitness_function=None,
        replication_rate=1.0,
        fitness_baseline=None,
        k3_function=None,
        exhaustion_rate_function=None,
        exhausted_k3_value=0.001 * 60.0,
        x_range=(-5.0, 5.0),
        Nx=400,
        dt=1e-4,
        Total_T=1.0,
        eps=1e-14,
    ):
        if exhaustion_rate_function is None:
            raise TypeError("exhaustion_rate_function must be provided")
        if not callable(exhaustion_rate_function):
            raise TypeError("exhaustion_rate_function must be callable")

        super().__init__(
            likelihood_function=likelihood_function,
            potential_function=potential_function,
            D=D,
            tau=tau,
            v=v,
            fitness_function=fitness_function,
            replication_rate=replication_rate,
            fitness_baseline=fitness_baseline,
            k3_function=k3_function,
            decay_rate_function=None,
            x_range=x_range,
            Nx=Nx,
            dt=dt,
            Total_T=Total_T,
            eps=eps,
        )

        self.exhaustion_rate_function = exhaustion_rate_function
        self.exhaustion_rate_grid = self._evaluate_exhaustion_rate_function(self.x)
        self.exhausted_k3_value = float(exhausted_k3_value)
        if self.exhausted_k3_value < 0:
            raise ValueError("exhausted_k3_value must be non-negative")

        self.nonexhausted_fraction_history = np.zeros(self.Nt + 1)
        self.exhausted_fraction_history = np.zeros(self.Nt + 1)
        self.k3_nonexhausted_component_history = np.zeros(self.Nt + 1)
        self.k3_exhausted_component_history = np.zeros(self.Nt + 1)
        self.snapshots_m = {}
        self.snapshots_exhausted = {}
        self.final_phi = None
        self.final_m = None

    def _evaluate_exhaustion_rate_function(self, x):
        values = self._as_grid_values(
            self.exhaustion_rate_function(x),
            x,
            "exhaustion_rate_function",
        )
        if np.any(values < 0):
            raise ValueError("exhaustion_rate_function must return non-negative values")
        return values

    def exhaustion_rate(self):
        """Return kappa(x), the functional exhaustion-rate function."""
        return self.exhaustion_rate_grid.copy()

    def local_nonexhausted_fraction(self, p, m):
        """Return a(x,t)=m(x,t)/phi(x,t), with zero where phi is zero."""
        return np.divide(m, p, out=np.zeros_like(m), where=p > self.eps)

    def effective_k3(self, p=None, m=None):
        """
        Return local effective k3(x,t).

        If p and m are omitted, this uses the final densities from the most
        recent solve call.
        """
        if p is None or m is None:
            if self.final_phi is None or self.final_m is None:
                raise ValueError("solve must be called before effective_k3() without p and m")
            p = self.final_phi
            m = self.final_m
        nonexhausted_fraction = self.local_nonexhausted_fraction(p, m)
        return (
            self.k3_grid * nonexhausted_fraction
            + self.exhausted_k3_value * (1.0 - nonexhausted_fraction)
        )

    def _bayesian_multiplier(self, reference_p):
        if self.tau == 0:
            return np.zeros_like(reference_p)
        sigma_x = self._density_std(reference_p)
        likelihood_grid = self._evaluate_likelihood_function(self.x, sigma_x)
        self.likelihood_grid = likelihood_grid.copy()
        evidence = np.sum(likelihood_grid * reference_p) * self.dx
        if evidence < self.eps:
            return np.zeros_like(reference_p)
        return self.tau * (likelihood_grid / evidence - 1.0)

    def _mean_fitness_from_phi(self, p):
        if self.fitness_baseline is None:
            return np.sum(self.fitness_grid * p) * self.dx
        return self.fitness_baseline

    def _replicator_source_with_mean(self, density, mean_fitness):
        if self.fitness_function is None or self.replication_rate == 0:
            return np.zeros_like(density)
        return self.replication_rate * (self.fitness_grid - mean_fitness) * density

    def _functional_exhaustion_source(self, m):
        return -self.exhaustion_rate_grid * self.fitness_grid * m

    def _clip_subdensity(self, p, m):
        m = np.clip(m, 0.0, None)
        return np.minimum(m, p)

    def _initial_nonexhausted_density(self, p, initial_nonexhausted_fraction):
        if callable(initial_nonexhausted_fraction):
            fraction = self._as_grid_values(
                initial_nonexhausted_fraction(self.x),
                self.x,
                "initial_nonexhausted_fraction",
            )
        else:
            fraction = np.full_like(self.x, float(initial_nonexhausted_fraction), dtype=float)
        if np.any(fraction < 0) or np.any(fraction > 1):
            raise ValueError("initial_nonexhausted_fraction must be between 0 and 1")
        return fraction * p

    def _store_moments(self, p, m, step):
        mass = np.sum(p) * self.dx
        p_norm = p / mass if mass > 0 else p
        m = self._clip_subdensity(p_norm, m)
        exhausted_density = p_norm - m

        mean = np.sum(self.x * p_norm) * self.dx
        variance = np.sum((self.x - mean) ** 2 * p_norm) * self.dx
        mean_fitness = np.sum(self.fitness_grid * p_norm) * self.dx

        k3_nonexhausted = np.sum(self.k3_grid * m) * self.dx
        k3_exhausted = self.exhausted_k3_value * np.sum(exhausted_density) * self.dx
        mean_k3 = k3_nonexhausted + k3_exhausted

        self.mass_history[step] = mass
        self.mean_history[step] = mean
        self.variance_history[step] = variance
        self.mean_fitness_history[step] = mean_fitness
        self.k3_history[step] = mean_k3
        self.nonexhausted_fraction_history[step] = np.sum(m) * self.dx
        self.exhausted_fraction_history[step] = np.sum(exhausted_density) * self.dx
        self.k3_nonexhausted_component_history[step] = k3_nonexhausted
        self.k3_exhausted_component_history[step] = k3_exhausted

    def solve(
        self,
        initial_mean=1.0,
        initial_sigma=0.2,
        initial_nonexhausted_fraction=1.0,
        renorm=True,
        snapshot_steps=None,
        snapshot_times=None,
    ):
        p = self._initial_density(initial_mean, initial_sigma)
        m = self._initial_nonexhausted_density(p, initial_nonexhausted_fraction)
        self.k3_grid = self.initial_k3_grid.copy()

        if snapshot_steps is None:
            snapshot_steps = {} if snapshot_times is not None else {
                0: "Initial",
                self.Nt // 2: "Middle",
                self.Nt: "Final",
            }
        else:
            snapshot_steps = dict(snapshot_steps)

        if snapshot_times is not None:
            for snapshot_time, label in snapshot_times.items():
                step = int(round(float(snapshot_time) / self.dt))
                if step < 0 or step > self.Nt:
                    raise ValueError("snapshot_times must fall within [0, Total_T]")
                snapshot_steps[step] = label

        self.snapshots = {}
        self.snapshots_m = {}
        self.snapshots_exhausted = {}

        for step in range(self.Nt + 1):
            if step in snapshot_steps:
                label = snapshot_steps[step]
                self.snapshots[label] = p.copy()
                self.snapshots_m[label] = m.copy()
                self.snapshots_exhausted[label] = np.clip(p - m, 0.0, None)
            self._store_moments(p, m, step)

            if step == self.Nt:
                break

            bayesian_multiplier = self._bayesian_multiplier(p)
            mean_fitness = self._mean_fitness_from_phi(p)

            dpdt = (
                self._transport(p)
                + bayesian_multiplier * p
                + self._replicator_source_with_mean(p, mean_fitness)
            )
            dmdt = (
                self._transport(m)
                + bayesian_multiplier * m
                + self._replicator_source_with_mean(m, mean_fitness)
                + self._functional_exhaustion_source(m)
            )

            p = p + self.dt * dpdt
            m = m + self.dt * dmdt
            p = np.clip(p, 0.0, None)
            if renorm:
                p = self._normalize(p)
            m = self._clip_subdensity(p, m)

        self.final_phi = p.copy()
        self.final_m = m.copy()
        return p, m

    def get_moment_history(self):
        history = super().get_moment_history()
        history.update({
            "nonexhausted_fraction": self.nonexhausted_fraction_history.copy(),
            "exhausted_fraction": self.exhausted_fraction_history.copy(),
            "k3_nonexhausted_component": self.k3_nonexhausted_component_history.copy(),
            "k3_exhausted_component": self.k3_exhausted_component_history.copy(),
        })
        return history


class BayesianFokkerPlanckSolver2D:
    """
    Two-dimensional Bayesian Fokker-Planck solver for p(x1, x2, t).

    The two coordinates have separate potentials:

        U1 = U1(x1),  F1 = -dU1/dx1
        U2 = U2(x2),  F2 = -dU2/dx2

    The Bayesian source acts only through x1:

        S_Bayes = tau * ((p(y|x1) / p(y)) - 1) * p(x1, x2, t)

    The transport equation is

        dp/dt = -dJ1/dx1 - dJ2/dx2 + S_Bayes + S_rep

    with reflecting boundaries and fluxes

        J1 = F1(x1) p - v_x1 f(x1) p - D_x1 dp/dx1
        J2 = F2(x2) p - drift_x2_direction(x1) f(x1) p - D_x2 dp/dx2

    Here x2 has no Bayesian term. Its drift can still depend on x1 through
    drift_x2_direction(x1), and k3 is averaged from a user-provided
    k3_function(x1, x2).
    """

    def __init__(
        self,
        likelihood_function,
        potential_function_x1,
        potential_function_x2,
        fitness_function_x1=None,
        drift_x2_direction=None,
        k3_function=None,
        D_x1=0.05,
        D_x2=0.05,
        tau=1.0,
        v_x1=0.0,
        replication_rate=0.0,
        fitness_baseline=None,
        x1_range=(-5.0, 5.0),
        x2_range=(-5.0, 5.0),
        Nx1=200,
        Nx2=200,
        dt=1e-4,
        Total_T=1.0,
        eps=1e-14,
    ):
        if not callable(likelihood_function):
            raise TypeError("likelihood_function must be callable")
        if not callable(potential_function_x1):
            raise TypeError("potential_function_x1 must be callable")
        if not callable(potential_function_x2):
            raise TypeError("potential_function_x2 must be callable")
        if fitness_function_x1 is not None and not callable(fitness_function_x1):
            raise TypeError("fitness_function_x1 must be callable when provided")
        if drift_x2_direction is not None and not callable(drift_x2_direction):
            raise TypeError("drift_x2_direction must be callable when provided")
        if k3_function is not None and not callable(k3_function):
            raise TypeError("k3_function must be callable when provided")

        self.likelihood_function = likelihood_function
        self.potential_function_x1 = potential_function_x1
        self.potential_function_x2 = potential_function_x2
        self.fitness_function_x1 = fitness_function_x1
        self.drift_x2_direction = drift_x2_direction
        self.k3_function = k3_function

        self.D_x1 = float(D_x1)
        self.D_x2 = float(D_x2)
        self.tau = float(tau)
        self.v_x1 = float(v_x1)
        self.replication_rate = float(replication_rate)
        self.fitness_baseline = None if fitness_baseline is None else float(fitness_baseline)
        self.x1_min, self.x1_max = BayesianFokkerPlanckSolver1D._validate_range(x1_range, "x1_range")
        self.x2_min, self.x2_max = BayesianFokkerPlanckSolver1D._validate_range(x2_range, "x2_range")
        self.Nx1 = int(Nx1)
        self.Nx2 = int(Nx2)
        self.dt = float(dt)
        self.T = float(Total_T)
        self.eps = float(eps)
        self.Nt = int(round(self.T / self.dt))

        if self.D_x1 < 0 or self.D_x2 < 0:
            raise ValueError("D_x1 and D_x2 must be non-negative")
        if self.tau < 0:
            raise ValueError("tau must be non-negative; use tau=0 to turn Bayesian term off")
        if self.replication_rate < 0:
            raise ValueError("replication_rate must be non-negative")
        if self.Nx1 < 3 or self.Nx2 < 3:
            raise ValueError("Nx1 and Nx2 must be at least 3")
        if self.dt <= 0 or self.T <= 0:
            raise ValueError("dt and Total_T must be positive")

        self.x1 = np.linspace(self.x1_min, self.x1_max, self.Nx1)
        self.x2 = np.linspace(self.x2_min, self.x2_max, self.Nx2)
        self.dx1 = self.x1[1] - self.x1[0]
        self.dx2 = self.x2[1] - self.x2[0]
        self.X1, self.X2 = np.meshgrid(self.x1, self.x2, indexing="ij")

        self.likelihood_grid_x1 = self._evaluate_likelihood_function(self.x1, sigma_x1=1.0)
        self.potential_grid_x1 = self._evaluate_1d_function(
            self.potential_function_x1,
            self.x1,
            "potential_function_x1",
        )
        self.potential_grid_x2 = self._evaluate_1d_function(
            self.potential_function_x2,
            self.x2,
            "potential_function_x2",
        )
        self.fitness_grid_x1 = self._evaluate_fitness_function_x1(self.x1)
        self.drift_x2_grid = self._evaluate_drift_x2_direction(self.X1)
        self.k3_grid = self._evaluate_k3_function(self.X1, self.X2)

        self.force_x1_edge = -(self.potential_grid_x1[1:] - self.potential_grid_x1[:-1]) / self.dx1
        self.force_x2_edge = -(self.potential_grid_x2[1:] - self.potential_grid_x2[:-1]) / self.dx2

        self.times = np.linspace(0.0, self.Nt * self.dt, self.Nt + 1)
        self.mass_history = np.zeros(self.Nt + 1)
        self.mean_x1_history = np.zeros(self.Nt + 1)
        self.mean_x2_history = np.zeros(self.Nt + 1)
        self.variance_x1_history = np.zeros(self.Nt + 1)
        self.variance_x2_history = np.zeros(self.Nt + 1)
        self.mean_fitness_history = np.zeros(self.Nt + 1)
        self.k3_history = np.zeros(self.Nt + 1)
        self.snapshots = {}

    @staticmethod
    def _as_1d_grid_values(values, x, name):
        values = np.asarray(values, dtype=float)
        if values.shape == ():
            values = np.full_like(x, float(values), dtype=float)
        if values.shape != x.shape:
            raise ValueError(f"{name} must return a scalar or an array with shape {x.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} returned non-finite values")
        return values

    @staticmethod
    def _as_2d_grid_values(values, grid_shape, name):
        values = np.asarray(values, dtype=float)
        if values.shape == ():
            values = np.full(grid_shape, float(values), dtype=float)
        if values.shape != grid_shape:
            raise ValueError(f"{name} must return a scalar or an array with shape {grid_shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} returned non-finite values")
        return values

    def _evaluate_1d_function(self, function, x, name):
        return self._as_1d_grid_values(function(x), x, name)

    def _call_likelihood_function(self, x1, sigma_x1):
        try:
            return self.likelihood_function(x1, sigma_x1)
        except TypeError:
            return self.likelihood_function(x1)

    def _evaluate_likelihood_function(self, x1, sigma_x1):
        values = self._as_1d_grid_values(
            self._call_likelihood_function(x1, sigma_x1),
            x1,
            "likelihood_function",
        )
        if np.any(values < 0):
            raise ValueError("likelihood_function must return non-negative values")
        if np.max(values) <= self.eps:
            raise ValueError("likelihood_function is zero everywhere on the x1 grid")
        return values

    def _evaluate_fitness_function_x1(self, x1):
        if self.fitness_function_x1 is None:
            return np.ones_like(x1, dtype=float)
        return self._as_1d_grid_values(self.fitness_function_x1(x1), x1, "fitness_function_x1")

    def _evaluate_drift_x2_direction(self, X1):
        if self.drift_x2_direction is None:
            return np.zeros_like(X1, dtype=float)
        values = self.drift_x2_direction(X1)
        return self._as_2d_grid_values(values, X1.shape, "drift_x2_direction")

    def _evaluate_k3_function(self, X1, X2):
        if self.k3_function is None:
            return np.zeros_like(X1, dtype=float)
        values = self.k3_function(X1, X2)
        return self._as_2d_grid_values(values, X1.shape, "k3_function")

    @staticmethod
    def _gaussian_2d_product(X1, X2, mean_x1, sigma_x1, mean_x2, sigma_x2):
        p_x1 = np.exp(-0.5 * ((X1 - mean_x1) / sigma_x1) ** 2) / (sigma_x1 * np.sqrt(2.0 * np.pi))
        p_x2 = np.exp(-0.5 * ((X2 - mean_x2) / sigma_x2) ** 2) / (sigma_x2 * np.sqrt(2.0 * np.pi))
        return p_x1 * p_x2

    def p_y_given_x1(self):
        return self.likelihood_grid_x1.copy()

    def potential_x1(self):
        return self.potential_grid_x1.copy()

    def potential_x2(self):
        return self.potential_grid_x2.copy()

    def fitness_x1(self):
        return self.fitness_grid_x1.copy()

    def drift_x2(self):
        return self.drift_x2_grid.copy()

    def k3(self):
        return self.k3_grid.copy()

    def force_x1(self):
        return -np.gradient(self.potential_grid_x1, self.dx1, edge_order=2)

    def force_x2(self):
        return -np.gradient(self.potential_grid_x2, self.dx2, edge_order=2)

    def _normalize(self, p):
        mass = np.sum(p) * self.dx1 * self.dx2
        if mass <= 0:
            raise ValueError("density mass became non-positive")
        return p / mass

    def _initial_density(self, initial_mean_x1, initial_sigma_x1, initial_mean_x2, initial_sigma_x2):
        initial_sigma_x1 = float(initial_sigma_x1)
        initial_sigma_x2 = float(initial_sigma_x2)
        if initial_sigma_x1 <= 0 or initial_sigma_x2 <= 0:
            raise ValueError("initial_sigma_x1 and initial_sigma_x2 must be positive")
        p = self._gaussian_2d_product(
            self.X1,
            self.X2,
            float(initial_mean_x1),
            initial_sigma_x1,
            float(initial_mean_x2),
            initial_sigma_x2,
        )
        return self._normalize(p)

    def _density_std_x1(self, p):
        mass = np.sum(p) * self.dx1 * self.dx2
        if mass <= 0:
            return self.eps
        p_norm = p / mass
        mean_x1 = np.sum(self.X1 * p_norm) * self.dx1 * self.dx2
        variance_x1 = np.sum((self.X1 - mean_x1) ** 2 * p_norm) * self.dx1 * self.dx2
        return max(np.sqrt(max(variance_x1, 0.0)), self.eps)

    def _transport_x1(self, p):
        p_edge = 0.5 * (p[:-1, :] + p[1:, :])
        fitness_p_edge = 0.5 * (
            self.fitness_grid_x1[:-1, None] * p[:-1, :]
            + self.fitness_grid_x1[1:, None] * p[1:, :]
        )
        J1 = (
            self.force_x1_edge[:, None] * p_edge
            - self.v_x1 * fitness_p_edge
            - self.D_x1 * (p[1:, :] - p[:-1, :]) / self.dx1
        )

        dpdt = np.zeros_like(p)
        dpdt[0, :] = -J1[0, :] / self.dx1
        dpdt[1:-1, :] = -(J1[1:, :] - J1[:-1, :]) / self.dx1
        dpdt[-1, :] = J1[-1, :] / self.dx1
        return dpdt

    def _transport_x2(self, p):
        p_edge = 0.5 * (p[:, :-1] + p[:, 1:])
        drift_fitness_grid = self.drift_x2_grid * self.fitness_grid_x1[:, None]
        drift_p_edge = 0.5 * (
            drift_fitness_grid[:, :-1] * p[:, :-1]
            + drift_fitness_grid[:, 1:] * p[:, 1:]
        )
        J2 = (
            self.force_x2_edge[None, :] * p_edge
            - drift_p_edge
            - self.D_x2 * (p[:, 1:] - p[:, :-1]) / self.dx2
        )

        dpdt = np.zeros_like(p)
        dpdt[:, 0] = -J2[:, 0] / self.dx2
        dpdt[:, 1:-1] = -(J2[:, 1:] - J2[:, :-1]) / self.dx2
        dpdt[:, -1] = J2[:, -1] / self.dx2
        return dpdt

    def _transport(self, p):
        return self._transport_x1(p) + self._transport_x2(p)

    def _bayes_source(self, p):
        if self.tau == 0:
            return np.zeros_like(p)
        sigma_x1 = self._density_std_x1(p)
        likelihood_grid_x1 = self._evaluate_likelihood_function(self.x1, sigma_x1)
        self.likelihood_grid_x1 = likelihood_grid_x1.copy()
        evidence = np.sum(likelihood_grid_x1[:, None] * p) * self.dx1 * self.dx2
        if evidence < self.eps:
            return np.zeros_like(p)
        return self.tau * (likelihood_grid_x1[:, None] / evidence - 1.0) * p

    def _replicator_source(self, p):
        if self.fitness_function_x1 is None or self.replication_rate == 0:
            return np.zeros_like(p)
        if self.fitness_baseline is None:
            f0 = np.sum(self.fitness_grid_x1[:, None] * p) * self.dx1 * self.dx2
        else:
            f0 = self.fitness_baseline
        return self.replication_rate * (self.fitness_grid_x1[:, None] - f0) * p

    def _store_moments(self, p, step):
        mass = np.sum(p) * self.dx1 * self.dx2
        p_norm = p / mass if mass > 0 else p
        mean_x1 = np.sum(self.X1 * p_norm) * self.dx1 * self.dx2
        mean_x2 = np.sum(self.X2 * p_norm) * self.dx1 * self.dx2
        variance_x1 = np.sum((self.X1 - mean_x1) ** 2 * p_norm) * self.dx1 * self.dx2
        variance_x2 = np.sum((self.X2 - mean_x2) ** 2 * p_norm) * self.dx1 * self.dx2
        mean_fitness = np.sum(self.fitness_grid_x1[:, None] * p_norm) * self.dx1 * self.dx2
        mean_k3 = np.sum(self.k3_grid * p_norm) * self.dx1 * self.dx2

        self.mass_history[step] = mass
        self.mean_x1_history[step] = mean_x1
        self.mean_x2_history[step] = mean_x2
        self.variance_x1_history[step] = variance_x1
        self.variance_x2_history[step] = variance_x2
        self.mean_fitness_history[step] = mean_fitness
        self.k3_history[step] = mean_k3

    def solve(
        self,
        initial_mean_x1=0.0,
        initial_sigma_x1=0.2,
        initial_mean_x2=1.0,
        initial_sigma_x2=0.2,
        renorm=True,
        snapshot_steps=None,
        snapshot_times=None,
    ):
        p = self._initial_density(
            initial_mean_x1,
            initial_sigma_x1,
            initial_mean_x2,
            initial_sigma_x2,
        )

        if snapshot_steps is None:
            snapshot_steps = {} if snapshot_times is not None else {
                0: "Initial",
                self.Nt // 2: "Middle",
                self.Nt: "Final",
            }
        else:
            snapshot_steps = dict(snapshot_steps)

        if snapshot_times is not None:
            for snapshot_time, label in snapshot_times.items():
                step = int(round(float(snapshot_time) / self.dt))
                if step < 0 or step > self.Nt:
                    raise ValueError("snapshot_times must fall within [0, Total_T]")
                snapshot_steps[step] = label

        self.snapshots = {}

        for step in range(self.Nt + 1):
            if step in snapshot_steps:
                self.snapshots[snapshot_steps[step]] = p.copy()
            self._store_moments(p, step)

            if step == self.Nt:
                break

            dpdt = self._transport(p) + self._bayes_source(p) + self._replicator_source(p)
            p = p + self.dt * dpdt
            p = np.clip(p, 0.0, None)
            if renorm:
                p = self._normalize(p)

        return p

    def get_moment_history(self):
        return {
            "time": self.times.copy(),
            "mass": self.mass_history.copy(),
            "mean_x1": self.mean_x1_history.copy(),
            "mean_x2": self.mean_x2_history.copy(),
            "variance_x1": self.variance_x1_history.copy(),
            "variance_x2": self.variance_x2_history.copy(),
            "mean_fitness": self.mean_fitness_history.copy(),
            "k3": self.k3_history.copy(),
        }

    def marginal_x1(self, p):
        return np.sum(p, axis=1) * self.dx2

    def marginal_x2(self, p):
        return np.sum(p, axis=0) * self.dx1

    def suggest_stable_dt(self, safety=0.2):
        max_speed_x1 = np.max(np.abs(self.force_x1() + self.v_x1 * self.fitness_grid_x1))
        max_speed_x2 = np.max(np.abs(self.force_x2())) + np.max(
            np.abs(self.drift_x2_grid * self.fitness_grid_x1[:, None])
        )

        candidates = []
        if self.D_x1 > 0:
            candidates.append(0.5 * self.dx1 ** 2 / self.D_x1)
        if self.D_x2 > 0:
            candidates.append(0.5 * self.dx2 ** 2 / self.D_x2)
        if max_speed_x1 > 0:
            candidates.append(self.dx1 / max_speed_x1)
        if max_speed_x2 > 0:
            candidates.append(self.dx2 / max_speed_x2)
        if not candidates:
            return np.inf
        return float(safety * min(candidates))


# Alias requested in the project notes/user wording.
bayesianfokkerplack1dexhaustion = BayesianFokkerPlanckSolver1DExhaustion
bayesianfokkerplack2d = BayesianFokkerPlanckSolver2D


