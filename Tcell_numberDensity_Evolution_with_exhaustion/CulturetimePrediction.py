from pathlib import Path
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mylib import (
    BayesianDensityEvolution1D,
    double_well_potential_function,
    make_bayesian_likelihood_1d,
    make_exhaustion_rate_function,
    make_fitness_function,
    relation_function,
)


# ======================================================================
# chosen parameters
# ======================================================================

x_min, x_max = -4.0, 4.0
nx = int(os.environ.get("DENSITY_NX", 500))
fp_dt = float(os.environ.get("DENSITY_DT", 0.01))
total_days = int(os.environ.get("DENSITY_TOTAL_DAYS", 7))
total_hours = 24 * total_days

x_star = 1.0
x0, sigma0 = 1.0, 0.2
initial_nonexhausted_total = 1.0
initial_exhausted_total = 0.0
sigma_y_given_x = 1.0

fitness_left = 0.02
fitness_right = 0.04
exhausted_k3_value = 0.001 * 60
k3_left = 0.001 * 60
k3_right = 0.01 * 60

k1 = 1e-5 * 60
k2 = 2e-3 * 60
r_contour_value = 1.0

# Choose any one Bayesian model parameter to vary.
# Available numeric names:
#   "delta", "rho", "D", "potential_strength", "tau", "tau_nonexhausted",
#   "tau_exhausted", "v", "kappa_max", "exhausted_fixed_fitness"
vary_parameter = "delta"
vary_values = np.linspace(0, 0.4, 5)
vary_axis_log_scale = False

allowed_vary_parameters = {
    "delta", "rho", "D", "potential_strength", "tau", "tau_nonexhausted",
    "tau_exhausted", "v", "kappa_max", "exhausted_fixed_fitness"
}
if vary_parameter not in allowed_vary_parameters:
    raise ValueError(f"vary_parameter must be one of {sorted(allowed_vary_parameters)}")

fixed_parameters = {
    "delta": 0.1,
    "rho": 0.085,
    "D": 0.01,
    "potential_strength": 0.01,
    "tau": 6.0,
    "tau_nonexhausted": 6.0,
    "tau_exhausted": 0.0,
    "v": 0.0,
    "kappa_max": 0.2,
    "exhausted_fitness_mode": "fixed",
    "exhausted_fixed_fitness": 0.01*0.5,
}

use_diffusion_as_potential_strength = True

if vary_axis_log_scale:
    if np.any(vary_values <= 0):
        raise ValueError("vary_values must be positive when vary_axis_log_scale is True")
    vary_values = np.geomspace(vary_values[0], vary_values[-1], len(vary_values))

vary_count_override = os.environ.get("DENSITY_CULTURE_VARY_COUNT")
if vary_count_override is not None:
    vary_count_override = int(vary_count_override)
    if vary_count_override < 2:
        raise ValueError("DENSITY_CULTURE_VARY_COUNT must be at least 2")
    if vary_axis_log_scale:
        vary_values = np.geomspace(vary_values[0], vary_values[-1], vary_count_override)
    else:
        vary_values = np.linspace(vary_values[0], vary_values[-1], vary_count_override)

day_hours = np.arange(24, total_hours + 1, 24)
day_values = day_hours / 24.0


# ======================================================================
# model functions
# ======================================================================

fitness_function = make_fitness_function(fitness_left, fitness_right, mode="hill", steepness=5.0)


def k3_nonexhausted_function(x):
    return k3_left + (k3_right - k3_left) / (1.0 + np.exp(-5.0 * x))


# ======================================================================
# simulation
# ======================================================================

def make_parameters(vary_value):
    parameters = fixed_parameters.copy()
    parameters[vary_parameter] = float(vary_value)
    if use_diffusion_as_potential_strength and vary_parameter != "potential_strength":
        parameters["potential_strength"] = parameters["D"]
    return parameters


def simulate(parameters):
    solver = BayesianDensityEvolution1D(
        likelihood_function=make_bayesian_likelihood_1d(
            parameters["rho"],
            parameters["delta"],
            x_star=x_star,
            sigma_y_given_x=sigma_y_given_x,
        ),
        potential_function=lambda x: double_well_potential_function(x, parameters["potential_strength"]),
        fitness_function=fitness_function,
        exhaustion_rate_function=make_exhaustion_rate_function(parameters["kappa_max"], steepness=5.0),
        k3_nonexhausted_function=k3_nonexhausted_function,
        exhausted_k3_value=exhausted_k3_value,
        D=parameters["D"],
        tau=parameters["tau"],
        tau_nonexhausted=parameters["tau_nonexhausted"],
        tau_exhausted=parameters["tau_exhausted"],
        v=parameters["v"],
        exhausted_fitness_mode=parameters["exhausted_fitness_mode"],
        exhausted_fixed_fitness=parameters["exhausted_fixed_fitness"],
        x_range=(x_min, x_max),
        Nx=nx,
        dt=fp_dt,
        Total_T=total_hours,
    )
    solver.solve(
        initial_nonexhausted_total=initial_nonexhausted_total,
        initial_exhausted_total=initial_exhausted_total,
        initial_mean=x0,
        initial_sigma=sigma0,
    )
    history = solver.get_history()

    time = np.asarray(history["time"])
    mean = np.asarray(history["mean_x_total"])
    mean_fitness = np.asarray(history["mean_fitness_total"])
    n_total = np.asarray(history["N_total"])
    n_nonexhausted = np.asarray(history["N_nonexhausted"])
    n_exhausted = np.asarray(history["N_exhausted"])
    k3_history = np.asarray(history["mean_k3_total"])
    r_history = relation_function(k1, k2, k3_history, n_total)

    return {
        "time": time,
        "mean": mean,
        "mean_fitness": mean_fitness,
        "N_total": n_total,
        "N_nonexhausted": n_nonexhausted,
        "N_exhausted": n_exhausted,
        "k3": k3_history,
        "r": r_history,
    }


def values_at_days(time, values):
    return np.asarray([
        values[int(round(hour * (len(time) - 1) / time[-1]))]
        for hour in day_hours
    ])


def cell_edges(values, log_scale=False):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        raise ValueError("at least two values are needed for a phase diagram")
    if log_scale:
        if np.any(values <= 0):
            raise ValueError("log-scale cell edges require positive values")
        log_values = np.log(values)
        midpoints = 0.5 * (log_values[:-1] + log_values[1:])
        first_edge = log_values[0] - (midpoints[0] - log_values[0])
        last_edge = log_values[-1] + (log_values[-1] - midpoints[-1])
        return np.exp(np.concatenate([[first_edge], midpoints, [last_edge]]))
    midpoints = 0.5 * (values[:-1] + values[1:])
    first_edge = values[0] - (midpoints[0] - values[0])
    last_edge = values[-1] + (values[-1] - midpoints[-1])
    return np.concatenate([[first_edge], midpoints, [last_edge]])


def format_tick_labels(values):
    return [f"{value:.3g}" for value in values]


def display_parameter_name(name):
    if name == "delta":
        return r"$\delta$"
    if name == "rho":
        return r"$\rho$"
    if name == "tau":
        return r"$\tau$"
    if name == "tau_nonexhausted":
        return r"$\tau_{nonexhausted}$"
    if name == "tau_exhausted":
        return r"$\tau_{exhausted}$"
    if name == "exhausted_fixed_fitness":
        return r"$f_{exhausted}$"
    if name == "kappa_max":
        return r"$\kappa_{\max}$"
    return name


mean_grid = np.zeros((len(day_hours), len(vary_values)))
n_total_grid = np.zeros_like(mean_grid)
n_exhausted_grid = np.zeros_like(mean_grid)
k3_grid = np.zeros_like(mean_grid)
r_grid = np.zeros_like(mean_grid)

for vary_index, vary_value in enumerate(vary_values):
    result = simulate(make_parameters(vary_value))
    mean_grid[:, vary_index] = values_at_days(result["time"], result["mean"])
    n_total_grid[:, vary_index] = values_at_days(result["time"], result["N_total"])
    n_exhausted_grid[:, vary_index] = values_at_days(result["time"], result["N_exhausted"])
    k3_grid[:, vary_index] = values_at_days(result["time"], result["k3"])
    r_grid[:, vary_index] = values_at_days(result["time"], result["r"])

    print(
        f"{vary_parameter}={vary_value:.5g}: "
        f"mean(day {total_days})={mean_grid[-1, vary_index]:+.4f}, "
        f"N_total(day {total_days})={n_total_grid[-1, vary_index]:.4g}, "
        f"N_exhausted(day {total_days})={n_exhausted_grid[-1, vary_index]:.4g}, "
        f"k3(day {total_days})={k3_grid[-1, vary_index]:.5f}, "
        f"r(day {total_days})={r_grid[-1, vary_index]:.5f}"
    )


# ======================================================================
# phase diagrams
# ======================================================================

vary_edges = cell_edges(vary_values, log_scale=vary_axis_log_scale)
day_edges = cell_edges(day_values, log_scale=False)
vary_mesh, day_mesh = np.meshgrid(vary_values, day_values)

fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=False, constrained_layout=True)

panels = [
    (axes[0, 0], mean_grid, "mean phenotype", "mean phenotype"),
    (axes[0, 1], n_total_grid, "total cell number N", "N"),
    (axes[1, 0], k3_grid, r"$\langle k_3\rangle$", r"$\langle k_3\rangle$"),
    (axes[1, 1], r_grid, r"$r = k_1 \langle k_3\rangle N/(k_2+\langle k_3\rangle)$", "r"),
]
r_axis = axes[1, 1]

for ax, grid, title, colorbar_label in panels:
    mesh = ax.pcolormesh(
        vary_edges,
        day_edges,
        grid,
        shading="flat",
        cmap="viridis",
        edgecolors="none",
        linewidth=0,
    )
    fig.colorbar(mesh, ax=ax, label=colorbar_label)
    ax.set_title(title)
    ax.set_xlabel(display_parameter_name(vary_parameter))
    ax.set_ylabel("time (days)")
    if vary_axis_log_scale:
        ax.set_xscale("log")
    ax.set_xticks(vary_values)
    ax.set_xticklabels(format_tick_labels(vary_values), rotation=45, ha="right")
    ax.set_yticks(day_values)
    ax.set_yticklabels(format_tick_labels(day_values))
    ax.grid(False)

if np.nanmin(r_grid) <= r_contour_value <= np.nanmax(r_grid):
    contour = r_axis.contour(
        vary_mesh,
        day_mesh,
        r_grid,
        levels=[r_contour_value],
        colors="white",
        linestyles=":",
        linewidths=2.5,
    )

fig.suptitle("culture-time prediction from Bayesian density evolution model")
plt.show()

