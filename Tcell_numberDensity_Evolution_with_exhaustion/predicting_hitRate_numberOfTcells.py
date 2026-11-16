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
)


# ======================================================================
# choose what to vary
# ======================================================================

# Available numeric parameter names:
#   "D", "rho", "tau", "tau_nonexhausted", "tau_exhausted", "v",
#   "kappa_max", "delta", "exhausted_fixed_fitness"
parameter_1 = "rho"
parameter_2 = "delta"

parameter_1_values = np.linspace(0.01, 0.1, 7)
parameter_2_values = np.linspace(0.0, 0.5, 7)
x_axis_log_scale = False
k3_threshold = 0.3
k3_relative_threshold = 0.5
fold_change_threshold = 8

if x_axis_log_scale:
    if np.any(parameter_1_values <= 0):
        raise ValueError("parameter_1_values must be positive when x_axis_log_scale is True")
    parameter_1_values = np.geomspace(parameter_1_values[0], parameter_1_values[-1], len(parameter_1_values))

grid_count_override = os.environ.get("DENSITY_PHASE_GRID_COUNT") or os.environ.get("K3_PHASE_GRID_COUNT")
if grid_count_override is not None:
    grid_count_override = int(grid_count_override)
    if grid_count_override < 2:
        raise ValueError("grid count override must be at least 2")
    if x_axis_log_scale:
        parameter_1_values = np.geomspace(parameter_1_values[0], parameter_1_values[-1], grid_count_override)
    else:
        parameter_1_values = np.linspace(parameter_1_values[0], parameter_1_values[-1], grid_count_override)
    parameter_2_values = np.linspace(parameter_2_values[0], parameter_2_values[-1], grid_count_override)

bayesian_parameters = {
    "D": 0.01,
    "potential_strength": 0.01,
    "rho": 0.085,
    "tau": 6.0,
    "tau_nonexhausted": 6.0,
    "tau_exhausted": 0.0,
    "v": 0.0,
    "delta": 0.07,
    "kappa_max": 0.2,
    "exhausted_fitness_mode": "fixed",
    "exhausted_fixed_fitness": 0.01*0.5,
}

use_diffusion_as_potential_strength = True


# ======================================================================
# numerical setup
# ======================================================================

x_min, x_max = -4.0, 4.0
nx = int(os.environ.get("DENSITY_NX", 500))
fp_dt = float(os.environ.get("DENSITY_DT", 0.01))
total_days = int(os.environ.get("DENSITY_TOTAL_DAYS", 14))
total_hours = 24 * total_days

x_star = 1.0
x0, sigma0 = 1.0, 0.1
initial_nonexhausted_total = 1.0
initial_exhausted_total = 0.0
sigma_y_given_x = 1.0

h1 = 24 * 1
h3 = 24 * 3


# ======================================================================
# model functions
# ======================================================================

fitness_left = 0.02
fitness_right = 0.04
exhausted_k3_value = 0.001 * 60
k3_left = 0.001 * 60
k3_right = 0.01 * 60

fitness_function = make_fitness_function(fitness_left, fitness_right, mode="hill", steepness=5.0)


def k3_nonexhausted_function(x):
    return k3_left + (k3_right - k3_left) / (1.0 + np.exp(-5.0 * x))


# ======================================================================
# simulation helpers
# ======================================================================

def value_at(time, values, hour):
    idx = int(round(hour * (len(time) - 1) / time[-1]))
    return values[idx]


def value_at_or_nan(time, values, hour):
    if hour > time[-1]:
        return np.nan
    return value_at(time, values, hour)


def first_reduction_time(time, values, fraction):
    initial_value = values[0]
    threshold = fraction * initial_value
    crossing_indices = np.where(values <= threshold)[0]
    crossing_indices = crossing_indices[crossing_indices > 0]
    if len(crossing_indices) == 0:
        return np.nan

    index = crossing_indices[0]
    previous_index = index - 1
    previous_value = values[previous_index]
    current_value = values[index]
    previous_time = time[previous_index]
    current_time = time[index]

    if np.isclose(previous_value, current_value):
        return current_time / 24.0

    fraction_between_steps = (threshold - previous_value) / (current_value - previous_value)
    return (previous_time + fraction_between_steps * (current_time - previous_time)) / 24.0


def simulate(parameters):
    parameters = parameters.copy()
    if use_diffusion_as_potential_strength:
        parameters["potential_strength"] = parameters["D"]

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
    k3_history = np.asarray(history["mean_k3_total"])
    k3_initial = max(k3_history[0], 1e-12)
    k3_relative_history = k3_history / k3_initial
    n_total_history = np.asarray(history["N_total"])
    fold_change_history = n_total_history / max(n_total_history[0], 1e-12)
    print("Intial number of T cells=",n_total_history[0])
    half_time = first_reduction_time(time, k3_history, 0.5)
    quarter_time = first_reduction_time(time, k3_history, 0.25)
    second_half_time = quarter_time - half_time
    if not np.isfinite(half_time) or half_time <= 0 or not np.isfinite(second_half_time):
        half_time_ratio = np.nan
    else:
        half_time_ratio = second_half_time / half_time

    return {
        "k3_day1": value_at(time, k3_history, h1),
        "k3_day3": value_at_or_nan(time, k3_history, h3),
        "k3_last_day": value_at(time, k3_history, total_hours),
        "k3_relative_day1": value_at_or_nan(time, k3_relative_history, h1),
        "k3_relative_day3": value_at_or_nan(time, k3_relative_history, h3),
        "k3_relative_last_day": value_at(time, k3_relative_history, total_hours),
        "fold_change_day1": value_at_or_nan(time, fold_change_history, h1),
        "fold_change_day3": value_at_or_nan(time, fold_change_history, h3),
        "exhausted_fraction_last_day": value_at(time, history["N_exhausted"], total_hours) / max(value_at(time, history["N_total"], total_hours), 1e-12),
        "half_time": half_time,
        "second_half_time": second_half_time,
        "half_time_ratio": half_time_ratio,
    }


def make_parameters(parameter_1_value, parameter_2_value):
    parameters = bayesian_parameters.copy()
    parameters[parameter_1] = float(parameter_1_value)
    parameters[parameter_2] = float(parameter_2_value)
    return parameters


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


# ======================================================================
# run phase diagram
# ======================================================================

if parameter_1 == parameter_2:
    raise ValueError("parameter_1 and parameter_2 must be different")

k3_day1_grid = np.zeros((len(parameter_2_values), len(parameter_1_values)))
k3_day3_grid = np.zeros_like(k3_day1_grid)
k3_last_day_grid = np.zeros_like(k3_day1_grid)
k3_relative_day1_grid = np.zeros_like(k3_day1_grid)
k3_relative_day3_grid = np.zeros_like(k3_day1_grid)
k3_relative_last_day_grid = np.zeros_like(k3_day1_grid)
exhausted_fraction_last_day_grid = np.zeros_like(k3_day1_grid)
fold_change_day1_grid = np.zeros_like(k3_day1_grid)
fold_change_day3_grid = np.zeros_like(k3_day1_grid)
half_time_grid = np.zeros_like(k3_day1_grid)
second_half_time_grid = np.zeros_like(k3_day1_grid)
half_time_ratio_grid = np.zeros_like(k3_day1_grid)

total_runs = len(parameter_1_values) * len(parameter_2_values)
run_index = 0

for row, parameter_2_value in enumerate(parameter_2_values):
    for col, parameter_1_value in enumerate(parameter_1_values):
        run_index += 1
        output = simulate(make_parameters(parameter_1_value, parameter_2_value))
        k3_day1_grid[row, col] = output["k3_day1"]
        k3_day3_grid[row, col] = output["k3_day3"]
        k3_last_day_grid[row, col] = output["k3_last_day"]
        k3_relative_day1_grid[row, col] = output["k3_relative_day1"]
        k3_relative_day3_grid[row, col] = output["k3_relative_day3"]
        k3_relative_last_day_grid[row, col] = output["k3_relative_last_day"]
        fold_change_day1_grid[row, col] = output["fold_change_day1"]
        fold_change_day3_grid[row, col] = output["fold_change_day3"]
        exhausted_fraction_last_day_grid[row, col] = output["exhausted_fraction_last_day"]
        half_time_grid[row, col] = output["half_time"]
        second_half_time_grid[row, col] = output["second_half_time"]
        half_time_ratio_grid[row, col] = output["half_time_ratio"]
        print(f"completed {run_index:4d}/{total_runs} simulations", end="\r")
print()


# ======================================================================
# plot helpers
# ======================================================================

x_edges = cell_edges(parameter_1_values, log_scale=x_axis_log_scale)
y_edges = cell_edges(parameter_2_values, log_scale=False)
x_grid, y_grid = np.meshgrid(parameter_1_values, parameter_2_values)


def add_threshold_contour(ax, grid, threshold, label):
    finite_values = grid[np.isfinite(grid)]
    if (
        len(finite_values) > 0
        and np.nanmin(finite_values) <= threshold <= np.nanmax(finite_values)
        and len(np.unique(np.round(finite_values, 12))) > 1
    ):
        contour = ax.contour(
            x_grid,
            y_grid,
            grid,
            levels=[threshold],
            colors="white",
            linestyles=":",
            linewidths=2.5,
        )
        ax.clabel(contour, fmt={threshold: label}, inline=True, fontsize=12, colors="white")


def draw_block_panel(ax, grid, title, colorbar_label, cmap="viridis", threshold=None, threshold_label=None,vmin=None,
    vmax=None):
    masked_grid = np.ma.masked_invalid(grid)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        masked_grid,
        shading="flat",
        cmap=cmap,
        edgecolors="none",
        linewidth=0,
        vmin=vmin,
        vmax=vmax,
    )
    plt.gcf().colorbar(mesh, ax=ax, label=colorbar_label)
    if threshold is not None:
        add_threshold_contour(ax, grid, threshold, threshold_label or f"{threshold:g}")
    ax.set_xlabel(display_parameter_name(parameter_1))
    ax.set_ylabel(display_parameter_name(parameter_2))
    ax.set_title(title)
    if x_axis_log_scale:
        ax.set_xscale("log")
    ax.set_xticks(parameter_1_values)
    ax.set_xticklabels(format_tick_labels(parameter_1_values), rotation=45, ha="right")
    ax.set_yticks(parameter_2_values)
    ax.set_yticklabels(format_tick_labels(parameter_2_values))
    ax.grid(False)


# ======================================================================
# plot mean k3 and reduction phase diagrams
# ======================================================================

fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
axes = axes.ravel()

plots = [
    (axes[0], k3_day1_grid, r"$\langle k_3\rangle$ at day 1", r"$\langle k_3\rangle$", "viridis", k3_threshold, rf"$\langle k_3\rangle$ = {k3_threshold:g}"),
    (axes[1], k3_day3_grid, r"$\langle k_3\rangle$ at day 3", r"$\langle k_3\rangle$", "viridis", k3_threshold, rf"$\langle k_3\rangle$ = {k3_threshold:g}"),
    (axes[2], k3_last_day_grid, rf"$\langle k_3\rangle$ at day {total_hours / 24:g}", r"$\langle k_3\rangle$", "viridis", k3_threshold, rf"$\langle k_3\rangle$ = {k3_threshold:g}"),
    (axes[3], exhausted_fraction_last_day_grid, f"exhausted fraction at day {total_hours / 24:g}", "fraction", "plasma", None, None),
    (axes[4], half_time_grid, "time to first half", "days", "magma", None, None),
    (axes[5], second_half_time_grid, "time from 1/2 to 1/4", "days", "magma", None, None),
    (axes[6], half_time_ratio_grid, "second-half time / first-half time", "ratio", "cividis", None, None),
]

for args in plots:
    draw_block_panel(*args)

axes[7].axis("off")
fig.suptitle(r"$\langle k_3\rangle$ phase diagram from Bayesian density evolution model")
plt.show()


# ======================================================================
# plot relative k3 phase diagram
# ======================================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
relative_k3_plots = [
    (axes[0], k3_relative_day1_grid, r"$\langle k_3\rangle(1)/\langle k_3\rangle(0)$"),
    (axes[1], k3_relative_day3_grid, r"$\langle k_3\rangle(3)/\langle k_3\rangle(0)$"),
    (axes[2], k3_relative_last_day_grid, rf"$\langle k_3\rangle({total_hours / 24:g})/\langle k_3\rangle(0)$"),
]
for ax, grid, title in relative_k3_plots:
    draw_block_panel(
        ax,
        grid,
        title,
        r"$\langle k_3\rangle(t)/\langle k_3\rangle(0)$",
        cmap="viridis",
        threshold=k3_relative_threshold,
        threshold_label=f"ratio = {k3_relative_threshold:g}",
        vmin=0,
        vmax=1,
    )
fig.suptitle(r"relative $\langle k_3\rangle$ from Bayesian density evolution model")
plt.show()


# ======================================================================
# plot cell-number fold-change phase diagram
# ======================================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
fold_change_plots = [
    (axes[0], fold_change_day1_grid, "N(day 1) / N0"),
    (axes[1], fold_change_day3_grid, "N(day 3) / N0"),
]
for ax, grid, title in fold_change_plots:
    draw_block_panel(
        ax,
        grid,
        title,
        "fold change",
        cmap="viridis",
        threshold=fold_change_threshold,
        threshold_label=f"fold = {fold_change_threshold:g}",
    )
fig.suptitle("cell-number fold change from density evolution model")
plt.show()

