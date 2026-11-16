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
    make_k3_target_data,
)


# ======================================================================
# choose Bayesian density-model parameters here
# ======================================================================

bayesian_parameters = {
    "D": 0.01,
    "potential_strength": 0.01,
    "rho": 0.075,
    "tau": 6.0,
    "tau_nonexhausted": 0*6.0,
    "tau_exhausted": 0.0,
    "v": 0.0,
    "kappa_max": 0.3*0,
    "exhausted_fitness_mode": "fixed",
    "exhausted_fixed_fitness": 0.01,
}

delta_values = np.linspace(0.0, 0.3, 2)
delta_count_override = os.environ.get("DENSITY_PREDICTION_DELTA_COUNT")
if delta_count_override is not None:
    delta_count_override = int(delta_count_override)
    if delta_count_override < 1:
        raise ValueError("DENSITY_PREDICTION_DELTA_COUNT must be at least 1")
    delta_values = np.linspace(delta_values[0], delta_values[-1], delta_count_override)
use_diffusion_as_potential_strength = True


# ======================================================================
# numerical setup
# ======================================================================

x_min, x_max = -4.0, 4.0
nx = int(os.environ.get("DENSITY_NX", 500))
fp_dt = float(os.environ.get("DENSITY_DT", 0.01))
total_days = int(os.environ.get("DENSITY_TOTAL_DAYS", 20))
total_hours = 24 * total_days

x_star = 1.0
x0, sigma0 = 1.0, 0.1
initial_nonexhausted_total = 1.0
initial_exhausted_total = 0.0
sigma_y_given_x = 1.0

snapshot_days = sorted({day for day in [0, 1, 3, total_days] if day <= total_days})
snapshot_times = {24 * day: f"day {day}" for day in snapshot_days}


# ======================================================================
# model functions and target data
# ======================================================================

fitness_left = 0.02
fitness_right = 0.02
exhausted_k3_value = 0.001 * 60
k3_left = 0.001 * 60
k3_right = 0.01 * 60

fitness_function = make_fitness_function(fitness_left, fitness_right, mode="hill", steepness=5.0)


def k3_nonexhausted_function(x):
    return k3_left + (k3_right - k3_left) / (1.0 + np.exp(-5.0 * x))


k3_target_data = make_k3_target_data(
    target_day_count=int(os.environ.get("DENSITY_PREDICTION_TARGET_COUNT", 15)),
    target_end_day=min(total_days, 4),
    interpolation_method="linear",
)
k3_data_targets = k3_target_data["k3_data_targets"]


# ======================================================================
# simulation
# ======================================================================

def value_at(time, values, hour):
    idx = int(round(hour * (len(time) - 1) / time[-1]))
    return values[idx]


def simulate(delta):
    parameters = bayesian_parameters.copy()
    if use_diffusion_as_potential_strength:
        parameters["potential_strength"] = parameters["D"]

    solver = BayesianDensityEvolution1D(
        likelihood_function=make_bayesian_likelihood_1d(
            parameters["rho"],
            delta,
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
        snapshot_times=snapshot_times,
    )
    history = solver.get_history()
    return {
        "x": np.asarray(solver.x),
        "time": np.asarray(history["time"]),
        "time_days": np.asarray(history["time"]) / 24.0,
        "mean": np.asarray(history["mean_x_total"]),
        "mean_nonexhausted": np.asarray(history["mean_x_nonexhausted"]),
        "mean_exhausted": np.asarray(history["mean_x_exhausted"]),
        "k3": np.asarray(history["mean_k3_total"]),
        "k3_nonexhausted": np.asarray(history["mean_k3_nonexhausted"]),
        "N_nonexhausted": np.asarray(history["N_nonexhausted"]),
        "N_exhausted": np.asarray(history["N_exhausted"]),
        "N_total": np.asarray(history["N_total"]),
        "transfer_total": np.asarray(history["transfer_total"]),
        "snapshots_nonexhausted": {key: value.copy() for key, value in solver.snapshots_nonexhausted.items()},
        "snapshots_exhausted": {key: value.copy() for key, value in solver.snapshots_exhausted.items()},
        "snapshots_total": {key: value.copy() for key, value in solver.snapshots_total.items()},
    }


predictions = {float(delta): simulate(float(delta)) for delta in delta_values}


# ======================================================================
# print values at useful days
# ======================================================================

print("chosen Bayesian density-evolution parameters")
for name, value in bayesian_parameters.items():
    if isinstance(value, (int, float, np.floating)):
        print(f"  {name:22s} = {value:g}")
    else:
        print(f"  {name:22s} = {value}")
print()

for delta, prediction in predictions.items():
    print(f"delta = {delta:g}")
    useful_days = sorted({day for day in [0, 1, 3, 4, total_days] if day <= total_days})
    for day in useful_days:
        hour = 24 * day
        print(
            f"  day {day:4g}: "
            f"mean={value_at(prediction['time'], prediction['mean'], hour):+.4f}, "
            f"k3={value_at(prediction['time'], prediction['k3'], hour):.5f}, "
            f"N_total={value_at(prediction['time'], prediction['N_total'], hour):.5g}, "
            f"N_exhausted={value_at(prediction['time'], prediction['N_exhausted'], hour):.5g}"
        )


# ======================================================================
# plot delta sweep
# ======================================================================

colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(delta_values)))

fig, axes = plt.subplots(4, 1, figsize=(8, 11), sharex=True, constrained_layout=True)
for color, (delta, prediction) in zip(colors, predictions.items()):
    label = f"delta={delta:g}"
    axes[0].plot(prediction["time_days"], prediction["mean"], color=color, lw=2, label=label)
    axes[1].plot(prediction["time_days"], prediction["k3"], color=color, lw=2, label=label)
    axes[2].plot(prediction["time_days"], prediction["N_total"], color=color, lw=2, label=label)
    axes[3].plot(prediction["time_days"], prediction["N_exhausted"], color=color, lw=2, label=label)

for hour, case, value in k3_data_targets:
    axes[1].plot(hour / 24.0, value, "ko", mfc="none", ms=6, alpha=0.55)

axes[0].axhline(0.0, color="k", lw=0.8)
axes[0].set_ylabel("mean phenotype")
axes[0].set_title("Bayesian density evolution across delta")
axes[1].set_ylabel("mean k3")
axes[2].set_ylabel("total cell number")
axes[3].set_ylabel("exhausted cell number")
axes[3].set_xlabel("time (days)")
for ax in axes:
    ax.grid(True, alpha=0.3)
    ax.legend()
plt.show()


# ======================================================================
# plot density snapshots
# ======================================================================

snapshot_styles = {"day 0": "-", "day 1": "--", "day 3": "-.", f"day {total_days}": ":"}

fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, constrained_layout=True)
for color, (delta, prediction) in zip(colors, predictions.items()):
    for label, linestyle in snapshot_styles.items():
        if label in prediction["snapshots_total"]:
            axes[0].plot(
                prediction["x"],
                prediction["snapshots_total"][label],
                color=color,
                ls=linestyle,
                lw=1.8,
                label=f"delta={delta:g}, {label}",
            )
        if label in prediction["snapshots_nonexhausted"]:
            axes[1].plot(
                prediction["x"],
                prediction["snapshots_nonexhausted"][label],
                color=color,
                ls=linestyle,
                lw=1.8,
                label=f"delta={delta:g}, {label}",
            )
        if label in prediction["snapshots_exhausted"]:
            axes[2].plot(
                prediction["x"],
                prediction["snapshots_exhausted"][label],
                color=color,
                ls=linestyle,
                lw=1.8,
                label=f"delta={delta:g}, {label}",
            )

axes[0].set_ylabel("n_total(x,t)")
axes[0].set_title("total density snapshots")
axes[1].set_ylabel("n_m(x,t)")
axes[1].set_title("non-exhausted density snapshots")
axes[2].set_xlabel("x")
axes[2].set_ylabel("n_e(x,t)")
axes[2].set_title("exhausted density snapshots")
for ax in axes:
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
plt.show()


# ======================================================================
# plot model functions
# ======================================================================

x_plot = np.linspace(-2, 2, 400)
parameters = bayesian_parameters.copy()
if use_diffusion_as_potential_strength:
    parameters["potential_strength"] = parameters["D"]

fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True, constrained_layout=True)
axes[0].plot(x_plot, double_well_potential_function(x_plot, parameters["potential_strength"]), lw=2)
axes[0].set_ylabel("U(x)")
axes[1].plot(x_plot, fitness_function(x_plot), lw=2, label="non-exhausted")
if parameters["exhausted_fitness_mode"] == "fixed":
    axes[1].axhline(parameters["exhausted_fixed_fitness"], color="tab:red", ls="--", label="exhausted fixed")
else:
    axes[1].plot(x_plot, fitness_function(x_plot), color="tab:red", ls="--", lw=1.5, label="exhausted same")
axes[1].set_ylabel("fitness")
axes[1].legend()
axes[2].plot(x_plot, k3_nonexhausted_function(x_plot), lw=2, label="non-exhausted k3")
axes[2].axhline(exhausted_k3_value, color="tab:red", ls="--", label="exhausted k3")
axes[2].set_ylabel("k3")
axes[2].legend()
axes[3].plot(x_plot, make_exhaustion_rate_function(parameters["kappa_max"])(x_plot), lw=2)
axes[3].set_ylabel("kappa(x)")
axes[3].set_xlabel("x")
for ax in axes:
    ax.grid(True, alpha=0.3)
plt.show()

