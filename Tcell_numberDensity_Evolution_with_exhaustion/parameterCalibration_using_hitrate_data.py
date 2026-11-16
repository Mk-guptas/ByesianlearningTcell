from pathlib import Path
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

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
# numerical setup
# ======================================================================

x_min, x_max = -4.0, 4.0
nx = int(os.environ.get("DENSITY_CALIBRATION_NX", 500))
fp_dt = float(os.environ.get("DENSITY_CALIBRATION_DT", 0.01))
x_star = 1.0
x0, sigma0 = 1.0, 0.1
initial_nonexhausted_total = 1.0
initial_exhausted_total = 0.0
sigma_y_given_x = 1.0
max_hours = 24 * int(os.environ.get("DENSITY_CALIBRATION_TOTAL_DAYS", 4))
max_nfev_env = os.environ.get("DENSITY_CALIBRATION_MAX_NFEV")
max_nfev = None if max_nfev_env is None else int(max_nfev_env)


# ======================================================================
# model-function parameters
# ======================================================================

fitness_left = 0.02
fitness_right = 0.04
exhausted_k3_value = 0.001 * 60
exhausted_fixed_fitness = 0.01*0.5
k3_left = 0.001 * 60
k3_right = 0.01 * 60

fitness_function = make_fitness_function(fitness_left, fitness_right, mode="hill", steepness=5.0)


def k3_nonexhausted_function(x):
    return k3_left + (k3_right - k3_left) / (1.0 + np.exp(-5.0 * x))


# ======================================================================
# k3 target data
# ======================================================================

target_day_count = int(os.environ.get("DENSITY_CALIBRATION_TARGET_COUNT", 15))
k3_interpolation_method = "linear"  # choose "linear", "exponential_decay", or "raw"

k3_target_data = make_k3_target_data(
    target_day_count=target_day_count,
    target_end_day=max_hours / 24.0,
    interpolation_method=k3_interpolation_method,
)
k3_data_targets = k3_target_data["k3_data_targets"]

print("=" * 64)
print("k3 data used for the joint fit  (coated / uncoated)")
print("=" * 64)
for hour, case, value in k3_data_targets:
    print(f"  day {hour / 24:.2f}  {case:8s}  k3 = {value:.4f}")
print("=" * 64)


# ======================================================================
# simulation and residuals
# ======================================================================

def simulate(diffusion_coefficient, rho, v, tau, tau_exhausted, delta, kappa_max, total_hours=max_hours):
    potential_strength = diffusion_coefficient
    solver = BayesianDensityEvolution1D(
        likelihood_function=make_bayesian_likelihood_1d(
            rho,
            delta,
            x_star=x_star,
            sigma_y_given_x=sigma_y_given_x,
        ),
        potential_function=lambda x: double_well_potential_function(x, potential_strength),
        fitness_function=fitness_function,
        exhaustion_rate_function=make_exhaustion_rate_function(kappa_max, steepness=5.0),
        k3_nonexhausted_function=k3_nonexhausted_function,
        exhausted_k3_value=exhausted_k3_value,
        D=diffusion_coefficient,
        tau=tau,
        tau_nonexhausted=tau,
        tau_exhausted=tau_exhausted,
        v=v,
        exhausted_fitness_mode="same",
        exhausted_fixed_fitness=exhausted_fixed_fitness,
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
    return {
        "time": np.asarray(history["time"]),
        "mean": np.asarray(history["mean_x_total"]),
        "mean_fitness": np.asarray(history["mean_fitness_total"]),
        "k3": np.asarray(history["mean_k3_total"]),
        "k3_nonexhausted": np.asarray(history["mean_k3_nonexhausted"]),
        "k3_exhausted": np.asarray(history["mean_k3_exhausted"]),
        "N_nonexhausted": np.asarray(history["N_nonexhausted"]),
        "N_exhausted": np.asarray(history["N_exhausted"]),
        "N_total": np.asarray(history["N_total"]),
        "transfer_total": np.asarray(history["transfer_total"]),
        "solver": solver,
    }


def value_at(time, values, hour):
    idx = int(round(hour * (len(time) - 1) / time[-1]))
    return values[idx]


def run_pair(theta):
    parameters = unpack_parameters(theta)
    coated = simulate(
        parameters["D"],
        parameters["rho"],
        parameters["v"],
        parameters["tau"],
        parameters["tau_exhausted"],
        parameters["delta_coated"],
        parameters["kappa_max"],
    )
    uncoated = simulate(
        parameters["D"],
        parameters["rho"],
        parameters["v"],
        parameters["tau"],
        parameters["tau_exhausted"],
        parameters["delta_uncoated"],
        parameters["kappa_max"],
    )
    return {"coated": coated, "uncoated": uncoated}


def residuals(theta):
    parameters = unpack_parameters(theta)
    total_residual_count = len(k3_data_targets) + (len(theta) if regularization_lambda > 0 else 0)
    for name in fit_parameter_names:
        lower, upper = parameter_config[name]["bounds"]
        if parameters[name] < lower or parameters[name] > upper:
            return np.full(total_residual_count, 1e3)

    try:
        predictions = run_pair(theta)
    except Exception:
        return np.full(total_residual_count, 1e3)

    residual = []
    for hour, case, value in k3_data_targets:
        prediction = predictions[case]
        k3_value = value_at(prediction["time"], prediction["k3"], hour)
        if not np.isfinite(k3_value) or k3_value <= 0 or value <= 0:
            residual.append(1e3)
        else:
            residual.append(np.log(k3_value) - np.log(value))

    if regularization_lambda > 0:
        theta_scaled = np.asarray(theta, dtype=float) / regularization_scales
        residual.extend(np.sqrt(regularization_lambda) * theta_scaled)

    return np.asarray(residual, dtype=float)


# ======================================================================
# choose fitted and fixed parameters
# ======================================================================

parameter_config = {
    "delta_coated": {"fit": True, "initial": 0.05, "bounds": (0.0, 1.0)},
    "delta_uncoated": {"fit": True, "initial": 0.5, "bounds": (0.0, 1.0)},
    "rho": {"fit": True, "initial": 0.02, "bounds": (0.005, 0.2)},
    "D": {"fit": False ,"initial": 0.01, "bounds": (0.0005, 0.1)},
    "kappa_max": {"fit": True, "initial": 0.2, "bounds": (0.0, 2.0)},
    "tau": {"fit": False, "value": 6, "initial": 6.0, "bounds": (0.5, 10.0)},
    "tau_exhausted": {"fit": False, "value":0.0, "initial": 0.0, "bounds": (0.0, 10.0)},
    "v": {"fit": False, "value": 0.0, "initial": 0.0, "bounds": (0.0, 0.3)},
}

fit_parameter_names = [name for name, settings in parameter_config.items() if settings["fit"]]
fixed_parameter_names = [name for name, settings in parameter_config.items() if not settings["fit"]]

initial_guess = np.array([parameter_config[name]["initial"] for name in fit_parameter_names], dtype=float)
lower_bounds = np.array([parameter_config[name]["bounds"][0] for name in fit_parameter_names], dtype=float)
upper_bounds = np.array([parameter_config[name]["bounds"][1] for name in fit_parameter_names], dtype=float)


def unpack_parameters(theta):
    parameters = {}
    for name, settings in parameter_config.items():
        parameters[name] = settings.get("value", settings.get("initial"))
    for value, name in zip(theta, fit_parameter_names):
        parameters[name] = float(value)
    return parameters


lambda_values = np.array([ 0.01, 0.1])
lambda_data_tolerance = 0.05
regularization_lambda = 0.0
regularization_scales = upper_bounds.copy()

initial_residual = residuals(initial_guess)
print("\ninitial guess:")
print(f"  fitted parameters     = {fit_parameter_names}")
print(f"  fixed parameters      = {fixed_parameter_names}")
initial_parameters = unpack_parameters(initial_guess)
for name in parameter_config:
    label = "fit" if parameter_config[name]["fit"] else "fixed"
    print(f"  {name:22s} = {initial_parameters[name]:.5f}  [{label}]")
print(f"  residual norm         = {np.linalg.norm(initial_residual):.5f}")

lambda_scan = []
start_guess = initial_guess.copy()
for lambda_value in lambda_values:
    regularization_lambda = float(lambda_value)
    scan_result = least_squares(
        residuals,
        start_guess,
        bounds=(lower_bounds, upper_bounds),
        method="trf",
        xtol=1e-5,
        ftol=1e-3,
        gtol=1e-5,
        max_nfev=max_nfev,
        verbose=0,
    )
    start_guess = scan_result.x.copy()
    scan_data_residual = scan_result.fun[:len(k3_data_targets)]
    scan_reg_residual = scan_result.fun[len(k3_data_targets):]
    lambda_scan.append({
        "lambda": regularization_lambda,
        "result": scan_result,
        "data_residual_squared": float(np.sum(scan_data_residual**2)),
        "reg_residual_squared": float(np.sum(scan_reg_residual**2)),
        "parameter_norm": float(np.linalg.norm(scan_result.x / regularization_scales)),
    })

best_data_residual_squared = min(item["data_residual_squared"] for item in lambda_scan)
allowed_data_residual_squared = best_data_residual_squared * (1.0 + lambda_data_tolerance)
eligible_indices = [
    index for index, item in enumerate(lambda_scan)
    if item["data_residual_squared"] <= allowed_data_residual_squared
]
selected_index = min(eligible_indices, key=lambda index: lambda_scan[index]["parameter_norm"])
selected = lambda_scan[selected_index]
regularization_lambda = selected["lambda"]
result = selected["result"]
fit_parameters = unpack_parameters(result.x)

print("\nlambda scan:")
print("  lambda      data_resid_sq   reg_resid_sq    scaled_param_norm")
for item in lambda_scan:
    marker = "  <-- selected" if item is selected else ""
    print(
        f"  {item['lambda']:<10.1g} "
        f"{item['data_residual_squared']:<15.6g} "
        f"{item['reg_residual_squared']:<15.6g} "
        f"{item['parameter_norm']:<15.6g}"
        f"{marker}"
    )

data_residual = result.fun[:len(k3_data_targets)]
regularization_residual = result.fun[len(k3_data_targets):]

residual_variance = np.sum(result.fun**2) / max(len(result.fun) - len(result.x), 1)
try:
    covariance = residual_variance * np.linalg.inv(result.jac.T @ result.jac)
    parameter_error = np.sqrt(np.diag(covariance))
except np.linalg.LinAlgError:
    parameter_error = np.full(len(fit_parameter_names), np.nan)

error_by_name = {name: 0.0 for name in parameter_config}
for name, error in zip(fit_parameter_names, parameter_error):
    error_by_name[name] = error

print("\n" + "=" * 64)
print("best-fit parameters")
print("=" * 64)
for name in parameter_config:
    label = "fit" if parameter_config[name]["fit"] else "fixed"
    print(f"  {name:22s} = {fit_parameters[name]:.5f} +/- {error_by_name[name]:.5f}  [{label}]")
print(f"  potential_strength     = {fit_parameters['D']:.5f} +/- {error_by_name['D']:.5f}")
print(f"  data residual norm     = {np.linalg.norm(data_residual):.5f}")
print(f"  reg residual norm      = {np.linalg.norm(regularization_residual):.5f}")
print(f"  total residual norm    = {np.linalg.norm(result.fun):.5f}")
print(f"  least-squares cost     = {result.cost:.5f}")

lambda_plot_values = np.array([item["lambda"] for item in lambda_scan])
data_residual_squared_values = np.array([item["data_residual_squared"] for item in lambda_scan])

plt.figure(figsize=(7, 4), constrained_layout=True)
plt.plot(lambda_plot_values, data_residual_squared_values, "o-", lw=2)
plt.plot(regularization_lambda, selected["data_residual_squared"], "r*", ms=14, label="selected lambda")
plt.axhline(allowed_data_residual_squared, color="gray", ls="--", lw=1)
plt.xscale("symlog", linthresh=1e-6)
plt.xlabel("regularization lambda")
plt.ylabel("data residual squared")
plt.title("Lambda scan for T-cell density evolution model")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


# ======================================================================
# compare fitted model with data
# ======================================================================

fitted = run_pair(result.x)

print("\nfitted predictions at data points:")
for hour, case, value in k3_data_targets:
    prediction = fitted[case]
    print(
        f"  day {hour / 24:.2f} {case:8s}: "
        f"mean={value_at(prediction['time'], prediction['mean'], hour):+.3f}, "
        f"k3={value_at(prediction['time'], prediction['k3'], hour):.5f}, "
        f"N_total={value_at(prediction['time'], prediction['N_total'], hour):.5g}, "
        f"N_exhausted={value_at(prediction['time'], prediction['N_exhausted'], hour):.5g} "
        f"(data {value:.5f})"
    )

time_coated = fitted["coated"]["time"] / 24.0
time_uncoated = fitted["uncoated"]["time"] / 24.0



# ======================================================================
# Plotting fit vs data
# ======================================================================



fig, axes = plt.subplots(4, 1, figsize=(8, 11), constrained_layout=True)
axes[0].plot(time_coated, fitted["coated"]["mean"],"--", lw=2, label=f"SIN-treated")
axes[0].plot(time_uncoated, fitted["uncoated"]["mean"],  lw=2, label=f"Untreated")
axes[0].axhline(0.0, color="k", lw=0.8)
axes[0].set_ylabel("mean phenotype")

axes[1].plot(time_coated, fitted["coated"]["N_total"], lw=2, label="coated total")
axes[1].plot(time_uncoated, fitted["uncoated"]["N_total"], "--", lw=2, label="uncoated total")
axes[1].plot(time_coated, fitted["coated"]["N_exhausted"], lw=1.5, label="coated exhausted")
axes[1].plot(time_uncoated, fitted["uncoated"]["N_exhausted"], "--", lw=1.5, label="uncoated exhausted")
axes[1].set_ylabel("cell number")

axes[2].plot(time_coated, fitted["coated"]["k3"], "--",lw=2, label=f"SIN-treated")
axes[2].plot(time_uncoated, fitted["uncoated"]["k3"], "--", lw=2, label=f"Untreated")
for hour, case, value in k3_data_targets:
    color = "tab:blue" if case == "coated" else "tab:orange"
    axes[2].plot(hour / 24.0, value, "o", color=color, mfc="none", ms=9)
axes[2].set_ylabel("mean k3")
axes[2].legend()


axes[3].plot(time_coated, fitted["coated"]["transfer_total"], lw=2, label=f"SIN-treated")
axes[3].plot(time_uncoated, fitted["uncoated"]["transfer_total"], "--", lw=2, label=f"Untreated")
axes[3].set_xlabel("time (days)")
axes[3].set_ylabel("exhaustion transfer")
plt.show()




x_plot = np.linspace(-2, 2, 400)
fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True, constrained_layout=True)
axes[0].plot(x_plot, double_well_potential_function(x_plot, fit_parameters["D"]), lw=2)
axes[0].set_ylabel("U(x)")
axes[1].plot(x_plot, fitness_function(x_plot), lw=2, label="non-exhausted")
axes[1].axhline(exhausted_fixed_fitness, color="tab:red", ls="--", label="exhausted fixed")
axes[1].set_ylabel("fitness")
axes[1].legend()
axes[2].plot(x_plot, k3_nonexhausted_function(x_plot), lw=2, label="non-exhausted")
axes[2].axhline(exhausted_k3_value, color="tab:red", ls="--", label="exhausted")
axes[2].set_ylabel("k3")
axes[2].legend()
axes[3].plot(x_plot, make_exhaustion_rate_function(fit_parameters["kappa_max"])(x_plot), lw=2)
axes[3].set_xlabel("x")
axes[3].set_ylabel("kappa(x)")
plt.show()

