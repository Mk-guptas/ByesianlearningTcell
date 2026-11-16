import numpy as np


def clipped_linear_function(x, left_value, right_value, left_x=-1.0, right_x=1.0):
    x_clipped = np.clip(x, left_x, right_x)
    return left_value + (right_value - left_value) * (x_clipped - left_x) / (right_x - left_x)


def clipped_quadratic_function(x, left_value, right_value):
    x_clipped = np.clip(x, -1.0, 1.0)
    return left_value + (right_value - left_value) * (1 - x_clipped**2)



def hill_function(x, left_value, right_value, steepness=5.0):
    sigmoidal = 1 / (1 + np.exp(-steepness * x))
    return left_value + (right_value - left_value) * sigmoidal


def negative_hill_function(x, left_value, right_value, steepness=2.0):
    sigmoidal = 1 / (1 + np.exp(steepness * x))
    return left_value + (right_value - left_value) * sigmoidal


# Backward-compatible names used in older scripts.
hillnon_linear_function = hill_function
negative_hillnon_linear_function = negative_hill_function


def gaussian_pdf(y, mean, sigma):
    return np.exp(-0.5 * ((y - mean) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def gaussian_1d(x, mean, sigma):
    return gaussian_pdf(x, mean, sigma)


def double_well_potential_function(x, potential_strength):
    return potential_strength * (0.25 * x**4 - 0.5 * x**2)


double_well_potential = double_well_potential_function


def make_fitness_function(left_value, right_value, mode="hill", steepness=5.0):
    if mode == "hill":
        return lambda x: hill_function(x, left_value, right_value, steepness=steepness)
    if mode == "negative_hill":
        return lambda x: negative_hill_function(x, left_value, right_value, steepness=steepness)
    if mode == "clipped_linear":
        return lambda x: clipped_linear_function(x, left_value, right_value)
    if mode == "clipped_quadratic":
        return lambda x: clipped_quadratic_function(x, left_value, right_value)
    raise ValueError("mode must be hill, negative_hill, clipped_linear, or clipped_quadratic")


def make_k3_function(left_value, right_value, mode="hill", steepness=5.0):
    return make_fitness_function(left_value, right_value, mode=mode, steepness=steepness)


def make_decay_rate_function(max_decay, steepness=5.0):
    return lambda x: hill_function(x, 0.0, max_decay, steepness=steepness)


def make_exhaustion_rate_function(kappa_max, steepness=5.0):
    return lambda x: hill_function(x, 0.0, kappa_max, steepness=steepness)


def make_bayesian_likelihood_1d(
    rho,
    delta,
    x_star=1.0,
    sigma_y_given_x=1.0,
    mu_y_at_x_star=0.0,
    y_obs=0.0,
):
    sigma_y = sigma_y_given_x / np.sqrt(1.0 - rho**2)

    def likelihood_function(x, sigma_x):
        sigma_x = max(float(sigma_x), 1e-12)
        mu_y_given_x = (
            delta
            + rho * (sigma_y / sigma_x) * (x - x_star)
        )
        return gaussian_pdf(y_obs, mu_y_given_x, sigma_y_given_x)

    return likelihood_function


def relation_function(k1, k2, k3, n):
    return k1 * k3 * n / ((k2 + k3)*0.03)
