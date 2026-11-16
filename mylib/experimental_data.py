import numpy as np


def make_k3_target_data(
    target_day_count=15,
    target_end_day=3.0,
    interpolation_method="linear",
    scale=60.0,
    coated_anchor_days=None,
    coated_anchor_k3=None,
    uncoated_anchor_days=None,
    uncoated_anchor_k3=None,
):
    """Return shared coated/uncoated k3 target data in model units."""
    if coated_anchor_days is None:
        coated_anchor_days = [0.0, 4.0]
    if coated_anchor_k3 is None:
        coated_anchor_k3 = [0.01, 0.005]
    if uncoated_anchor_days is None:
        uncoated_anchor_days = [0.0, 1.0,3]
    if uncoated_anchor_k3 is None:
        uncoated_anchor_k3 = [0.01, 0.005,0.002]

    coated_anchor_days = np.asarray(coated_anchor_days, dtype=float)
    coated_anchor_k3 = np.asarray(coated_anchor_k3, dtype=float) * scale
    uncoated_anchor_days = np.asarray(uncoated_anchor_days, dtype=float)
    uncoated_anchor_k3 = np.asarray(uncoated_anchor_k3, dtype=float) * scale

    interpolation_method = interpolation_method.lower()
    if interpolation_method in {"raw", "none", "no_interpolation"}:
        target_days = np.unique(np.concatenate([coated_anchor_days, uncoated_anchor_days]))
        coated_target_days = coated_anchor_days
        coated_target_k3 = coated_anchor_k3
        uncoated_target_days = uncoated_anchor_days
        uncoated_target_k3 = uncoated_anchor_k3
    else:
        target_days = np.linspace(0.0, float(target_end_day), int(target_day_count))
        coated_target_days = target_days
        uncoated_target_days = target_days
        coated_target_k3 = interpolate_k3_data(
            coated_target_days,
            coated_anchor_days,
            coated_anchor_k3,
            interpolation_method,
        )
        uncoated_target_k3 = interpolate_k3_data(
            uncoated_target_days,
            uncoated_anchor_days,
            uncoated_anchor_k3,
            interpolation_method,
        )

    k3_data_targets = []
    for day, value in zip(coated_target_days, coated_target_k3):
        k3_data_targets.append((24.0 * day, "coated", value))
    for day, value in zip(uncoated_target_days, uncoated_target_k3):
        k3_data_targets.append((24.0 * day, "uncoated", value))

    return {
        "coated_anchor_days": coated_anchor_days,
        "coated_anchor_k3": coated_anchor_k3,
        "uncoated_anchor_days": uncoated_anchor_days,
        "uncoated_anchor_k3": uncoated_anchor_k3,
        "target_days": target_days,
        "coated_target_days": coated_target_days,
        "coated_target_k3": coated_target_k3,
        "uncoated_target_days": uncoated_target_days,
        "uncoated_target_k3": uncoated_target_k3,
        "k3_data_targets": k3_data_targets,
        "interpolation_method": interpolation_method,
    }


def interpolate_k3_data(target_days, anchor_days, anchor_k3, interpolation_method="linear"):
    interpolation_method = interpolation_method.lower()
    if interpolation_method == "linear":
        return np.interp(target_days, anchor_days, anchor_k3)

    if interpolation_method in {"exponential", "exponential_decay"}:
        if np.any(anchor_k3 <= 0):
            raise ValueError("exponential decay interpolation requires positive k3 values")
        log_anchor_k3 = np.log(anchor_k3)
        return np.exp(np.interp(target_days, anchor_days, log_anchor_k3))

    raise ValueError(
        "interpolation_method must be 'linear', 'exponential_decay', or 'raw'"
    )
