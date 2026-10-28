"""Pure numerical kernels embedded in the E1 notebook; cast is supplied by the runtime."""
import hashlib
import numpy as np


def readout_value(hidden, mean, scale, coef, intercept):
    return float(((np.asarray(hidden, dtype=np.float64) - mean) / scale) @ coef + intercept)


def numerical_correction(hidden, target, mean, scale, coef, intercept, cast,
                         tolerance=0.05, steps=32):
    """Calibrate only the external readout; downstream always receives native dtype.

    The initial edit minimizes standardized-space norm. Subsequent quantization
    calibration need not be the globally smallest native-dtype perturbation.
    """
    h = np.asarray(hidden, dtype=np.float64)
    a = np.asarray(coef, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    assert np.all(scale > 0) and float(a @ a) > 0 and steps > 0
    initial = readout_value(h, mean, scale, a, intercept)
    unit = scale * a / float(a @ a)
    planned = (target - initial) * unit
    # Preserve an exact zero edit for already-in-tolerance examples.
    best = h.copy()
    error = abs(initial - target)
    if error <= tolerance:
        return best, dict(planned_delta_norm=0.0, residual=error, achieved=True,
                          iterations=0, originally_within_tolerance=True)
    work = planned.copy()
    for iteration in range(steps):
        candidate = np.asarray(cast(h + work), dtype=np.float64)
        value = readout_value(candidate, mean, scale, a, intercept)
        candidate_error = abs(target - value)
        if candidate_error < error:
            best, error = candidate.copy(), candidate_error
        if error <= tolerance:
            break
        work += (target - value) * unit
    return best, dict(planned_delta_norm=float(np.linalg.norm(planned)),
                      residual=float(error), achieved=bool(error <= tolerance),
                      iterations=iteration + 1, originally_within_tolerance=False)


def deterministic_direction(shape, key, scale, coef, orthogonal=False):
    seed = int(hashlib.sha256(str(key).encode()).hexdigest()[:16], 16)
    z = np.random.default_rng(seed).normal(size=shape)
    if orthogonal:
        z -= coef * float(z @ coef) / float(coef @ coef)
    raw = z * scale
    assert np.linalg.norm(raw) > 0
    return raw / np.linalg.norm(raw)


def match_native_norm(hidden, direction, target_norm, cast, tolerance=0.02, steps=48):
    """Bounded amplitude search, matching actual raw residual norm after casting.

    A discrete native dtype can make exact norm matching impossible. Return the
    best candidate and an explicit failure flag; never silently call it matched.
    """
    h = np.asarray(hidden, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    assert target_norm >= 0 and np.linalg.norm(direction) > 0
    direction = direction / np.linalg.norm(direction)
    if target_norm == 0:
        return h.copy(), dict(norm_error_relative=0.0, norm_matched=True, amplitude=0.0)
    lo, hi = 0.0, max(target_norm, 1e-12)
    for _ in range(32):
        if np.linalg.norm(np.asarray(cast(h + hi * direction)) - h) >= target_norm:
            break
        hi *= 2
    best, best_error, best_amplitude = h.copy(), 1.0, 0.0
    for _ in range(steps):
        amplitude = (lo + hi) / 2
        candidate = np.asarray(cast(h + amplitude * direction), dtype=np.float64)
        norm = float(np.linalg.norm(candidate - h))
        error = abs(norm - target_norm) / target_norm
        if error < best_error:
            best, best_error, best_amplitude = candidate.copy(), error, amplitude
        if error <= tolerance:
            break
        if norm < target_norm:
            lo = amplitude
        else:
            hi = amplitude
    return best, dict(norm_error_relative=float(best_error),
                      norm_matched=bool(best_error <= tolerance), amplitude=float(best_amplitude))


def scene_bootstrap(values, seed=20260913, replicates=10000):
    """Rows must already be independent scene/family means, not mappings/seeds."""
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1 and len(values) and np.isfinite(values).all()
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, replicates, 250):
        size = min(250, replicates - start)
        means.extend(values[rng.integers(0, len(values), (size, len(values)))].mean(axis=1))
    return dict(mean=float(values.mean()), n=len(values),
                lower=float(np.quantile(means, .025)), upper=float(np.quantile(means, .975)))


def paired_on_ids(left, right, id_column, value_column):
    """Do not silently turn partial overlap or duplicate rows into a paired comparison."""
    assert left[id_column].is_unique and right[id_column].is_unique
    assert set(left[id_column]) == set(right[id_column]), 'Different paired support'
    joined = left[[id_column, value_column]].merge(
        right[[id_column, value_column]], on=id_column, suffixes=('_left', '_right'), validate='one_to_one')
    return joined[value_column + '_left'].to_numpy() - joined[value_column + '_right'].to_numpy()
