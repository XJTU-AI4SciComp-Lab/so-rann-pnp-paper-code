from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy.sparse import bmat, csr_matrix, lil_matrix
from scipy.sparse.linalg import splu, spsolve


PI = np.pi


def s_field(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    return np.sin(PI * x) * np.sin(PI * y) * np.sin(t)


def exact_c1(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    return s_field(x, y, t) + 1.1


def exact_c2(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    return -s_field(x, y, t) + 1.1


def exact_phi(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    return s_field(x, y, t) / PI**2


def exact_derivatives(x: np.ndarray, y: np.ndarray, t: float) -> dict[str, np.ndarray]:
    sx = PI * np.cos(PI * x) * np.sin(PI * y) * np.sin(t)
    sy = PI * np.sin(PI * x) * np.cos(PI * y) * np.sin(t)
    return {
        "c1x": sx,
        "c1y": sy,
        "c2x": -sx,
        "c2y": -sy,
        "phix": sx / PI**2,
        "phiy": sy / PI**2,
    }


def sources(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    d1: float = 1.0,
    d2: float = 1.0,
    z1: float = 1.0,
    z2: float = -1.0,
    epsilon: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = s_field(x, y, t)
    c1 = s + 1.1
    c2 = -s + 1.1
    st = np.sin(PI * x) * np.sin(PI * y) * np.cos(t)
    sx = PI * np.cos(PI * x) * np.sin(PI * y) * np.sin(t)
    sy = PI * np.sin(PI * x) * np.cos(PI * y) * np.sin(t)
    phi_x, phi_y = sx / PI**2, sy / PI**2
    lap_s = -2.0 * PI**2 * s
    lap_phi = -2.0 * s
    div_c1_grad_phi = sx * phi_x + sy * phi_y + c1 * lap_phi
    div_c2_grad_phi = -sx * phi_x - sy * phi_y + c2 * lap_phi
    f1 = st - d1 * (lap_s + z1 * div_c1_grad_phi)
    f2 = -st - d2 * (-lap_s + z2 * div_c2_grad_phi)
    g = -(epsilon**2) * lap_phi - (z1 * c1 + z2 * c2)
    return f1, f2, g


def _index(i: int, j: int, n: int) -> int:
    return i * n + j


def _div_c_grad_phi(c: np.ndarray, phi: np.ndarray, h: float) -> np.ndarray:
    result = np.zeros_like(c)
    fx_plus = 0.5 * (c[2:, 1:-1] + c[1:-1, 1:-1]) * (phi[2:, 1:-1] - phi[1:-1, 1:-1]) / h
    fx_minus = 0.5 * (c[1:-1, 1:-1] + c[:-2, 1:-1]) * (phi[1:-1, 1:-1] - phi[:-2, 1:-1]) / h
    fy_plus = 0.5 * (c[1:-1, 2:] + c[1:-1, 1:-1]) * (phi[1:-1, 2:] - phi[1:-1, 1:-1]) / h
    fy_minus = 0.5 * (c[1:-1, 1:-1] + c[1:-1, :-2]) * (phi[1:-1, 1:-1] - phi[1:-1, :-2]) / h
    result[1:-1, 1:-1] = (fx_plus - fx_minus + fy_plus - fy_minus) / h
    return result


def _build_poisson_lu(n: int, h: float, epsilon: float):
    unknowns = n * n
    matrix = lil_matrix((unknowns, unknowns))
    for i in range(n):
        for j in range(n):
            row = _index(i, j, n)
            if i == 0:
                matrix[row, _index(0, j, n)] = -3.0
                matrix[row, _index(1, j, n)] = 4.0
                matrix[row, _index(2, j, n)] = -1.0
            elif i == n - 1:
                matrix[row, _index(n - 1, j, n)] = 3.0
                matrix[row, _index(n - 2, j, n)] = -4.0
                matrix[row, _index(n - 3, j, n)] = 1.0
            elif j == 0:
                matrix[row, _index(i, 0, n)] = -3.0
                matrix[row, _index(i, 1, n)] = 4.0
                matrix[row, _index(i, 2, n)] = -1.0
            elif j == n - 1:
                matrix[row, _index(i, n - 1, n)] = 3.0
                matrix[row, _index(i, n - 2, n)] = -4.0
                matrix[row, _index(i, n - 3, n)] = 1.0
            else:
                coefficient = epsilon**2 / h**2
                matrix[row, row] = 4.0 * coefficient
                matrix[row, _index(i + 1, j, n)] = -coefficient
                matrix[row, _index(i - 1, j, n)] = -coefficient
                matrix[row, _index(i, j + 1, n)] = -coefficient
                matrix[row, _index(i, j - 1, n)] = -coefficient

    ones = np.ones((unknowns, 1))
    augmented = bmat(
        [[matrix.tocsr(), csr_matrix(ones)], [csr_matrix(ones.T), csr_matrix((1, 1))]],
        format="csc",
    )
    return splu(augmented)


def _solve_poisson(
    lu,
    charge_plus_source: np.ndarray,
    phi_x: np.ndarray,
    phi_y: np.ndarray,
    h: float,
) -> np.ndarray:
    n = charge_plus_source.shape[0]
    rhs = charge_plus_source.reshape(-1).copy()
    for j in range(n):
        rhs[_index(0, j, n)] = 2.0 * h * phi_x[0, j]
        rhs[_index(n - 1, j, n)] = 2.0 * h * phi_x[n - 1, j]
    for i in range(1, n - 1):
        rhs[_index(i, 0, n)] = 2.0 * h * phi_y[i, 0]
        rhs[_index(i, n - 1, n)] = 2.0 * h * phi_y[i, n - 1]
    solution = lu.solve(np.concatenate([rhs, np.array([0.0])]))[:-1]
    return solution.reshape((n, n))


def _build_concentration_lu(alpha: float, diffusion: float, n: int, h: float):
    matrix = lil_matrix((n * n, n * n))
    coefficient = diffusion / h**2
    for i in range(n):
        for j in range(n):
            row = _index(i, j, n)
            if i == 0:
                matrix[row, _index(0, j, n)] = -3.0
                matrix[row, _index(1, j, n)] = 4.0
                matrix[row, _index(2, j, n)] = -1.0
            elif i == n - 1:
                matrix[row, _index(n - 1, j, n)] = 3.0
                matrix[row, _index(n - 2, j, n)] = -4.0
                matrix[row, _index(n - 3, j, n)] = 1.0
            elif j == 0:
                matrix[row, _index(i, 0, n)] = -3.0
                matrix[row, _index(i, 1, n)] = 4.0
                matrix[row, _index(i, 2, n)] = -1.0
            elif j == n - 1:
                matrix[row, _index(i, n - 1, n)] = 3.0
                matrix[row, _index(i, n - 2, n)] = -4.0
                matrix[row, _index(i, n - 3, n)] = 1.0
            else:
                matrix[row, row] = alpha + 4.0 * coefficient
                matrix[row, _index(i + 1, j, n)] = -coefficient
                matrix[row, _index(i - 1, j, n)] = -coefficient
                matrix[row, _index(i, j + 1, n)] = -coefficient
                matrix[row, _index(i, j - 1, n)] = -coefficient
    return splu(matrix.tocsc())


def _explicit_boundary_derivatives(value: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    derivative_x = np.zeros_like(value)
    derivative_y = np.zeros_like(value)
    derivative_x[0, :] = (-3.0 * value[0, :] + 4.0 * value[1, :] - value[2, :]) / (2.0 * h)
    derivative_x[-1, :] = (3.0 * value[-1, :] - 4.0 * value[-2, :] + value[-3, :]) / (2.0 * h)
    derivative_y[:, 0] = (-3.0 * value[:, 0] + 4.0 * value[:, 1] - value[:, 2]) / (2.0 * h)
    derivative_y[:, -1] = (3.0 * value[:, -1] - 4.0 * value[:, -2] + value[:, -3]) / (2.0 * h)
    return derivative_x, derivative_y


def _concentration_rhs(
    history_rhs: np.ndarray,
    explicit_c: np.ndarray,
    explicit_phi: np.ndarray,
    source: np.ndarray,
    exact_c: np.ndarray,
    exact_c_x: np.ndarray,
    exact_c_y: np.ndarray,
    exact_phi_x: np.ndarray,
    exact_phi_y: np.ndarray,
    diffusion: float,
    valence: float,
    h: float,
) -> np.ndarray:
    n = exact_c.shape[0]
    rhs = np.zeros(n * n)
    interior_rhs = history_rhs + diffusion * valence * _div_c_grad_phi(explicit_c, explicit_phi, h) + source
    explicit_phi_x, explicit_phi_y = _explicit_boundary_derivatives(explicit_phi, h)
    for i in range(n):
        for j in range(n):
            row = _index(i, j, n)
            if i == 0 or i == n - 1:
                total_flux_x = exact_c_x[i, j] + valence * exact_c[i, j] * exact_phi_x[i, j]
                implicit_c_x = total_flux_x - valence * explicit_c[i, j] * explicit_phi_x[i, j]
                rhs[row] = 2.0 * h * implicit_c_x
            elif j == 0 or j == n - 1:
                total_flux_y = exact_c_y[i, j] + valence * exact_c[i, j] * exact_phi_y[i, j]
                implicit_c_y = total_flux_y - valence * explicit_c[i, j] * explicit_phi_y[i, j]
                rhs[row] = 2.0 * h * implicit_c_y
            else:
                rhs[row] = interior_rhs[i, j]
    return rhs


def _solve_concentration(
    lu,
    history_rhs: np.ndarray,
    explicit_c: np.ndarray,
    explicit_phi: np.ndarray,
    source: np.ndarray,
    exact_c: np.ndarray,
    exact_c_x: np.ndarray,
    exact_c_y: np.ndarray,
    phi_x: np.ndarray,
    phi_y: np.ndarray,
    diffusion: float,
    valence: float,
    h: float,
) -> np.ndarray:
    rhs = _concentration_rhs(
        history_rhs, explicit_c, explicit_phi, source, exact_c, exact_c_x, exact_c_y,
        phi_x, phi_y, diffusion, valence, h,
    )
    n = exact_c.shape[0]
    return lu.solve(rhs).reshape((n, n))


def _trapezoidal_integral(value: np.ndarray, h: float) -> float:
    weights = np.ones_like(value)
    weights[[0, -1], :] *= 0.5
    weights[:, [0, -1]] *= 0.5
    return float(h**2 * np.sum(weights * value))


def _l2_error(value: np.ndarray, exact: np.ndarray, h: float) -> float:
    return np.sqrt(_trapezoidal_integral((value - exact) ** 2, h))


@dataclass
class FDMResult:
    grid_points: int
    h: float
    time_steps: int
    dt: float
    c1_l2_error: float
    c2_l2_error: float
    phi_l2_error: float
    c1_relative_l2_error: float
    c2_relative_l2_error: float
    phi_relative_l2_error: float
    final_charge_defect: float
    final_normalized_charge_defect: float
    runtime_seconds: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def solve_example4_1(
    grid_points: int = 41,
    dt: float = 2.5e-3,
    final_time: float = 1.0,
    d1: float = 1.0,
    d2: float = 1.0,
    z1: float = 1.0,
    z2: float = -1.0,
    epsilon: float = 1.0,
) -> FDMResult:
    if grid_points < 5:
        raise ValueError("grid_points must be at least 5")
    time_steps = int(round(final_time / dt))
    dt = final_time / time_steps
    coordinates = np.linspace(-1.0, 1.0, grid_points)
    h = coordinates[1] - coordinates[0]
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    start = time.perf_counter()
    poisson_lu = _build_poisson_lu(grid_points, h, epsilon)
    bdf1_lu_c1 = _build_concentration_lu(1.0 / dt, d1, grid_points, h)
    bdf1_lu_c2 = _build_concentration_lu(1.0 / dt, d2, grid_points, h)
    bdf2_lu_c1 = _build_concentration_lu(3.0 / (2.0 * dt), d1, grid_points, h)
    bdf2_lu_c2 = _build_concentration_lu(3.0 / (2.0 * dt), d2, grid_points, h)

    c1_nm1 = exact_c1(x, y, 0.0)
    c2_nm1 = exact_c2(x, y, 0.0)
    phi_nm1 = exact_phi(x, y, 0.0)
    c1_n, c2_n, phi_n = c1_nm1.copy(), c2_nm1.copy(), phi_nm1.copy()

    for step in range(time_steps):
        next_time = (step + 1) * dt
        f1, f2, g = sources(x, y, next_time, d1, d2, z1, z2, epsilon)
        derivatives = exact_derivatives(x, y, next_time)
        exact1 = exact_c1(x, y, next_time)
        exact2 = exact_c2(x, y, next_time)

        if step == 0:
            lu_c1, lu_c2 = bdf1_lu_c1, bdf1_lu_c2
            history1, history2 = c1_n / dt, c2_n / dt
            explicit1, explicit2, explicit_phi = c1_n, c2_n, phi_n
        else:
            lu_c1, lu_c2 = bdf2_lu_c1, bdf2_lu_c2
            history1 = (4.0 * c1_n - c1_nm1) / (2.0 * dt)
            history2 = (4.0 * c2_n - c2_nm1) / (2.0 * dt)
            explicit1 = 2.0 * c1_n - c1_nm1
            explicit2 = 2.0 * c2_n - c2_nm1
            explicit_phi = 2.0 * phi_n - phi_nm1

        c1_np1 = _solve_concentration(
            lu_c1, history1, explicit1, explicit_phi, f1, exact1,
            derivatives["c1x"], derivatives["c1y"], derivatives["phix"], derivatives["phiy"],
            d1, z1, h,
        )
        c2_np1 = _solve_concentration(
            lu_c2, history2, explicit2, explicit_phi, f2, exact2,
            derivatives["c2x"], derivatives["c2y"], derivatives["phix"], derivatives["phiy"],
            d2, z2, h,
        )
        phi_np1 = _solve_poisson(
            poisson_lu, z1 * c1_np1 + z2 * c2_np1 + g,
            derivatives["phix"], derivatives["phiy"], h,
        )
        c1_nm1, c1_n = c1_n, c1_np1
        c2_nm1, c2_n = c2_n, c2_np1
        phi_nm1, phi_n = phi_n, phi_np1

    runtime = time.perf_counter() - start
    exact1 = exact_c1(x, y, final_time)
    exact2 = exact_c2(x, y, final_time)
    exactp = exact_phi(x, y, final_time)
    exactp -= _trapezoidal_integral(exactp, h) / 4.0
    errors = (_l2_error(c1_n, exact1, h), _l2_error(c2_n, exact2, h), _l2_error(phi_n, exactp, h))
    exact_norms = (
        np.sqrt(_trapezoidal_integral(exact1**2, h)),
        np.sqrt(_trapezoidal_integral(exact2**2, h)),
        np.sqrt(_trapezoidal_integral(exactp**2, h)),
    )
    mass1 = _trapezoidal_integral(c1_n, h)
    mass2 = _trapezoidal_integral(c2_n, h)
    charge = z1 * mass1 + z2 * mass2
    charge_scale = abs(z1) * abs(mass1) + abs(z2) * abs(mass2)
    return FDMResult(
        grid_points=grid_points,
        h=float(h),
        time_steps=time_steps,
        dt=float(dt),
        c1_l2_error=float(errors[0]),
        c2_l2_error=float(errors[1]),
        phi_l2_error=float(errors[2]),
        c1_relative_l2_error=float(errors[0] / exact_norms[0]),
        c2_relative_l2_error=float(errors[1] / exact_norms[1]),
        phi_relative_l2_error=float(errors[2] / exact_norms[2]),
        final_charge_defect=float(abs(charge)),
        final_normalized_charge_defect=float(abs(charge) / max(charge_scale, 1e-30)),
        runtime_seconds=float(runtime),
    )
