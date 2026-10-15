"""UE25MA242A - Mathematical Foundation for AI and Data Science
Interpolation, Extrapolation and Climate Change.
Run: python climate_maths_project_final.py
"""

import csv
import io
import os
import urllib.request

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAVE_PLOTS = True
except ImportError:
    HAVE_PLOTS = False

# Settings and data

USE_LIVE_DATA = True
OUTPUT_FOLDER = "project_outputs"
DATA_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
SAVED_START_YEAR = 1880

SAVED_TEMPS = [
    -0.17, -0.09, -0.11, -0.18, -0.28, -0.33, -0.31, -0.36, -0.17, -0.10,
    -0.35, -0.22, -0.27, -0.31, -0.30, -0.23, -0.11, -0.11, -0.27, -0.18,
    -0.09, -0.16, -0.28, -0.37, -0.47, -0.26, -0.22, -0.39, -0.43, -0.48,
    -0.43, -0.44, -0.36, -0.35, -0.16, -0.14, -0.36, -0.46, -0.30, -0.27,
    -0.27, -0.19, -0.28, -0.26, -0.27, -0.22, -0.11, -0.22, -0.20, -0.36,
    -0.16, -0.09, -0.16, -0.29, -0.13, -0.20, -0.15, -0.03, -0.01, -0.02,
    0.13, 0.19, 0.07, 0.09, 0.20, 0.09, -0.07, -0.03, -0.11, -0.11,
    -0.17, -0.07, 0.01, 0.08, -0.13, -0.14, -0.19, 0.05, 0.06, 0.03,
    -0.03, 0.06, 0.03, 0.05, -0.20, -0.11, -0.06, -0.02, -0.08, 0.05,
    0.03, -0.08, 0.01, 0.16, -0.07, -0.01, -0.10, 0.18, 0.07, 0.16,
    0.26, 0.32, 0.14, 0.31, 0.16, 0.12, 0.18, 0.32, 0.39, 0.27,
    0.45, 0.40, 0.22, 0.23, 0.32, 0.45, 0.33, 0.46, 0.61, 0.38,
    0.39, 0.54, 0.63, 0.62, 0.53, 0.68, 0.64, 0.66, 0.54, 0.65,
    0.72, 0.61, 0.65, 0.68, 0.75, 0.90, 1.01, 0.92, 0.85, 0.98,
    1.02, 0.85, 0.89, 1.17, 1.28,
]

np.set_printoptions(precision=4, suppress=True, linewidth=120)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def load_data():
    if USE_LIVE_DATA:
        try:
            request = urllib.request.Request(
                DATA_URL,
                headers={"User-Agent": "MFAD-mini-project/1.0"},
            )
            with urllib.request.urlopen(request, timeout=8) as response:
                raw_text = response.read().decode("utf-8", errors="replace")

            lines = raw_text.splitlines()
            header_index = next(i for i, line in enumerate(lines) if line.startswith("Year,"))
            reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
            loaded_years, loaded_temps = [], []
            for row in reader:
                year_text = (row.get("Year") or "").strip()
                value_text = (row.get("J-D") or "").strip()
                if not year_text or not value_text or value_text == "***":
                    continue
                loaded_years.append(int(year_text))
                loaded_temps.append(float(value_text))

            years = np.asarray(loaded_years, dtype=int)
            temps = np.asarray(loaded_temps, dtype=float)
            if len(years) > 100 and len(years) == len(temps) and np.all(np.diff(years) > 0):
                return years, temps, "NASA GISTEMP v4 (live download)"
            print("  Live data did not pass basic validation; using the saved copy.")
        except Exception as error:
            print(f"  Live download failed ({error.__class__.__name__}); using the saved copy.")

    years = np.arange(SAVED_START_YEAR, SAVED_START_YEAR + len(SAVED_TEMPS), dtype=int)
    temps = np.asarray(SAVED_TEMPS, dtype=float)
    return years, temps, "NASA GISTEMP v4 (saved 1880-2024 copy)"

# Linear algebra methods implemented for the project

def rref(matrix, tol=1e-10):
    R = np.array(matrix, dtype=float, copy=True)
    if R.ndim != 2:
        raise ValueError("RREF expects a two-dimensional matrix")

    n_rows, n_cols = R.shape
    pivot_cols = []
    pivot_row = 0
    for col in range(n_cols):
        if pivot_row >= n_rows:
            break
        best = pivot_row + int(np.argmax(np.abs(R[pivot_row:, col])))
        if abs(R[best, col]) < tol:
            continue
        R[[pivot_row, best]] = R[[best, pivot_row]]
        R[pivot_row] = R[pivot_row] / R[pivot_row, col]
        for other in range(n_rows):
            if other != pivot_row:
                R[other] -= R[other, col] * R[pivot_row]
        pivot_cols.append(col)
        pivot_row += 1
    return R, pivot_cols

def back_substitution(U, b, tol=1e-14):
    U = np.asarray(U, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = len(b)
    if U.shape != (n, n):
        raise ValueError("U must be square and have the same size as b")
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < tol:
            raise ValueError("Upper-triangular system has a zero or near-zero pivot")
        x[i] = (b[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x

def gauss_solve(A, b, tol=1e-14):
    A = np.array(A, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True).reshape(-1)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] != len(b):
        raise ValueError("A must be square and its size must match b")

    n = A.shape[0]
    for k in range(n):
        best = k + int(np.argmax(np.abs(A[k:, k])))
        if abs(A[best, k]) < tol:
            raise ValueError("Matrix is singular or numerically singular")
        A[[k, best]] = A[[best, k]]
        b[[k, best]] = b[[best, k]]
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]
    return back_substitution(A, b, tol=tol)

def lu_decompose(A, tol=1e-14):
    U = np.array(A, dtype=float, copy=True)
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("LU decomposition expects a square matrix")
    n = U.shape[0]
    L = np.eye(n)
    P = np.eye(n)

    for k in range(n - 1):
        best = k + int(np.argmax(np.abs(U[k:, k])))
        if abs(U[best, k]) < tol:
            raise ValueError("Matrix is singular or numerically singular")
        if best != k:
            U[[k, best]] = U[[best, k]]
            P[[k, best]] = P[[best, k]]
            L[[k, best], :k] = L[[best, k], :k]
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    if abs(U[-1, -1]) < tol:
        raise ValueError("Matrix is singular or numerically singular")
    return P, L, U

def lu_solve(P, L, U, b):
    b = np.asarray(b, dtype=float).reshape(-1)
    pb = P @ b
    n = len(pb)
    y = np.zeros(n)
    for i in range(n):
        if abs(L[i, i]) < 1e-14:
            raise ValueError("L has a zero diagonal entry")
        y[i] = (pb[i] - L[i, :i] @ y[:i]) / L[i, i]
    return back_substitution(U, y)

def gram_schmidt(A, tol=1e-12):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("Gram-Schmidt expects a two-dimensional matrix")
    n_rows, n_cols = A.shape
    Q = np.zeros((n_rows, n_cols))
    R = np.zeros((n_cols, n_cols))
    work = A.copy()
    scale = max(1.0, float(np.max(np.linalg.norm(A, axis=0))))

    for j in range(n_cols):
        R[j, j] = np.linalg.norm(work[:, j])
        if R[j, j] <= tol * scale:
            raise ValueError("Gram-Schmidt encountered dependent or near-zero columns")
        Q[:, j] = work[:, j] / R[j, j]
        for k in range(j + 1, n_cols):

            correction = Q[:, j] @ work[:, k]
            R[j, k] += correction
            work[:, k] -= correction * Q[:, j]

            correction2 = Q[:, j] @ work[:, k]
            R[j, k] += correction2
            work[:, k] -= correction2 * Q[:, j]
    return Q, R

def power_iteration(S, steps=500, tol=1e-14):
    S = np.asarray(S, dtype=float)
    v = np.ones(S.shape[0], dtype=float)
    v /= np.linalg.norm(v)
    for _ in range(steps):
        w = S @ v
        norm_w = np.linalg.norm(w)
        if norm_w < tol:
            raise ValueError("Power iteration reached a zero vector")
        new_v = w / norm_w
        if min(np.linalg.norm(new_v - v), np.linalg.norm(new_v + v)) < tol:
            v = new_v
            break
        v = new_v
    return float(v @ S @ v), v

def vandermonde(x, degree):
    x = np.asarray(x, dtype=float).reshape(-1)
    if degree < 0:
        raise ValueError("Polynomial degree must be non-negative")
    return np.column_stack([x ** power for power in range(degree + 1)])

def scale_years(years):
    return (np.asarray(years, dtype=float) - 1950.0) / 50.0

def fit_polynomial(scaled_years, values, degree):
    A = vandermonde(scaled_years, degree)
    Q, R = gram_schmidt(A)
    return back_substitution(R, Q.T @ np.asarray(values, dtype=float))

def rmse(predicted, actual):
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if predicted.size == 0 or predicted.shape != actual.shape:
        raise ValueError("RMSE needs non-empty arrays with matching shapes")
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))

def t_critical_975(df):
    if df <= 0:
        raise ValueError("Degrees of freedom must be positive")
    z = 1.959963984540054
    v = float(df)
    return float(
        z
        + (z**3 + z) / (4 * v)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * v**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * v**3)
    )

def header(title):
    """Print a clear heading and pause between sections in an interactive terminal."""
    if header.started:
        try:
            input("\n  Press Enter to continue to the next section... ")
        except EOFError:
            # Keep the script usable if the terminal input is closed mid-run.
            pass

    width = 78
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)
    header.started = True


header.started = False

# Main project pipeline

def main():
    header.started = False
    print("\nCLIMATE MATHEMATICS PROJECT")
    print("Follow each section at your own pace; press Enter when you are ready to continue.")
    header("STEP 1 - REAL-WORLD DATA")
    years, temps, source = load_data()
    if len(years) != len(temps) or len(years) < 100:
        raise RuntimeError("The temperature dataset is empty or inconsistent")
    if np.any(np.diff(years) <= 0):
        raise RuntimeError("Years must be sorted and unique")

    n = len(years)
    scaled = scale_years(years)
    print(f"  Source : {source}")
    print(f"  Years  : {years[0]}-{years[-1]} ({n} annual observations)")
    print("  Measurement: global annual mean temperature anomaly, relative to 1951-1980.")
    print(f"  Anomaly range: {temps.min():+.2f} to {temps.max():+.2f} C")

    header("STEP 2 - MATRIX REPRESENTATION (Vandermonde interpolation)")
    node_years = np.arange(1880, 1951, 10)
    year_to_index = {int(year): i for i, year in enumerate(years)}
    missing_nodes = [int(y) for y in node_years if int(y) not in year_to_index]
    if missing_nodes:
        raise RuntimeError(f"Required interpolation anchor years are missing: {missing_nodes}")

    node_index = np.array([year_to_index[int(y)] for y in node_years], dtype=int)
    node_x = scaled[node_index]
    node_temps = temps[node_index]
    interp_degree = len(node_years) - 1
    V = vandermonde(node_x, interp_degree)
    print(f"  Anchor years: {node_years.tolist()}")
    print(f"  Interpolation system: V c = y, with V shape {V.shape}")
    print(f"  Each row is [1, t, t^2, ..., t^{interp_degree}], where t=(year-1950)/50")
    print("  First three rows of V:")
    print(V[:3])
    interpolation_condition = float(np.linalg.cond(V))
    print(f"  Condition number of V: {interpolation_condition:.3e}")
    if interpolation_condition > 1e4:
        print("  Caution: this high condition number means the degree-7 interpolation coefficients")
        print("  may be sensitive to small changes in the anchor data (numerical conditioning).")
    print("\n  Regression design matrix A has one row per year and one column per polynomial feature.")
    print("  A maps a coefficient vector to fitted temperature anomalies: a linear transformation.")

    header("STEP 3 - MATRIX SIMPLIFICATION (RREF / Gaussian elimination / LU)")
    augmented = np.hstack([V, node_temps.reshape(-1, 1)])
    reduced, rref_pivots = rref(augmented)
    if rref_pivots[:V.shape[1]] != list(range(V.shape[1])):
        raise RuntimeError("Interpolation system did not reduce to a full-rank coefficient system")
    coef_rref = reduced[:, -1]
    coef_gauss = gauss_solve(V, node_temps)
    P_lu, L, U = lu_decompose(V)
    coef_lu = lu_solve(P_lu, L, U, node_temps)
    coef_numpy = np.linalg.solve(V, node_temps)

    print("  RREF of [V | y] (left block should become the identity):")
    print(reduced)
    print("  Coefficients from RREF  :", coef_rref)
    print("  Coefficients from Gauss :", coef_gauss)
    print("  Coefficients from LU    :", coef_lu)
    print(f"  Max coefficient difference (RREF vs NumPy): {np.max(np.abs(coef_rref - coef_numpy)):.2e}")
    print(f"  LU factorisation check ||P V - L U||: {np.linalg.norm(P_lu @ V - L @ U):.2e}")

    header("INTERPOLATION EXPERIMENT - FAIR COMPARISON ON THE SAME HELD-OUT YEARS")
    in_early_period = (years >= 1880) & (years <= 1950)
    between_anchor_years = in_early_period & ~np.isin(years, node_years)
    all_between_indices = np.where(between_anchor_years)[0]
    global_all_estimates = vandermonde(scaled[all_between_indices], interp_degree) @ coef_rref
    global_all_rmse = rmse(global_all_estimates, temps[all_between_indices])
    print(f"  Degree-{interp_degree} interpolant RMSE on all {len(all_between_indices)} between-anchor years: {global_all_rmse:.3f} C")

    eligible = np.where(
        in_early_period
        & ~np.isin(years, node_years)
        & (years >= 1883)
        & (years <= 1947)
    )[0]
    rng = np.random.default_rng(42)
    n_gaps = min(15, len(eligible))
    gap_index = np.sort(rng.choice(eligible, size=n_gaps, replace=False))

    global_gap_estimates = vandermonde(scaled[gap_index], interp_degree) @ coef_rref
    known_index = np.setdiff1d(np.arange(n), gap_index)
    local_gap_estimates = []
    for gap in gap_index:
        nearest_positions = np.argsort(np.abs(years[known_index] - years[gap]))[:4]
        nearest = known_index[nearest_positions]
        local_coefficients = gauss_solve(vandermonde(scaled[nearest], 3), temps[nearest])
        local_gap_estimates.append(float((vandermonde(scaled[gap:gap + 1], 3) @ local_coefficients)[0]))
    local_gap_estimates = np.asarray(local_gap_estimates)

    global_gap_rmse = rmse(global_gap_estimates, temps[gap_index])
    local_gap_rmse = rmse(local_gap_estimates, temps[gap_index])
    print(f"\n  Common held-out targets ({n_gaps} randomly selected years):")
    print(f"  Degree-{interp_degree} polynomial using eight decade-spaced anchors: RMSE = {global_gap_rmse:.3f} C")
    print(f"  Local cubic using four nearest available observations:         RMSE = {local_gap_rmse:.3f} C")
    print("  Both errors above are calculated on exactly the same withheld years.")
    print("  The methods use different amounts of available information, as defined by each strategy.")
    print("\n  Year    actual    global degree-7    local cubic")
    for gap, global_est, local_est in zip(gap_index, global_gap_estimates, local_gap_estimates):
        print(f"  {years[gap]}   {temps[gap]:+6.2f}       {global_est:+6.2f}          {local_est:+6.2f}")

    header("MODEL SELECTION - WALK-FORWARD VALIDATION")

    validation_cutoffs = [1980, 1990, 2000]
    validation_horizon = 10
    final_test_start = int(years[-1] - 9)
    usable_cutoffs = [
        cutoff for cutoff in validation_cutoffs
        if cutoff + validation_horizon < final_test_start
        and np.sum(years <= cutoff) > 10
        and np.sum((years > cutoff) & (years <= cutoff + validation_horizon)) >= 5
    ]
    if not usable_cutoffs:
        raise RuntimeError("Not enough data to perform chronological model validation")

    degrees = [1, 2, 3, 4, 5]
    validation_errors = {degree: [] for degree in degrees}
    print("  Each row reports RMSE over the 10 years after each historical cutoff.")
    print("  The validation windows are chronological and do not include the final holdout period.")
    print("  Degree  " + "".join(f"{cutoff:>10}" for cutoff in usable_cutoffs) + "   average")
    for degree in degrees:
        for cutoff in usable_cutoffs:
            train = years <= cutoff
            valid = (years > cutoff) & (years <= cutoff + validation_horizon)
            coefficients = fit_polynomial(scaled[train], temps[train], degree)
            predictions = vandermonde(scaled[valid], degree) @ coefficients
            validation_errors[degree].append(rmse(predictions, temps[valid]))
        average = float(np.mean(validation_errors[degree]))
        print(f"  {degree:^6}" + "".join(f"{e:10.3f}" for e in validation_errors[degree]) + f"   {average:7.3f}")

    mean_validation_error = {d: float(np.mean(errs)) for d, errs in validation_errors.items()}
    degree = min(mean_validation_error, key=mean_validation_error.get)
    print(f"\n  Selected degree: {degree} (lowest average walk-forward validation RMSE)")

    final_test = years >= final_test_start
    final_train = years < final_test_start
    if np.sum(final_test) < 5 or np.sum(final_train) <= degree + 1:
        raise RuntimeError("Not enough observations for the final holdout evaluation")
    test_coefficients = fit_polynomial(scaled[final_train], temps[final_train], degree)
    test_predictions = vandermonde(scaled[final_test], degree) @ test_coefficients
    final_test_rmse = rmse(test_predictions, temps[final_test])
    print(f"\n  Independent final holdout: {final_test_start}-{years[-1]} ({np.sum(final_test)} years)")
    print(f"  Training for this check uses only years before {final_test_start}.")
    print(f"  Holdout RMSE for selected degree-{degree} model: {final_test_rmse:.3f} C")
    print("  This holdout is not used to choose the polynomial degree.")

    header(f"STEP 4 - STRUCTURE OF THE SPACE (design matrix, degree {degree})")
    A = vandermonde(scaled, degree)
    p = A.shape[1]
    _, pivot_cols = rref(A)
    rank = len(pivot_cols)
    nullity = p - rank
    left_nullity = n - rank
    print(f"  A is {n}x{p}: {n} equations and {p} polynomial coefficients (overdetermined).")
    print(f"  rank(A) = {rank}; nullity(A) = {nullity}")
    print("  Four fundamental subspaces:")
    print(f"    Column space Col(A)     dimension {rank}, inside R^{n}")
    print(f"    Row space Row(A)        dimension {rank}, inside R^{p}")
    print(f"    Null space Null(A)      dimension {nullity}, inside R^{p}")
    print(f"    Left null Null(A^T)     dimension {left_nullity}, inside R^{n}")
    print(f"  Rank-nullity: {rank} + {nullity} = {p} columns")
    print(f"  Col(A) contains all sampled polynomial trend vectors of degree <= {degree}.")
    print("  If the observed vector is not in Col(A), no coefficient vector solves A x = y exactly;")
    print("  least squares finds the closest vector in Col(A).")

    header("STEP 5 - REMOVE REDUNDANCY (linear independence and basis selection)")
    redundant = np.hstack([vandermonde(scaled, 2), (2 * scaled + 1).reshape(-1, 1)])
    reduced_redundant, redundant_pivots = rref(redundant)
    print("  Example feature columns: [1, t, t^2, 2t+1]")
    print(f"  Pivot columns: {redundant_pivots}; rank {len(redundant_pivots)} for 4 columns, so dependent.")
    free_cols = [col for col in range(redundant.shape[1]) if col not in redundant_pivots]
    for free_col in free_cols:
        relation = np.zeros(redundant.shape[1])
        relation[free_col] = 1.0
        for row, pivot in enumerate(redundant_pivots):
            relation[pivot] = -reduced_redundant[row, free_col]
        print(f"  Dependency vector: {relation}")
        print(f"  ||A_redundant @ dependency|| = {np.linalg.norm(redundant @ relation):.2e}")
    print("  The dependency is the identity 2t+1 = 1 + 2*t.")

    raw_years_matrix = np.column_stack([np.ones(n), years.astype(float), years.astype(float) ** 2])
    scaled_years_matrix = vandermonde(scaled, 2)
    print("\n  Effect of scaling the polynomial input:")
    print(f"  condition number with raw [1, year, year^2]: {np.linalg.cond(raw_years_matrix):.2e}")
    print(f"  condition number with scaled [1, t, t^2] : {np.linalg.cond(scaled_years_matrix):.2f}")
    print(f"  Final design-matrix pivot columns: {pivot_cols}; all {p} columns are independent.")

    header("STEP 6 - ORTHOGONALIZATION (modified Gram-Schmidt / QR)")
    Q, R = gram_schmidt(A)
    print("  A = Q R, with orthonormal columns in Q.")
    print(f"  ||Q^T Q - I|| = {np.linalg.norm(Q.T @ Q - np.eye(p)):.2e}")
    print(f"  ||A - Q R||   = {np.linalg.norm(A - Q @ R):.2e}")
    print("  Upper-triangular R:")
    print(R)

    header("STEP 7 - ORTHOGONAL PROJECTION ONTO Col(A)")
    projector = Q @ Q.T
    fitted = projector @ temps
    residual = temps - fitted
    print(f"  Projection matrix P = Q Q^T, shape {projector.shape}.")
    print(f"  Idempotence ||P^2-P|| = {np.linalg.norm(projector @ projector - projector):.2e}")
    print(f"  Symmetry    ||P-P^T|| = {np.linalg.norm(projector - projector.T):.2e}")
    print(f"  trace(P) = {np.trace(projector):.4f}, equal to the model-space rank {rank} (up to rounding).")
    print(f"  Residual orthogonality ||A^T r|| = {np.linalg.norm(A.T @ residual):.2e}")
    print(f"  Pythagorean check: ||y||^2 = {temps @ temps:.4f}")
    print(f"                   ||Py||^2+||r||^2 = {fitted @ fitted + residual @ residual:.4f}")
    print("  The residual lies in Null(A^T), perpendicular to every fitted polynomial feature.")

    header("STEP 8 - PREDICTION / APPROXIMATION (least squares)")
    coef_normal = gauss_solve(A.T @ A, A.T @ temps)
    coef_qr = back_substitution(R, Q.T @ temps)
    coef_numpy = np.linalg.lstsq(A, temps, rcond=None)[0]
    print("  Normal equations (A^T A)x=A^T y:", coef_normal)
    print("  QR route R x=Q^T y            :", coef_qr)
    print("  NumPy least squares (check)     :", coef_numpy)
    difference = max(
        np.max(np.abs(coef_normal - coef_numpy)),
        np.max(np.abs(coef_qr - coef_numpy)),
    )
    print(f"  Maximum coefficient difference against NumPy: {difference:.2e}")

    rss = float(residual @ residual)
    degrees_freedom = n - p
    sigma = float(np.sqrt(rss / degrees_freedom))
    r_squared = float(1.0 - rss / np.sum((temps - temps.mean()) ** 2))
    fit_rmse = float(np.sqrt(rss / n))
    print(f"  In-sample RMSE = {fit_rmse:.4f} C; R^2 = {r_squared:.4f}; residual sigma = {sigma:.4f} C")
    coefficients = coef_qr

    header("STEP 9 - EIGENVALUES, POSITIVE DEFINITENESS AND SVD")
    G = A.T @ A
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    print("  G=A^T A is the Gram matrix of the polynomial features, not a climate covariance matrix.")
    print("  It is symmetric, so it has real eigenvalues and orthogonal eigenvectors.")
    print("  Eigenvalues (descending):", eigenvalues)
    power_value, _ = power_iteration(G)
    print(f"  Power iteration largest eigenvalue: {power_value:.6f}; eigh result: {eigenvalues[0]:.6f}")
    condition_number = float(np.sqrt(eigenvalues[0] / eigenvalues[-1]))
    print(f"  Condition number estimate of A: {condition_number:.4f}")
    print("  Small eigenvalues indicate less strongly constrained coefficient directions;")
    print("  such directions can make extrapolation sensitive to the fitted model.")
    print("  Eigen-direction coefficient-uncertainty scale under the classical iid model:")
    for i, value in enumerate(eigenvalues):
        uncertainty = sigma / np.sqrt(value) if value > 0 else float("inf")
        print(f"    direction {i + 1}: eigenvalue={value:.6g}, scale={uncertainty:.6g}")

    positive_tol = 1e-12 * max(1.0, float(np.max(np.abs(eigenvalues))))
    positive_definite = bool(np.all(eigenvalues > positive_tol))
    print("\n  Positive-definiteness check for G:")
    print(f"  All eigenvalues greater than numerical tolerance? {positive_definite}")
    try:
        np.linalg.cholesky(G)
        cholesky_ok = True
    except np.linalg.LinAlgError:
        cholesky_ok = False
    print(f"  Cholesky factorisation succeeds? {cholesky_ok}")
    print("  Since A has full column rank, G=A^T A is theoretically positive definite.")

    print("\n  Singular Value Decomposition of A = U S V^T:")
    U_svd, singular_values, Vt_svd = np.linalg.svd(A, full_matrices=False)
    svd_tolerance = np.finfo(float).eps * max(A.shape) * singular_values[0]
    inverse_singular_values = np.array([
        1.0 / value if value > svd_tolerance else 0.0 for value in singular_values
    ])
    coef_svd = Vt_svd.T @ (inverse_singular_values * (U_svd.T @ temps))
    print("  Singular values:", singular_values)
    print("  sqrt(eigenvalues of A^T A):", np.sqrt(np.maximum(eigenvalues, 0.0)))
    print("  Least-squares coefficients from SVD:", coef_svd)
    print(f"  Maximum difference from QR solution: {np.max(np.abs(coef_svd - coefficients)):.2e}")

    header("STEP 10 - SYSTEM SIMPLIFICATION (orthogonal diagonalisation and inverse)")
    Lambda = np.diag(eigenvalues)
    reconstruction_error = np.linalg.norm(G - eigenvectors @ Lambda @ eigenvectors.T)
    print(f"  G = E Lambda E^T; reconstruction error = {reconstruction_error:.2e}")
    if not positive_definite:
        raise RuntimeError("G is not numerically positive definite; inverse-based steps are unsafe")
    G_inverse_eigen = eigenvectors @ np.diag(1.0 / eigenvalues) @ eigenvectors.T
    coef_diagonal = G_inverse_eigen @ (A.T @ temps)
    print("  Least squares using diagonalisation:", coef_diagonal)
    print(f"  Maximum difference from QR solution: {np.max(np.abs(coef_diagonal - coefficients)):.2e}")

    augmented_inverse = np.hstack([G, np.eye(p)])
    reduced_inverse, inverse_pivots = rref(augmented_inverse, tol=1e-12)
    if inverse_pivots[:p] != list(range(p)):
        raise RuntimeError("Gauss-Jordan inversion failed: G did not have full pivot rank")
    G_inverse_gauss_jordan = reduced_inverse[:, p:]
    inverse_error = np.linalg.norm(G @ G_inverse_gauss_jordan - np.eye(p))
    print("\n  Inverse of G by Gauss-Jordan row reduction of [G | I]:")
    print(G_inverse_gauss_jordan)
    print(f"  Inverse check ||G G_inverse - I|| = {inverse_error:.2e}")
    print(f"  Difference from eigendecomposition inverse: {np.max(np.abs(G_inverse_gauss_jordan - G_inverse_eigen)):.2e}")
    G_inverse = G_inverse_gauss_jordan

    header("FINAL OUTPUT - EXTRAPOLATION / CLIMATE TREND PROJECTION")
    target_years = np.array([2030, 2050, 2075, 2100], dtype=int)
    X_future = vandermonde(scale_years(target_years), degree)
    prediction = X_future @ coefficients

    t_crit = t_critical_975(degrees_freedom)
    leverage_future = np.einsum("ij,jk,ik->i", X_future, G_inverse, X_future)
    mean_se = sigma * np.sqrt(np.maximum(leverage_future, 0.0))
    prediction_se = sigma * np.sqrt(np.maximum(1.0 + leverage_future, 0.0))
    mean_low, mean_high = prediction - t_crit * mean_se, prediction + t_crit * mean_se
    pred_low, pred_high = prediction - t_crit * prediction_se, prediction + t_crit * prediction_se

    print(f"  Selected model: degree-{degree} polynomial with t=(year-1950)/50")
    print("  Forecasts are anomalies relative to the 1951-1980 average, not warming relative to today.")
    print(f"  Approximate t critical value for 95% intervals (df={degrees_freedom}): {t_crit:.4f}")
    print("\n  Year   fitted anomaly   95% mean-trend interval   95% single-year interval (iid model)")
    for year, value, ml, mh, pl, ph in zip(target_years, prediction, mean_low, mean_high, pred_low, pred_high):
        print(f"  {year}    {value:+7.2f} C      [{ml:+.2f}, {mh:+.2f}] C             [{pl:+.2f}, {ph:+.2f}] C")

    if len(residual) > 2 and np.std(residual[:-1]) > 0 and np.std(residual[1:]) > 0:
        lag_one_corr = float(np.corrcoef(residual[:-1], residual[1:])[0, 1])
    else:
        lag_one_corr = float("nan")
    print(f"\n  Lag-1 residual correlation: {lag_one_corr:.3f}")
    print("  Positive residual correlation would contradict the iid-error assumption behind these bands.")
    print("  These intervals omit serial-correlation, model-choice, structural-change and emissions-scenario uncertainty;")
    print("  they should not be interpreted as complete climate forecast ranges.")

    print("\n  Forecast sensitivity to polynomial degree (each model fit on all available historical data):")
    print("  Degree" + "".join(f"{year:>10}" for year in target_years))
    sensitivity_forecasts = {}
    for candidate_degree in (1, 2, 3):
        candidate_coef = fit_polynomial(scaled, temps, candidate_degree)
        candidate_prediction = vandermonde(scale_years(target_years), candidate_degree) @ candidate_coef
        sensitivity_forecasts[candidate_degree] = candidate_prediction
        print(f"  {candidate_degree:>6}" + "".join(f"{value:10.2f}" for value in candidate_prediction))
    outside = [
        candidate_degree for candidate_degree, values in sensitivity_forecasts.items()
        if np.any((values < pred_low) | (values > pred_high))
    ]
    if outside:
        print(f"  Degree(s) {outside} produce forecasts outside the selected model's iid interval.")
        print("  This is evidence that model choice can matter beyond the displayed regression interval.")

    linear_all = np.polyfit(years.astype(float), temps, 1)
    recent_mask = years >= 1980
    linear_recent = np.polyfit(years[recent_mask].astype(float), temps[recent_mask], 1)
    print(f"\n  Straight-line slope over {years[0]}-{years[-1]}: {linear_all[0] * 10:.3f} C per decade")
    print(f"  Straight-line slope since 1980: {linear_recent[0] * 10:.3f} C per decade")
    print("\n  Limitation: polynomial extrapolation assumes the past curve shape continues.")
    print("  It does not model future emissions, climate feedbacks, volcanic effects, or ENSO explicitly.")
    print("  The outputs are mathematical trend projections, not results from a physical climate model.")

    # Plots

    header("STEP 11 - VISUAL OUTPUTS")

    if not HAVE_PLOTS:
        print("\n  matplotlib is not installed; calculations completed without plots.")
        print("  Install plotting support with: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(years[in_early_period], temps[in_early_period], "o", ms=4, color="grey", label="Observed (1880-1950)")
    curve_t = np.linspace(scale_years(1880), scale_years(1950), 300)
    ax.plot(curve_t * 50 + 1950, vandermonde(curve_t, interp_degree) @ coef_rref,
            color="crimson", label="Degree-7 global interpolant")
    ax.plot(node_years, node_temps, "ks", label="Eight interpolation anchors")
    ax.scatter(years[gap_index], global_gap_estimates, marker="x", s=55, color="darkorange",
               label="Global predictions at common gaps")
    ax.scatter(years[gap_index], local_gap_estimates, marker="^", s=45, color="green",
               label="Local cubic predictions at common gaps")
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature anomaly (C)")
    ax.set_title("Historical interpolation: global polynomial vs local cubic")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "interpolation.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(years, temps, ".", color="grey", label="NASA annual anomaly")
    display_years = np.arange(years[0], 2101)
    display_scaled = scale_years(display_years)
    ax.plot(display_years, vandermonde(display_scaled, degree) @ coefficients,
            color="crimson", lw=2, label=f"Selected least-squares degree-{degree}")
    for candidate_degree, style in ((1, "--"), (3, ":")):
        if candidate_degree != degree:
            candidate_coef = fit_polynomial(scaled, temps, candidate_degree)
            candidate_curve = vandermonde(display_scaled, candidate_degree) @ candidate_coef
            ax.plot(display_years, candidate_curve, style, label=f"Degree-{candidate_degree} comparison")

    future_display_years = np.arange(years[-1], 2101)
    X_band = vandermonde(scale_years(future_display_years), degree)
    band_leverage = np.einsum("ij,jk,ik->i", X_band, G_inverse, X_band)
    band_se = sigma * np.sqrt(np.maximum(1.0 + band_leverage, 0.0))
    band_center = X_band @ coefficients
    ax.fill_between(future_display_years, band_center - t_crit * band_se,
                    band_center + t_crit * band_se, color="crimson", alpha=0.15,
                    label="Approx. 95% iid regression band (incomplete uncertainty)")
    ax.axvline(years[-1], color="black", lw=0.8, label="Last observed year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Anomaly vs 1951-1980 (C)")
    ax.set_title("Global temperature trend extrapolation to 2100")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "forecast.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stem(years, residual, basefmt=" ")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Least-squares residuals (orthogonal to Col(A))")
    ax.set_xlabel("Year")
    ax.set_ylabel("Residual (C)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "residuals.png"), dpi=150)
    plt.close(fig)

    print(f"\n  Plots saved to ./{OUTPUT_FOLDER}/:")
    print("    interpolation.png, forecast.png, residuals.png")

if __name__ == "__main__":
    main()
