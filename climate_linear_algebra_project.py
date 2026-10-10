"""
UE25MA242A - Mathematical Foundation for AI & Data Science
MINI PROJECT - Problem 8: Interpolation, Extrapolation, and Climate Change
Idea: Global Temperature Trend Analysis and Future Forecasting

Workflow (follows the department guideline diagram):
  Real-world data -> Matrix representation (Vandermonde)
  -> Matrix simplification (Gaussian elimination / RREF / LU)
  -> Structure of the space (rank, nullity, subspaces)
  -> Remove redundancy (independence, basis selection)
  -> Orthogonalization (Gram-Schmidt, QR)
  -> Projection (P = Q Q^T)
  -> Prediction / approximation (least squares)
  -> Pattern discovery (eigenvalues & eigenvectors)
  -> System simplification (orthogonal diagonalization of A^T A)
  -> Final output (forecast of temperature anomaly for 2050 / 2100)

Data: NASA GISS Surface Temperature Analysis (GISTEMP v4), global annual mean
anomaly (deg C, relative to 1951-1980 average). The script downloads the live
file; if there is no internet it falls back to an embedded copy.

Run:  python climate_linear_algebra_project.py
Needs: numpy (required), matplotlib (optional, for plots)
"""

import csv
import io
import os
import urllib.request

import numpy as np  # pyright: ignore[reportMissingImports]

try:
    import matplotlib  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False

np.set_printoptions(precision=4, suppress=True, linewidth=120)
OUT_DIR = "project_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

GISTEMP_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"

# Backup copy of NASA GISTEMP annual (Jan-Dec) global anomalies, 1880-2024,
# rounded to 0.01 C. Used only when the live download fails.
_BACKUP_START = 1880
_BACKUP = [
    -0.17, -0.09, -0.11, -0.18, -0.28, -0.33, -0.31, -0.36, -0.17, -0.10,  # 1880s
    -0.35, -0.22, -0.27, -0.31, -0.30, -0.23, -0.11, -0.11, -0.27, -0.18,  # 1890s
    -0.09, -0.16, -0.28, -0.37, -0.47, -0.26, -0.22, -0.39, -0.43, -0.48,  # 1900s
    -0.43, -0.44, -0.36, -0.35, -0.16, -0.14, -0.36, -0.46, -0.30, -0.27,  # 1910s
    -0.27, -0.19, -0.28, -0.26, -0.27, -0.22, -0.11, -0.22, -0.20, -0.36,  # 1920s
    -0.16, -0.09, -0.16, -0.29, -0.13, -0.20, -0.15, -0.03, -0.01, -0.02,  # 1930s
    0.13, 0.19, 0.07, 0.09, 0.20, 0.09, -0.07, -0.03, -0.11, -0.11,        # 1940s
    -0.17, -0.07, 0.01, 0.08, -0.13, -0.14, -0.19, 0.05, 0.06, 0.03,       # 1950s
    -0.03, 0.06, 0.03, 0.05, -0.20, -0.11, -0.06, -0.02, -0.08, 0.05,      # 1960s
    0.03, -0.08, 0.01, 0.16, -0.07, -0.01, -0.10, 0.18, 0.07, 0.16,        # 1970s
    0.26, 0.32, 0.14, 0.31, 0.16, 0.12, 0.18, 0.32, 0.39, 0.27,            # 1980s
    0.45, 0.40, 0.22, 0.23, 0.32, 0.45, 0.33, 0.46, 0.61, 0.38,            # 1990s
    0.39, 0.54, 0.63, 0.62, 0.53, 0.68, 0.64, 0.66, 0.54, 0.65,            # 2000s
    0.72, 0.61, 0.65, 0.68, 0.75, 0.90, 1.01, 0.92, 0.85, 0.98,            # 2010s
    1.02, 0.85, 0.89, 1.17, 1.28,                                          # 2020-24
]


# ----------------------------------------------------------------------------
# Linear algebra toolbox (written from scratch so every step is visible)
# ----------------------------------------------------------------------------
def rref(M, tol=1e-10):
    """Reduced row echelon form with partial pivoting. Returns (R, pivot_columns)."""
    R = np.array(M, dtype=float)
    rows, cols = R.shape
    pivots, r = [], 0
    for c in range(cols):
        if r >= rows:
            break
        p = r + int(np.argmax(np.abs(R[r:, c])))
        if abs(R[p, c]) < tol:
            continue
        R[[r, p]] = R[[p, r]]
        R[r] = R[r] / R[r, c]
        for i in range(rows):
            if i != r:
                R[i] = R[i] - R[i, c] * R[r]
        pivots.append(c)
        r += 1
    return R, pivots


def gauss_solve(A, b):
    """Solve square system Ax=b by Gaussian elimination + back substitution."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    n = A.shape[0]
    M = np.hstack([A, b])
    for k in range(n):
        p = k + int(np.argmax(np.abs(M[k:, k])))
        if abs(M[p, k]) < 1e-14:
            raise ValueError("Matrix is singular")
        M[[k, p]] = M[[p, k]]
        for i in range(k + 1, n):
            M[i, k:] -= (M[i, k] / M[k, k]) * M[k, k:]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - M[i, i + 1:n] @ x[i + 1:n]) / M[i, i]
    return x


def lu_decompose(A):
    """PA = LU with partial pivoting. Returns P, L, U."""
    A = np.array(A, dtype=float)
    n = A.shape[0]
    U, L, P = A.copy(), np.eye(n), np.eye(n)
    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(U[k:, k])))
        if p != k:
            U[[k, p]] = U[[p, k]]
            P[[k, p]] = P[[p, k]]
            L[[k, p], :k] = L[[p, k], :k]
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    return P, L, U


def lu_solve(P, L, U, b):
    """Solve Ax=b given PA=LU (forward then backward substitution)."""
    pb = P @ np.asarray(b, dtype=float)
    n = len(pb)
    y = np.zeros(n)
    for i in range(n):
        y[i] = pb[i] - L[i, :i] @ y[:i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def gram_schmidt(A):
    """Modified Gram-Schmidt. Returns Q (orthonormal columns) and R (upper triangular), A = QR."""
    A = np.array(A, dtype=float)
    m, n = A.shape
    Q, R = np.zeros((m, n)), np.zeros((n, n))
    V = A.copy()
    for j in range(n):
        R[j, j] = np.linalg.norm(V[:, j])
        Q[:, j] = V[:, j] / R[j, j]
        for k in range(j + 1, n):
            R[j, k] = Q[:, j] @ V[:, k]
            V[:, k] -= R[j, k] * Q[:, j]
    return Q, R


def back_sub(R, y):
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - R[i, i + 1:] @ x[i + 1:]) / R[i, i]
    return x


def power_iteration(S, iters=500):
    """Dominant eigenpair of a symmetric matrix."""
    v = np.ones(S.shape[0]) / np.sqrt(S.shape[0])
    for _ in range(iters):
        w = S @ v
        v = w / np.linalg.norm(w)
    return v @ S @ v, v


def vandermonde(t, degree):
    """Rows [1, t, t^2, ..., t^degree]."""
    return np.vander(np.asarray(t, dtype=float), degree + 1, increasing=True)


def to_t(years):
    """Scale years to keep the Vandermonde matrix well conditioned."""
    return (np.asarray(years, dtype=float) - 1950.0) / 50.0


def poly_eval(coef, t):
    return vandermonde(t, len(coef) - 1) @ coef


def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ----------------------------------------------------------------------------
# STEP 1: Real-world data
# ----------------------------------------------------------------------------
def load_data():
    try:
        with urllib.request.urlopen(GISTEMP_URL, timeout=15) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
        lines = text.splitlines()
        start = next(i for i, ln in enumerate(lines) if ln.startswith("Year"))
        reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
        years, vals = [], []
        for row in reader:
            v = row.get("J-D", "").strip()
            if v and v != "***":
                years.append(int(row["Year"]))
                vals.append(float(v))
        if len(years) > 100:
            return np.array(years), np.array(vals), "NASA GISTEMP v4 (live download)"
    except Exception as exc:  # no internet, blocked, format change ...
        print(f"  (live download failed: {exc.__class__.__name__}; using embedded backup)")
    years = np.arange(_BACKUP_START, _BACKUP_START + len(_BACKUP))
    return years, np.array(_BACKUP), "NASA GISTEMP v4 (embedded backup copy)"


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def main():
    banner("STEP 1 - REAL-WORLD DATA")
    years, temp, source = load_data()
    n = len(years)
    print(f"  Source : {source}")
    print(f"  Years  : {years[0]}-{years[-1]}  ({n} annual values)")
    print(f"  Anomaly range: {temp.min():+.2f} to {temp.max():+.2f} C (baseline 1951-1980)")
    t = to_t(years)

    # ------------------------------------------------------------------
    banner("STEP 2 - MATRIX REPRESENTATION (system of linear equations)")
    # Interpolation system: 8 nodes, one every 10 years from 1880 -> 1950
    node_years = np.arange(1880, 1951, 10)
    node_idx = np.array([np.where(years == y)[0][0] for y in node_years])
    node_t, node_y = t[node_idx], temp[node_idx]
    V = vandermonde(node_t, len(node_years) - 1)
    print("  Interpolation system  V c = y  (Vandermonde, 8 early-record nodes):")
    print(f"  V is {V.shape[0]}x{V.shape[1]};  row i = [1, t_i, t_i^2, ..., t_i^7]")
    print("  First 3 rows of V:\n", V[:3])
    print(f"  Condition number of V = {np.linalg.cond(V):.3e}")

    # ------------------------------------------------------------------
    banner("STEP 3 - MATRIX SIMPLIFICATION (Gaussian elimination / RREF / LU)")
    aug = np.hstack([V, node_y.reshape(-1, 1)])
    R, piv = rref(aug)
    c_rref = R[:, -1]
    print("  RREF of augmented matrix [V | y] (left block becomes identity):")
    print(R)
    print("  Coefficients from RREF  :", c_rref)
    c_gauss = gauss_solve(V, node_y)
    P, L, U = lu_decompose(V)
    c_lu = lu_solve(P, L, U, node_y)
    c_np = np.linalg.solve(V, node_y)
    print("  Coefficients Gauss      :", c_gauss)
    print("  Coefficients LU         :", c_lu)
    print(f"  max |RREF - numpy| = {np.max(np.abs(c_rref - c_np)):.2e},   "
          f"||PV - LU|| = {np.linalg.norm(P @ V - L @ U):.2e}")

    # Interpolation: estimate the "missing" in-between years
    mask_early = (years >= 1880) & (years <= 1950)
    missing = mask_early & ~np.isin(years, node_years)
    est_global = poly_eval(c_rref, t[missing])
    rmse_global = np.sqrt(np.mean((est_global - temp[missing]) ** 2))
    print(f"\n  INTERPOLATION: treated {missing.sum()} early years as 'missing'.")
    print(f"  Degree-7 polynomial RMSE on missing years = {rmse_global:.3f} C")

    # Local interpolation (cubic through the 4 nearest known years) for random gaps
    rng = np.random.default_rng(42)
    early_idx = np.where(mask_early)[0][3:-3]
    gap_idx = np.sort(rng.choice(early_idx, size=15, replace=False))
    known = np.setdiff1d(np.arange(n), gap_idx)
    gap_est = []
    for g in gap_idx:
        near = known[np.argsort(np.abs(known - g))[:4]]
        Vl = vandermonde(t[near], 3)
        cl = gauss_solve(Vl, temp[near])
        gap_est.append(poly_eval(cl, t[g:g + 1])[0])
    gap_est = np.array(gap_est)
    rmse_local = np.sqrt(np.mean((gap_est - temp[gap_idx]) ** 2))
    print(f"  Local cubic gap-filling on 15 random missing years: RMSE = {rmse_local:.3f} C")
    print("  Year  true   estimated")
    for g, e in zip(gap_idx, gap_est):
        print(f"  {years[g]}  {temp[g]:+.2f}  {e:+.2f}")

    # ------------------------------------------------------------------
    # Model selection for the regression model (uses QR least squares)
    banner("MODEL SELECTION - which polynomial degree to extrapolate with?")
    train = years <= 2000
    test = ~train
    print("  Train on 1880-2000, test on 2001 onwards (a genuine extrapolation test)")
    print("  degree   train RMSE   test RMSE")
    scores = {}
    for d in range(1, 6):
        A_tr = vandermonde(t[train], d)
        Qd, Rd = gram_schmidt(A_tr)
        cd = back_sub(Rd, Qd.T @ temp[train])
        tr = np.sqrt(np.mean((A_tr @ cd - temp[train]) ** 2))
        te = np.sqrt(np.mean((vandermonde(t[test], d) @ cd - temp[test]) ** 2))
        scores[d] = te
        print(f"  {d:^6}   {tr:^10.4f}   {te:^9.4f}")
    DEG = min(scores, key=scores.get)
    print(f"  -> chosen degree = {DEG} (lowest out-of-sample error)")

    # ------------------------------------------------------------------
    banner(f"STEP 4 - STRUCTURE OF THE SPACE (design matrix A, degree {DEG})")
    A = vandermonde(t, DEG)      # n x (DEG+1): the overdetermined system A x = y
    p = A.shape[1]
    _, pivA = rref(A)
    rank = len(pivA)
    print(f"  A is {n}x{p}  ->  {n} equations, {p} unknowns (overdetermined)")
    print(f"  rank(A) = {rank}   nullity(A) = {p - rank}   "
          f"dim Col(A) = {rank}   dim Null(A^T) = {n - rank}")
    print(f"  Rank-nullity check: rank + nullity = {rank + (p - rank)} = number of columns {p}")
    print("  Meaning: Col(A) is a %d-dimensional subspace (all polynomial trends of degree <= %d)"
          " living inside R^%d;" % (rank, DEG, n))
    print("  the observed data vector y does NOT lie in it (noise) -> no exact solution.")

    # ------------------------------------------------------------------
    banner("STEP 5 - REMOVE REDUNDANCY (linear independence -> basis selection)")
    # Demonstrate with a deliberately redundant model [1, t, t^2, 2t+1]
    A_red = np.hstack([vandermonde(t, 2), (2 * t + 1).reshape(-1, 1)])
    Rr, piv_r = rref(A_red)
    print("  Candidate features: [1, t, t^2, (2t+1)]  -> 4th column is redundant.")
    print(f"  RREF pivot columns: {piv_r}  (rank {len(piv_r)} < 4 columns => dependent)")
    free = [c for c in range(A_red.shape[1]) if c not in piv_r]
    for f in free:
        null_vec = np.zeros(A_red.shape[1])
        null_vec[f] = 1
        for row, pc in enumerate(piv_r):
            null_vec[pc] = -Rr[row, f]
        print(f"  Null-space vector (dependency relation): {null_vec}  -> A_red @ v ~ "
              f"{np.linalg.norm(A_red @ null_vec):.1e}")
    print(f"  Basis selected (pivot columns kept): {piv_r}")
    print(f"  Our actual model has pivots {pivA}: all {p} columns independent -> they form a basis.")

    # ------------------------------------------------------------------
    banner("STEP 6 - ORTHOGONALIZATION (Gram-Schmidt -> orthonormal basis)")
    Q, Rq = gram_schmidt(A)
    print("  A = Q R, with Q's columns an orthonormal basis of Col(A)")
    print(f"  ||Q^T Q - I|| = {np.linalg.norm(Q.T @ Q - np.eye(p)):.2e}")
    print(f"  ||A - QR||    = {np.linalg.norm(A - Q @ Rq):.2e}")
    print("  R (upper triangular):\n", Rq)

    # ------------------------------------------------------------------
    banner("STEP 7 - PROJECTION onto Col(A)")
    P_proj = Q @ Q.T
    y_hat = P_proj @ temp
    resid = temp - y_hat
    print(f"  Projection matrix P = Q Q^T  ({P_proj.shape[0]}x{P_proj.shape[1]})")
    print(f"  Idempotent  ||P^2 - P||      = {np.linalg.norm(P_proj @ P_proj - P_proj):.2e}")
    print(f"  Symmetric   ||P - P^T||      = {np.linalg.norm(P_proj - P_proj.T):.2e}")
    print(f"  trace(P) = {np.trace(P_proj):.4f}  (= rank {rank})")
    print(f"  Residual is orthogonal to Col(A): ||A^T r|| = {np.linalg.norm(A.T @ resid):.2e}")
    print(f"  Pythagoras: ||y||^2 = ||y_hat||^2 + ||r||^2 -> "
          f"{temp @ temp:.4f} = {y_hat @ y_hat + resid @ resid:.4f}")

    # ------------------------------------------------------------------
    banner("STEP 8 - PREDICTION / APPROXIMATION (Least Squares solution)")
    x_normal = gauss_solve(A.T @ A, A.T @ temp)           # normal equations
    x_qr = back_sub(Rq, Q.T @ temp)                       # via Gram-Schmidt QR
    x_np = np.linalg.lstsq(A, temp, rcond=None)[0]        # reference
    print("  Normal equations (A^T A) x = A^T y :", x_normal)
    print("  QR route  R x = Q^T y              :", x_qr)
    print("  numpy.linalg.lstsq (check)         :", x_np)
    print(f"  max difference = {max(np.max(np.abs(x_normal - x_np)), np.max(np.abs(x_qr - x_np))):.2e}")
    rss = float(resid @ resid)
    sigma = np.sqrt(rss / (n - p))
    r2 = 1 - rss / float(((temp - temp.mean()) ** 2).sum())
    print(f"  Fit quality: RMSE = {np.sqrt(rss / n):.4f} C,  R^2 = {r2:.4f},  sigma_hat = {sigma:.4f} C")
    x_ls = x_qr

    # ------------------------------------------------------------------
    banner("STEP 9 - PATTERN DISCOVERY (eigenvalues & eigenvectors of A^T A)")
    G = A.T @ A
    lam, vecs = np.linalg.eigh(G)
    order = np.argsort(lam)[::-1]
    lam, vecs = lam[order], vecs[:, order]
    print("  A^T A is symmetric -> real eigenvalues, orthogonal eigenvectors")
    print("  Eigenvalues (descending):", lam)
    print("  Eigenvector matrix (columns):\n", vecs)
    lam_pi, _ = power_iteration(G)
    print(f"  Power iteration (dominant eigenvalue) = {lam_pi:.4f}  vs  eigh = {lam[0]:.4f}")
    cond = np.sqrt(lam[0] / lam[-1])
    print(f"  cond(A) = sqrt(lam_max / lam_min) = {cond:.2f}")
    share = lam / lam.sum() * 100
    print("  Share of total 'energy' per eigen-direction (%):", share)

    # ------------------------------------------------------------------
    banner("STEP 10 - SYSTEM SIMPLIFICATION (orthogonal diagonalization)")
    Lam = np.diag(lam)
    print(f"  A^T A = Q_e Lambda Q_e^T,  ||A^T A - Q_e Lambda Q_e^T|| = "
          f"{np.linalg.norm(G - vecs @ Lam @ vecs.T):.2e}")
    x_diag = vecs @ np.diag(1.0 / lam) @ vecs.T @ (A.T @ temp)
    print("  Least squares via diagonalization x = Q_e Lambda^-1 Q_e^T A^T y:", x_diag)
    print(f"  max |x_diag - x_ls| = {np.max(np.abs(x_diag - x_ls)):.2e}")
    G_inv = vecs @ np.diag(1.0 / lam) @ vecs.T

    # ------------------------------------------------------------------
    banner("FINAL APPLICATION OUTPUT - EXTRAPOLATION / CLIMATE FORECAST")
    target_years = np.array([2030, 2050, 2075, 2100])
    Xf = vandermonde(to_t(target_years), DEG)
    pred = Xf @ x_ls
    se = sigma * np.sqrt(1 + np.einsum("ij,jk,ik->i", Xf, G_inv, Xf))
    print(f"  Chosen model: degree-{DEG} polynomial, coefficients (in scaled time t=(year-1950)/50):")
    print("   ", x_ls)
    print("\n  Year   Predicted anomaly   approx. 95% band")
    for y_, pr, s in zip(target_years, pred, se):
        print(f"  {y_}   {pr:+7.2f} C          [{pr - 2 * s:+.2f}, {pr + 2 * s:+.2f}]")
    print("\n  Sensitivity to model choice (same least-squares machinery):")
    print("  degree  " + "  ".join(f"{y_:>7}" for y_ in target_years))
    sens = {}
    for d in (1, 2, 3):
        Ad = vandermonde(t, d)
        Qd, Rd = gram_schmidt(Ad)
        cd = back_sub(Rd, Qd.T @ temp)
        sens[d] = vandermonde(to_t(target_years), d) @ cd
        print(f"  {d:^6}  " + "  ".join(f"{v:+7.2f}" for v in sens[d]))
    lin = np.polyfit(years, temp, 1)
    print(f"\n  Linear trend slope = {lin[0] * 10:.3f} C per decade over {years[0]}-{years[-1]}")
    recent = years >= 1980
    lin_r = np.polyfit(years[recent], temp[recent], 1)
    print(f"  Since 1980 the slope is {lin_r[0] * 10:.3f} C per decade")
    print("\n  CAUTION: polynomial extrapolation assumes the past trend shape continues. It ignores")
    print("  emissions policy, feedbacks and volcanic/ENSO variability, so treat 2050/2100 values as")
    print("  a mathematical projection of the data, not a physical climate model.")

    # ------------------------------------------------------------------
    if HAVE_PLT:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(years[mask_early], temp[mask_early], "o", ms=4, color="grey", label="Actual (1880-1950)")
        tt = np.linspace(t[0], to_t(1950), 300)
        ax.plot(tt * 50 + 1950, poly_eval(c_rref, tt), color="crimson", label="Degree-7 interpolant (RREF)")
        ax.plot(node_years, node_y, "ks", label="Interpolation nodes")
        ax.plot(years[gap_idx], gap_est, "g^", label="Local cubic gap estimates")
        ax.set_xlabel("Year")
        ax.set_ylabel("Anomaly (C)")
        ax.set_title("Interpolation of early-record temperatures")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "interpolation.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(years, temp, ".", color="grey", label="NASA annual anomaly")
        fy = np.arange(years[0], 2101)
        ax.plot(fy, vandermonde(to_t(fy), DEG) @ x_ls, color="crimson", lw=2,
                label=f"Least-squares degree-{DEG}")
        for d, ls in ((1, "--"), (3, ":")):
            if d != DEG:
                Ad = vandermonde(t, d)
                Qd, Rd = gram_schmidt(Ad)
                cd = back_sub(Rd, Qd.T @ temp)
                ax.plot(fy, vandermonde(to_t(fy), d) @ cd, ls, label=f"Degree-{d} (comparison)")
        yy = np.arange(years[-1], 2101)
        Xb = vandermonde(to_t(yy), DEG)
        sb = sigma * np.sqrt(1 + np.einsum("ij,jk,ik->i", Xb, G_inv, Xb))
        ax.fill_between(yy, Xb @ x_ls - 2 * sb, Xb @ x_ls + 2 * sb, color="crimson", alpha=0.15,
                        label="~95% band")
        ax.axvline(years[-1], color="k", lw=0.8)
        ax.set_xlabel("Year")
        ax.set_ylabel("Anomaly vs 1951-1980 (C)")
        ax.set_title("Global temperature: least-squares trend and forecast to 2100")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "forecast.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.stem(years, resid, basefmt=" ")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title("Residuals (perpendicular to Col(A)) - the 'noise' removed by projection")
        ax.set_xlabel("Year")
        ax.set_ylabel("C")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "residuals.png"), dpi=150)
        plt.close(fig)
        print(f"\n  Plots saved in ./{OUT_DIR}/ (interpolation.png, forecast.png, residuals.png)")
    else:
        print("\n  (Install matplotlib to also get plots:  pip install matplotlib)")


if __name__ == "__main__":
    main()