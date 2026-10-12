# Interpolation, Extrapolation and Climate Change

**Course:** UE25MA242A — Mathematical Foundation for AI and Data Science  
**Project type:** Mini-project  
**Main program:** `climate_maths_project_final.py`

## 1. Project overview

This project uses historical global annual temperature anomalies to demonstrate how linear algebra can be applied to interpolation, curve fitting, and extrapolation.

The program reads annual temperature-anomaly observations, constructs polynomial models, solves the associated matrix equations using several linear algebra methods, evaluates interpolation and regression errors, and extends a fitted trend to selected future years.

The main forecast years are **2030, 2050, 2075, and 2100**.

> **Important interpretation:** The forecasts are mathematical extrapolations of a polynomial fitted to historical observations. They are not results from a physical climate model and should not be interpreted as official climate projections.

## 2. What question does the project address?

The project explores three related tasks:

- **Interpolation:** Estimate values between observations that are available.
- **Extrapolation:** Extend a fitted relationship beyond the observed time period.
- **Trend analysis:** Fit a mathematical curve to historical temperature anomalies and examine what it implies if that curve continues.

These tasks provide a real-world setting for concepts from linear systems, matrix decomposition, vector spaces, orthogonality, least squares, eigenvalues, and SVD.

## 3. Dataset

The code attempts to download the annual global mean temperature-anomaly series from NASA's GISTEMP v4 data file:

`https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv`

It reads the `J-D` column, representing the annual January-to-December value.

A temperature **anomaly** is a difference from a reference average. In this project the reference period is **1951–1980**. For example, an anomaly of `+1.00 °C` means the annual global value is 1.00 °C above that reference average. It does **not** mean 1.00 °C of additional warming compared with the present day.

If the download fails, the script uses its saved annual values for **1880–2024**. The output identifies which source was used. Because NASA's file can be updated, a successful live download may produce different values and forecasts from the saved copy.

The program requires the interpolation anchor years 1880, 1890, ..., 1950 to be present. It stops with an error if they are missing.

## 4. Requirements

- **Python 3** (Python 3.9 or newer is recommended)
- **NumPy** — required for arrays, matrix operations, and numerical calculations
- **Matplotlib** — optional, but needed to generate the three PNG plots
- **Internet connection** — optional; used to try to download the latest data

Install the packages from Command Prompt or a terminal:

```bash
py -m pip install numpy matplotlib
```

If the `py` command is unavailable, try:

```bash
python -m pip install numpy matplotlib
```

If you use IDLE, make sure the packages are installed for the same Python interpreter that launches IDLE.

## 5. How to run the program

### Option A: Python IDLE on Windows

1. Save `climate_maths_project_final.py` in a folder you can find easily.
2. Open the file in the IDLE editor.
3. Save it with **Ctrl+S**.
4. Choose **Run → Run Module**, or press **F5**.
5. Read the output in the IDLE Shell.
6. When the program finishes, open the `project_outputs` folder to view the plots.

### Option B: Command Prompt or terminal

Open a terminal in the folder containing the Python file and run:

```bash
python climate_maths_project_final.py
```

On some Windows installations, use:

```bash
py climate_maths_project_final.py
```

The program prints each stage to the console. It creates a `project_outputs` directory relative to the program's current working directory. You do not need to create this folder manually.

## 6. How the mathematical model works

The project uses a polynomial in scaled time:

\[
t = \frac{\text{year}-1950}{50}.
\]

Scaling keeps the time values in a more manageable numerical range than using raw years in powers such as `year²` or `year⁷`.

A polynomial of degree \(d\) is written as

\[
\hat y = c_0+c_1t+c_2t^2+\cdots+c_dt^d.
\]

The coefficients \(c_0,\ldots,c_d\) are unknown until they are calculated from the observed data.

A **Vandermonde matrix** builds the powers of each scaled year into columns. For a quadratic model, for example, the columns correspond to \(1\), \(t\), and \(t^2\). Multiplying the matrix by a coefficient vector gives the model's fitted values.

The interpolation and regression parts use related matrices but solve different tasks:

- Interpolation uses eight observations at ten-year intervals from 1880 to 1950 to fit one degree-7 polynomial exactly at those anchor points.
- Regression fits candidate polynomials of degrees 1 through 5 to a larger set of annual observations. The degree is selected using historical validation errors.

## 7. What each program stage does

### Step 1 — Real-world data

Loads the annual anomalies, checks that the years are ordered and unique, and reports the data source, years covered, and anomaly range.

**Why it is needed:** All later calculations depend on the observed values.

### Step 2 — Matrix representation

Selects the eight anchor years 1880, 1890, ..., 1950, scales the year values, and constructs an 8-by-8 Vandermonde matrix \(V\). The interpolation system is

\[
Vc=y,
\]

where \(c\) contains the eight polynomial coefficients and \(y\) contains the eight anchor temperatures.

The program also prints the matrix's condition number. A high condition number indicates that the solution may be sensitive to small changes in the input data.

### Step 3 — Matrix simplification

Solves the interpolation system using:

- **RREF / Gauss–Jordan row reduction**
- **Gaussian elimination with back substitution**
- **LU decomposition with partial pivoting**

NumPy's linear solver is used as a reference for numerical comparison. The program also checks whether the calculated LU factors satisfy \(PV\approxLU\).

**Why it is needed:** This step demonstrates different ways to solve the same system of linear equations.

### Interpolation experiment

The program evaluates the degree-7 interpolant at the annual years between its decade-spaced anchors. It then selects up to 15 repeatable, randomly chosen missing years and compares two estimates on those same target years:

1. The global degree-7 polynomial fitted to the eight decade-spaced anchors.
2. A local cubic polynomial fitted using the four nearest available annual observations, excluding the withheld target years.

The error measure is **RMSE** (root mean square error):

\[
\operatorname{RMSE}=\sqrt{\frac{1}{m}\sum_{i=1}^{m}(\hat y_i-y_i)^2}.
\]

Lower RMSE means smaller errors on the particular observations tested. The comparison is fairer because both estimates are scored on the same withheld target years; however, the two methods intentionally use different sets of available information.

### Model selection — walk-forward validation

The program compares polynomial degrees 1, 2, 3, 4, and 5. For each degree, it fits using observations up to historical cutoffs (1980, 1990, and 2000 when those windows are available) and predicts the following ten years. It averages the validation RMSE values and selects the degree with the lowest average.

It then evaluates the selected degree separately on the latest ten years in the dataset. That final holdout period is not used to choose the degree.

**Why it is needed:** A model that closely fits its training observations may still predict later years poorly. Chronological validation tests how well a model extends into years it has not seen.

The final holdout result is a historical evaluation, not proof that the same model will accurately predict far into the future.

### Step 4 — Structure of the space

The regression design matrix \(A\) has one row per annual observation and one column per polynomial feature. A quadratic model has three columns: \(1\), \(t\), and \(t^2\).

The program calculates the rank and dimensions of all four fundamental subspaces:

- **Column space** \(\operatorname{Col}(A)\): all vectors that can be produced by multiplying \(A\) by a coefficient vector. Here it contains the fitted polynomial trend vectors.
- **Row space** \(\operatorname{Row}(A)\): the space spanned by the rows of \(A\).
- **Null space** \(\operatorname{Null}(A)\): coefficient vectors mapped to the zero vector by \(A\).
- **Left null space** \(\operatorname{Null}(A^T)\): vectors perpendicular to every column of \(A\); least-squares residuals lie in this space.

The rank–nullity relationship is

\[
\operatorname{rank}(A)+\operatorname{nullity}(A)=\text{number of columns of }A.
\]

For a full-rank quadratic design matrix, the rank is 3 and the nullity is 0.

### Step 5 — Linear independence, basis selection, and scaling

The program constructs an example with feature columns \([1,t,t^2,2t+1]\). These four columns are dependent because

\[
2t+1=1+2t.
\]

RREF identifies the redundant column and provides a dependency relation.

The program also compares the condition numbers of polynomial feature matrices built from raw years and scaled years. Raw years and their squares can lead to columns with very different magnitudes. Scaling generally improves the numerical conditioning of this polynomial representation.

### Step 6 — Gram–Schmidt and QR decomposition

The code orthogonalizes the columns of \(A\), producing

\[
A=QR,
\]

where the columns of \(Q\) are orthonormal and \(R\) is upper triangular.

It checks two properties:

\[
Q^TQ\approx I,\qquad A\approx QR.
\]

These checks show whether the computed orthonormal basis and factorization are numerically consistent.

### Step 7 — Orthogonal projection

The projection matrix onto the column space of \(A\) is

\[
P=QQ^T.
\]

The fitted vector and residual vector are

\[
\hat y=Py,\qquad r=y-\hat y.
\]

The program checks that the projection matrix is approximately symmetric and idempotent, and that \(A^Tr\approx0\). The last property means the residual is perpendicular to each column of the design matrix.

### Step 8 — Least-squares approximation

A polynomial trend will generally not pass through every noisy annual observation. The least-squares method chooses coefficients to minimise the sum of squared residuals:

\[
\min_c \|Ac-y\|^2.
\]

The program calculates the coefficients three ways:

- **Normal equations:** \((A^TA)c=A^Ty\)
- **QR:** \(Rc=Q^Ty\)
- **NumPy reference:** `numpy.linalg.lstsq`

It compares the results and reports in-sample RMSE, \(R^2\), and a residual standard-deviation estimate. The QR solution is the main fitted coefficient vector used for the forecasts. QR is also preferable to explicitly solving the normal equations as a general numerical practice, because forming \(A^TA\) can worsen conditioning.

### Step 9 — Eigenvalues, positive definiteness, and SVD

The code forms

\[
G=A^TA.
\]

This is a **Gram matrix of the polynomial features**, not a covariance matrix of multiple climate variables. The distinction matters: the eigenvalues in this project describe the numerical geometry of the regression features, not principal components of climate variability.

The program:

- Calculates eigenvalues and eigenvectors of the symmetric matrix \(G\).
- Uses power iteration as a second way to estimate the largest eigenvalue.
- Estimates the condition number of \(A\) from the extreme eigenvalues of \(A^TA\).
- Checks positive definiteness using eigenvalues and Cholesky factorisation.
- Calculates the SVD \(A=U\Sigma V^T\).
- Uses SVD to calculate another least-squares solution and compares it with the QR result.

When \(A\) has full column rank, \(A^TA\) is theoretically positive definite. The singular values of \(A\) are the square roots of the eigenvalues of \(A^TA\), up to numerical rounding.

### Step 10 — Diagonalization and matrix inversion

The code reconstructs

\[
G=E\Lambda E^T,
\]

where the columns of \(E\) are orthonormal eigenvectors and \(\Lambda\) is diagonal. It then uses this decomposition to calculate an inverse-based least-squares solution.

It also calculates \(G^{-1}\) by applying Gauss–Jordan elimination to the augmented matrix \([G\mid I]\). The program checks the inverse by measuring how close \(GG^{-1}\) is to the identity matrix and compares the inverse with the one from diagonalization.

These inverse calculations are included to demonstrate the course concepts. For routine numerical least squares, the QR or SVD route is generally preferable to explicitly forming and inverting \(A^TA\).

### Final stage — Extrapolation and uncertainty

The selected polynomial is evaluated at 2030, 2050, 2075, and 2100. The output provides:

- The fitted anomaly for each target year.
- An approximate 95% interval for the fitted mean trend.
- An approximate 95% prediction interval for an individual annual value under a classical independent-error regression model.
- A comparison with degree-1 and degree-3 forecasts.
- Straight-line historical trend slopes over the full period and since 1980.
- The lag-1 correlation of the fitted residuals.

These intervals rely on simplified regression assumptions. Annual temperature residuals can be correlated, and polynomial model choice, structural changes, future emissions pathways, and climate physics are not fully represented. A positive residual correlation is one warning that the independent-error assumption may not hold.

The degree comparison is important because a narrow regression interval for one chosen polynomial does not account for all uncertainty about which model or future scenario is appropriate.

## 8. Understanding the three plots

The plots are saved in the `project_outputs` directory.

### `interpolation.png`

Shows the observed annual anomalies in the early period, the degree-7 global interpolant through the decade-spaced anchor points, and the two methods' estimates for the common withheld years. Use it to discuss interpolation and why different methods can behave differently between observations.

### `forecast.png`

Shows the historical annual anomalies, the selected least-squares polynomial extended through 2100, comparisons with other polynomial degrees, and an approximate regression band over the extrapolation period. The vertical line marks the last observed year. The shaded band is incomplete uncertainty, not a comprehensive climate projection range.

### `residuals.png`

Shows the residual for each historical year: observed anomaly minus fitted anomaly. Residuals near zero indicate years where the curve fits closely; larger positive or negative values indicate poorer fits. Patterns or runs of positive/negative residuals may indicate that errors are not independent over time.

## 9. Syllabus coverage

The implementation demonstrates these listed course areas:

- Systems of linear equations, Gaussian elimination, RREF/Gauss–Jordan, LU decomposition, and matrix rank.
- Vector spaces and the four fundamental subspaces, linear independence, basis, dimension, and a matrix transformation.
- Dot products, norms, orthogonality, Gram–Schmidt, orthogonal projections, and least squares.
- Eigenvalues, eigenvectors, orthogonal diagonalization, and positive-definiteness checks.
- Singular Value Decomposition (SVD).

The following syllabus topics are **not explicitly implemented** in this version:

- A covariance matrix calculated from multiple climate variables.
- Principal Component Analysis (PCA) and PCA-based feature extraction or dimensionality reduction.

The eigenvalues of \(A^TA\) should not be described as PCA results. This project uses one global temperature time series and polynomial time features; a meaningful climate PCA would require a multivariate dataset, such as temperatures from multiple regions.

RMSE, \(R^2\), walk-forward validation, prediction intervals, and residual correlation are additional statistical evaluation tools that support the forecasting task. They are not presented as replacements for the linear algebra concepts in the syllabus.

## 10. Files created

After a successful run with Matplotlib installed, the expected layout is:

```text
project-folder/
├── climate_maths_project_final.py
├── README_climate_maths_project.md
└── project_outputs/
    ├── interpolation.png
    ├── forecast.png
    └── residuals.png
```

The script creates `project_outputs` automatically in the current working directory. If the directory is not beside the script, check the directory from which IDLE or the terminal ran the program.

## 11. Common issues

**`ModuleNotFoundError: No module named 'numpy'`**  
Install NumPy for the same Python interpreter used to run the script:

```bash
py -m pip install numpy
```

**Matplotlib is missing**  
The numerical calculations can still run, but the plots will not be generated. Install it with:

```bash
py -m pip install matplotlib
```

**Live data download fails**  
The script switches to the saved 1880–2024 data. This is an expected fallback, not a fatal error.

**The output values differ from an example**  
Check whether the program used the live dataset or the saved copy. A new data endpoint response can change fitted coefficients, validation errors, and forecasts.

**A required interpolation anchor year is missing**  
The program stops because the degree-7 interpolation experiment requires the eight anchor years from 1880 to 1950 at ten-year intervals.

**The output contains many small scientific-notation differences**  
Values such as `1.0e-14` are floating-point numerical errors close to zero. Tiny discrepancies between independently calculated solutions are expected and generally indicate agreement within numerical precision.

## 12. Suggested viva explanations

Use the following **Concept → Purpose → Outcome** structure when explaining the project:

- **Vandermonde matrix:** Represents the polynomial equations in matrix form → makes coefficient solving systematic → produces the polynomial coefficients.
- **Gaussian elimination / LU / RREF:** Solves the interpolation system by different techniques → cross-checks the calculation → the coefficients agree to numerical precision.
- **Rank and subspaces:** Describes the design matrix's independent directions → checks how many independent polynomial features are present → for a full-rank quadratic model, the rank is 3 and nullity is 0.
- **Gram–Schmidt / QR:** Creates an orthonormal basis for the design matrix columns → supports projection and least squares → produces factors \(Q\) and \(R\) with \(A\approx QR\).
- **Projection:** Finds the closest vector in the model space → separates fitted values from residuals → the residual is orthogonal to the design matrix columns.
- **Least squares:** Fits a polynomial when the observed data cannot be matched exactly → minimises the sum of squared residuals → produces fitted coefficients and a trend curve.
- **Eigenvalues / positive definiteness:** Examines the geometry and conditioning of \(A^TA\) → checks matrix properties and numerical sensitivity → helps explain why feature scaling and stable solution methods matter.
- **SVD:** Factorises the design matrix → provides an alternative least-squares route and reveals singular values → gives coefficients that should agree with the QR solution within numerical precision.
- **Interpolation:** Estimates missing values between observed years → compares a global polynomial and local cubic method → produces estimates and RMSE values on withheld years.
- **Extrapolation:** Evaluates the fitted trend after the observed period → explores what happens if the fitted polynomial continues → produces mathematical anomaly projections with explicit limitations.

## 13. Limitations and responsible interpretation

1. A polynomial can fit historical data well and still extrapolate poorly far beyond it.
2. The quadratic degree is selected by historical walk-forward validation; that does not establish that a quadratic is the physically correct model for future climate.
3. The regression intervals use simplified assumptions. Residual autocorrelation indicates that the independent-error assumption may be questionable.
4. The model does not explicitly simulate future greenhouse-gas emissions, climate feedbacks, volcanic effects, El Niño–Southern Oscillation, or other physical drivers.
5. The anomaly baseline is 1951–1980. Do not describe the forecast values as warming above the present day or above the pre-industrial period.
6. The eigenvalues come from the polynomial feature Gram matrix \(A^TA\), not a climate covariance matrix. PCA is not performed in this code.

The project's goal is to demonstrate mathematical techniques on a real-world dataset, not to replace scientific climate modelling.

## 14. Data source

NASA GISS Surface Temperature Analysis (GISTEMP v4), global annual mean temperature anomaly table:  
<https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv>

The script includes a saved data copy as a fallback for environments without network access.
