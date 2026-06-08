import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.special import erf

np.random.seed(42)

from sklearn.gaussian_process.kernels import RBF

import warnings
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ===================================================================
# 1. Objective and constraint definitions
# ===================================================================

def f(x1, x2):
    """Objective to minimize."""
    return (
        2
        + 0.01 * (x2 - x1**2)**2
        + (1 - x1)**2
        + 2 * (2 - x2)**2
        + 7 * np.sin(0.5 * x1) * np.sin(0.7 * x1 * x2)
    )


def c(x1, x2):
    """Constraint: c(x) <= 0  => feasible region."""
    return (x1 - 2.5)**2 + (x2 - 2.5)**2 - 2.25


def obj(X):
    """X: (N,2); returns (N,)"""
    x1, x2 = X[:, 0], X[:, 1]
    return f(x1, x2)


def cons(X):
    """X: (N,2); returns (N,)"""
    x1, x2 = X[:, 0], X[:, 1]
    return c(x1, x2)


# Helper pdf/cdf (standard normal)
def norm_pdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


def norm_cdf(x):
    return 0.5 * (1 + erf(x / np.sqrt(2)))

# ===================================================================
# 2. cEI acquisition: EI × PoF (using sklearn GPs)
# ===================================================================

def expected_improvement(X, gp, y_best):
    """Compute EI at X given GP model and y_best (min value among feasible points)."""
    mu, std = gp.predict(X, return_std=True)
    mu = mu.flatten()
    std = std.flatten()

    with np.errstate(divide="ignore", invalid="ignore"):
        Z = (y_best - mu) / std
        ei = (y_best - mu) * norm_cdf(Z) + std * norm_pdf(Z)
        ei[std <= 1e-10] = 0.0
    return ei


def prob_feasibility(X, gp_c):
    """Compute probability that c(x) <= 0 (feasibility)."""
    mu_c, std_c = gp_c.predict(X, return_std=True)
    mu_c = mu_c.flatten()
    std_c = std_c.flatten()

    with np.errstate(divide="ignore", invalid="ignore"):
        Z = -mu_c / std_c  # P(c(x) <= 0) ≈ P(normal <= 0)
        pof = norm_cdf(Z)
    return pof


def cEI(X, gp_obj, gp_con, y_best):
    """Constrained Expected Improvement = EI × PoF."""
    ei = expected_improvement(X, gp_obj, y_best)
    pof = prob_feasibility(X, gp_con)
    return ei * pof


# ===================================================================
# 3. Setup domain and grid
# ===================================================================

lb = np.array([0.0, 0.0])
ub = np.array([5.0, 5.0])

def create_grid(n=100):
    x1 = np.linspace(lb[0], ub[0], n)
    x2 = np.linspace(lb[1], ub[1], n)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.stack([X1, X2], -1).reshape(-1, 2)
    return X, X1, X2


Xgrid, X1grid, X2grid = create_grid(100)


def plot_gp_status(gp_obj, gp_con, X_all, Yobj_all, Ycon_all, iteration, y_best_for_cei):
    mu_f, std_f = gp_obj.predict(Xgrid, return_std=True)
    mu_c, std_c = gp_con.predict(Xgrid, return_std=True)

    mu_f = mu_f.reshape(X1grid.shape)
    std_f = std_f.reshape(X1grid.shape)
    mu_c = mu_c.reshape(X1grid.shape)
    std_c = std_c.reshape(X1grid.shape)

    pof = prob_feasibility(Xgrid, gp_con).reshape(X1grid.shape)
    cei_map = cEI(Xgrid, gp_obj, gp_con, y_best_for_cei).reshape(X1grid.shape)

    C_true = cons(Xgrid).reshape(X1grid.shape)

    fig, axs = plt.subplots(2, 3, figsize=(15, 9))

    # GP objective mean
    cf = axs[0, 0].contourf(X1grid, X2grid, mu_f, levels=60, cmap="viridis")
    axs[0, 0].contour(X1grid, X2grid, C_true, levels=[0], colors="white", linewidths=2)
    axs[0, 0].plot(X_all[:, 0], X_all[:, 1], "ro", markersize=4)
    axs[0, 0].set_title(f"GP mean of objective, iter = {iteration}")
    fig.colorbar(cf, ax=axs[0, 0], label=r"$\mu_f(x)$")

    # GP objective uncertainty
    cf = axs[0, 1].contourf(X1grid, X2grid, std_f, levels=60, cmap="magma")
    axs[0, 1].plot(X_all[:, 0], X_all[:, 1], "ro", markersize=4)
    axs[0, 1].set_title("GP std of objective")
    fig.colorbar(cf, ax=axs[0, 1], label=r"$\sigma_f(x)$")

    # EI
    ei_map = expected_improvement(Xgrid, gp_obj, y_best_for_cei).reshape(X1grid.shape)
    cf = axs[0, 2].contourf(X1grid, X2grid, ei_map, levels=60, cmap="plasma")
    axs[0, 2].plot(X_all[:, 0], X_all[:, 1], "ro", markersize=4)
    axs[0, 2].set_title("Expected Improvement")
    fig.colorbar(cf, ax=axs[0, 2], label="EI")

    # GP constraint mean
    cf = axs[1, 0].contourf(X1grid, X2grid, mu_c, levels=60, cmap="coolwarm")
    axs[1, 0].contour(X1grid, X2grid, mu_c, levels=[0], colors="black", linewidths=2)
    axs[1, 0].contour(X1grid, X2grid, C_true, levels=[0], colors="white", linestyles="--", linewidths=2)
    axs[1, 0].plot(X_all[:, 0], X_all[:, 1], "ro", markersize=4)
    axs[1, 0].set_title("GP mean of constraint")
    fig.colorbar(cf, ax=axs[1, 0], label=r"$\mu_c(x)$")

    # GP constraint uncertainty
    cf = axs[1, 1].contourf(X1grid, X2grid, std_c, levels=60, cmap="magma")
    axs[1, 1].plot(X_all[:, 0], X_all[:, 1], "ro", markersize=4)
    axs[1, 1].set_title("GP std of constraint")
    fig.colorbar(cf, ax=axs[1, 1], label=r"$\sigma_c(x)$")

    # Probability of feasibility / cEI
    cf = axs[1, 2].contourf(X1grid, X2grid, pof, levels=60, cmap="viridis", vmin=0, vmax=1)
    axs[1, 2].contour(X1grid, X2grid, C_true, levels=[0], colors="white", linewidths=2)
    axs[1, 2].plot(X_all[:, 0], X_all[:, 1], "ro", markersize=4)
    axs[1, 2].set_title("Probability of feasibility")
    fig.colorbar(cf, ax=axs[1, 2], label=r"$P(c(x)\leq0)$")

    plt.tight_layout()
    plt.show()

# ===================================================================
# 4. Initial design and GP models
# ===================================================================

ninit = 4
Xinit = lb + (ub - lb) * np.random.rand(ninit, 2)
Yobj = obj(Xinit).reshape(-1, 1)
Ycon = cons(Xinit).reshape(-1, 1)

print(f"Initial feasible rate: {np.mean(Ycon <= 0)}")


# Loosen bounds on length scale and constant separately
kernel_obj = 0.1 * RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-3, 1e3))
kernel_con = 0.1 * RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-3, 1e3))


gp_obj = GaussianProcessRegressor(
    kernel=kernel_obj,
    normalize_y=True,
    n_restarts_optimizer=5,
    alpha=1e-6,
)

gp_con = GaussianProcessRegressor(
    kernel=kernel_con,
    normalize_y=True,
    n_restarts_optimizer=5,
    alpha=1e-6,
)

gp_obj.fit(Xinit, Yobj.ravel())
gp_con.fit(Xinit, Ycon.ravel())


# ===================================================================
# 5. Helper to update best feasible objective
# ===================================================================

def update_ybest(X, Yobj, Ycon):
    feasible = Ycon.ravel() <= 0
    if feasible.any():
        return Yobj[feasible].min()
    return np.inf


# ===================================================================
# 6. Constrained Bayesian Optimization loop with history tracking
# ===================================================================

X_all = Xinit.copy()
Yobj_all = Yobj.copy()
Ycon_all = Ycon.copy()

niter = 20

best_hist = []
xbest_hist = []
query_iter_id = []   # iteration number for each BO query after initial design

for it in range(niter):

    y_best = update_ybest(X_all, Yobj_all, Ycon_all)

    # If no feasible point exists yet, use a relaxed target
    if y_best == np.inf:
        y_best_for_cei = np.min(Yobj_all)
    else:
        y_best_for_cei = y_best

    # Compute cEI
    cei = cEI(Xgrid, gp_obj, gp_con, y_best_for_cei)

    ind_next = np.argmax(cei)
    x_next = Xgrid[ind_next:ind_next + 1]

    y_obj_next = obj(x_next).reshape(1, 1)
    y_con_next = cons(x_next).reshape(1, 1)

    # Add new sample
    X_all = np.vstack([X_all, x_next])
    Yobj_all = np.vstack([Yobj_all, y_obj_next])
    Ycon_all = np.vstack([Ycon_all, y_con_next])

    query_iter_id.append(it + 1)

    # Retrain GP models
    gp_obj.fit(X_all, Yobj_all.ravel())
    gp_con.fit(X_all, Ycon_all.ravel())

    plot_iters = [0, 3, 10, 20]

    if (it + 1) in plot_iters:
        plot_gp_status(
            gp_obj,
            gp_con,
            X_all,
            Yobj_all,
            Ycon_all,
            iteration=it + 1,
            y_best_for_cei=y_best_for_cei
        )

    # Update best feasible point after adding new sample
    feasible = Ycon_all.ravel() <= 0

    if feasible.any():
        feasible_idx = np.where(feasible)[0]
        best_idx_local = np.argmin(Yobj_all[feasible].ravel())
        best_idx = feasible_idx[best_idx_local]

        best_hist.append(Yobj_all[best_idx, 0])
        xbest_hist.append(X_all[best_idx])
    else:
        best_hist.append(np.nan)
        xbest_hist.append([np.nan, np.nan])

    print(
        f"Iter {it+1:02d}: "
        f"x_next = ({x_next[0,0]:.3f}, {x_next[0,1]:.3f}), "
        f"f = {y_obj_next[0,0]:.4f}, "
        f"c = {y_con_next[0,0]:.4f}, "
        f"best feasible f = {best_hist[-1]:.4f}"
    )

best_hist = np.array(best_hist)
xbest_hist = np.array(xbest_hist)

# Final best feasible point
feasible = Ycon_all.ravel() <= 0
feasible_idx = np.where(feasible)[0]
best_idx_local = np.argmin(Yobj_all[feasible].ravel())
best_idx = feasible_idx[best_idx_local]

x_best_final = X_all[best_idx]
y_best_final = Yobj_all[best_idx, 0]
c_best_final = Ycon_all[best_idx, 0]

print("\nFinal best feasible design:")
print(f"x_best = ({x_best_final[0]:.4f}, {x_best_final[1]:.4f})")
print(f"f_best = {y_best_final:.4f}")
print(f"c_best = {c_best_final:.4f}")

# ===================================================================
# 7. Improved visualization
# ===================================================================

F = obj(Xgrid).reshape(X1grid.shape)
C = cons(Xgrid).reshape(X1grid.shape)

X_bo = X_all[ninit:]       # BO query points only
X_initial = X_all[:ninit]  # all initial points

cei_final = cEI(Xgrid, gp_obj, gp_con, y_best_final).reshape(X1grid.shape)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# ---------------------------------------------------------------
# Plot 1: Objective landscape + query trajectory
# ---------------------------------------------------------------
# cf0 = axs[0, 0].contourf(X1grid, X2grid, F, levels=60, cmap="viridis")
vmin = np.percentile(F, 5)
vmax = np.percentile(F, 95)

cf0 = axs[0, 0].contourf(
    X1grid, X2grid, F,
    levels=60,
    cmap="viridis",
    vmin=vmin,
    vmax=vmax
)
axs[0, 0].contour(X1grid, X2grid, C, levels=[0.0], colors="white", linewidths=2)

axs[0, 0].plot(
    X_initial[:, 0], X_initial[:, 1],
    "ks", markersize=6, label="Initial samples"
)

axs[0, 0].plot(
    X_bo[:, 0], X_bo[:, 1],
    "ro-", markersize=5, linewidth=1.2, label="BO query path"
)

for i, x in enumerate(X_bo):
    axs[0, 0].text(
        x[0] + 0.04, x[1] + 0.04,
        str(i + 1),
        fontsize=8,
        color="white"
    )

axs[0, 0].plot(
    x_best_final[0], x_best_final[1],
    marker="*", color="yellow", markersize=18,
    markeredgecolor="black",
    label="Final best feasible"
)

axs[0, 0].set_title("Objective landscape + BO trajectory")
axs[0, 0].set_xlabel("$x_1$")
axs[0, 0].set_ylabel("$x_2$")
axs[0, 0].legend()
fig.colorbar(cf0, ax=axs[0, 0], label="Objective f(x)")


# ---------------------------------------------------------------
# Plot 2: Constraint boundary and feasible region
# ---------------------------------------------------------------
cf1 = axs[0, 1].contourf(X1grid, X2grid, C, levels=60, cmap="coolwarm", alpha=0.75)
axs[0, 1].contour(X1grid, X2grid, C, levels=[0.0], colors="k", linewidths=2)

axs[0, 1].plot(X_initial[:, 0], X_initial[:, 1], "ks", markersize=6, label="Initial samples")
axs[0, 1].plot(X_bo[:, 0], X_bo[:, 1], "ro-", markersize=5, linewidth=1.2, label="BO query path")
axs[0, 1].plot(
    x_best_final[0], x_best_final[1],
    marker="*", color="yellow", markersize=18,
    markeredgecolor="black",
    label="Final best feasible"
)

axs[0, 1].set_title("Constraint boundary: feasible region is c(x) ≤ 0")
axs[0, 1].set_xlabel("$x_1$")
axs[0, 1].set_ylabel("$x_2$")
axs[0, 1].legend()
fig.colorbar(cf1, ax=axs[0, 1], label="Constraint c(x)")


# ---------------------------------------------------------------
# Plot 3: Best feasible objective value versus iteration
# ---------------------------------------------------------------
axs[1, 0].plot(
    np.arange(1, niter + 1),
    best_hist,
    "bo-",
    linewidth=2,
    markersize=5
)

axs[1, 0].set_title("Improvement of best feasible objective")
axs[1, 0].set_xlabel("BO iteration")
axs[1, 0].set_ylabel("Best feasible f(x)")
axs[1, 0].grid(True)

best_text = (
    f"Final best:\n"
    f"x = ({x_best_final[0]:.3f}, {x_best_final[1]:.3f})\n"
    f"f = {y_best_final:.4f}\n"
    f"c = {c_best_final:.4f}"
)

axs[1, 0].text(
    0.60, 0.75,
    best_text,
    transform=axs[1, 0].transAxes,
    bbox=dict(facecolor="white", edgecolor="black", alpha=0.85)
)


# ---------------------------------------------------------------
# Plot 4: Final cEI acquisition map
# ---------------------------------------------------------------
cf2 = axs[1, 1].contourf(X1grid, X2grid, cei_final, levels=60, cmap="plasma")
axs[1, 1].contour(X1grid, X2grid, C, levels=[0.0], colors="white", linewidths=2)

axs[1, 1].plot(X_initial[:, 0], X_initial[:, 1], "ks", markersize=6, label="Initial samples")
axs[1, 1].plot(X_bo[:, 0], X_bo[:, 1], "ro", markersize=5, label="BO queries")

axs[1, 1].plot(
    x_best_final[0], x_best_final[1],
    marker="*", color="yellow", markersize=18,
    markeredgecolor="black",
    label="Final best feasible"
)

axs[1, 1].set_title("Final cEI acquisition")
axs[1, 1].set_xlabel("$x_1$")
axs[1, 1].set_ylabel("$x_2$")
axs[1, 1].legend()
fig.colorbar(cf2, ax=axs[1, 1], label="cEI value")

plt.tight_layout()
plt.show()