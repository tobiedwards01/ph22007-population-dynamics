"""
PH22007 Scientific Computing in Python - Coursework: Population Dynamics

Part 1: Lotka-Volterra (rabbits and foxes), Euler vs RK4
Part 2: Zombie outbreak model (humans, zombies, dead)
Part 3: Extensions - phase-space vector fields and time-varying parameters

Running this file produces every figure used in the report and saves them
to the ./figures folder as 300 dpi PNGs. Figures are drawn at their final
printed size (6.7 in wide = A4 text width with 2 cm margins), so they should
be included in the report at 100% scale. Key numerical results are printed
and also written to ./figures/results_summary.txt.
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Plot style: one consistent look, sized for print
# ---------------------------------------------------------------------------
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FIG_WIDTH = 6.7     # inches: A4 width minus 2 cm margins
FIG_HEIGHT = 2.3    # inches: default panel height

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.5,
})

# Fixed colour for each population/method so it is recognisable across figures
COLOURS = {
    "rabbits": "#2a7fbf",   # blue
    "foxes":   "#e4572e",   # orange-red
    "humans":  "#2a9d8f",   # teal
    "zombies": "#8e44ad",   # purple
    "dead":    "#6c757d",   # grey
    "euler":   "#e09f3e",   # amber
    "rk4":     "#264653",   # dark slate
}
PHASE_CMAP = "viridis"      # colour map used to show time along orbits

# Common phase-space limits so Euler and RK4 orbits can be compared by eye
LV_XLIM, LV_YLIM = (0, 165), (0, 45)

# Numerical results collected while running, written out at the end
RESULTS = {}


def new_figure(ncols, height=FIG_HEIGHT, width_ratios=None):
    """Row of panels at final printed size, with automatic spacing."""
    fig, axes = plt.subplots(1, ncols, figsize=(FIG_WIDTH, height),
                             width_ratios=width_ratios, layout="constrained")
    return fig, axes


def panel_title(ax, letter, text):
    """Consistent '(a) Title' panel labels, used by every figure."""
    ax.set_title(f"({letter}) {text}", loc="left")


def save_figure(fig, name):
    """Save a figure as a 300 dpi PNG at its final printed size."""
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"))
    print(f"  saved figures/{name}.png")


def sweep_colours(n, cmap="plasma"):
    """n distinct colours from a colour map (avoiding the very pale end)."""
    return plt.get_cmap(cmap)(np.linspace(0.0, 0.85, n))


def sci(x, digits=1):
    """Format a number as LaTeX scientific notation, e.g. 9.5e-3 -> 9.5x10^-3."""
    mantissa, exponent = f"{x:.{digits}e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def local_maxima(y):
    """Indices of interior local maxima of a sampled curve."""
    return np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1


# ---------------------------------------------------------------------------
# Generic ODE tools
# Each solver takes f(t, u, p) returning du/dt as a NumPy array, so the same
# code solves the 2-variable rabbit-fox and 3-variable zombie systems.
# ---------------------------------------------------------------------------
def make_time_grid(t0, tN, h):
    """Evenly spaced times t_i = t0 + i*h from t0 to tN inclusive."""
    N = int(round((tN - t0) / h))
    return t0 + h * np.arange(N + 1)


def euler(f, u0, t, p):
    """
    Explicit (forward) Euler: u_{i+1} = u_i + h f(t_i, u_i). Global error O(h).
    u0 may have shape (n_vars,) or (n_vars, n_runs) to solve many runs at once.
    """
    u0 = np.asarray(u0, dtype=float)
    u = np.empty((len(t),) + u0.shape)
    u[0] = u0
    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        u[i + 1] = u[i] + h * f(t[i], u[i], p)
    return u


def rk4(f, u0, t, p):
    """
    Classical fourth-order Runge-Kutta. Global error O(h^4).
    u0 may have shape (n_vars,) or (n_vars, n_runs): parameters can then be
    arrays of length n_runs, so a whole parameter sweep runs in one loop.
    """
    u0 = np.asarray(u0, dtype=float)
    u = np.empty((len(t),) + u0.shape)
    u[0] = u0
    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        ti, ui = t[i], u[i]
        k1 = f(ti, ui, p)
        k2 = f(ti + h / 2, ui + h / 2 * k1, p)
        k3 = f(ti + h / 2, ui + h / 2 * k2, p)
        k4 = f(ti + h, ui + h * k3, p)
        u[i + 1] = ui + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return u


def value(param, t):
    """Allow any model parameter to be a constant or a function of time."""
    return param(t) if callable(param) else param


def crossing_times(t, g, direction=+1):
    """
    Times where g(t) crosses zero, found by linear interpolation between
    grid points. direction=+1 for upward (- to +), -1 for downward crossings.
    """
    s = direction * g
    idx = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    return t[idx] - g[idx] * (t[idx + 1] - t[idx]) / (g[idx + 1] - g[idx])


# ---------------------------------------------------------------------------
# PART 1: Lotka-Volterra model
# ---------------------------------------------------------------------------
LV_PARAMS = {"alpha": 1.0, "beta": 0.1, "gamma": 0.5, "delta": 0.02}
LV_U0 = np.array([100.0, 20.0])          # 100 rabbits, 20 foxes
LV_T0, LV_TN, LV_H = 0.0, 100.0, 0.01    # days


def lotka_volterra(t, u, p):
    """Right-hand side of Eq. (1). u = [x (rabbits), y (foxes)]."""
    x, y = u
    dxdt = value(p["alpha"], t) * x - p["beta"] * x * y
    dydt = p["delta"] * x * y - p["gamma"] * y
    return np.array([dxdt, dydt])


def lv_invariant(x, y, p):
    """
    Conserved quantity of Eq. (1). Eliminating time gives
    dy/dx = y(delta x - gamma) / [x(alpha - beta y)], which separates to
    V = delta x - gamma ln x + beta y - alpha ln y = constant.
    Exact solutions keep V fixed, so its drift measures numerical error.
    """
    return (p["delta"] * x - p["gamma"] * np.log(x)
            + p["beta"] * y - p["alpha"] * np.log(y))


def lv_fixed_points(p):
    """
    Both derivatives vanish when
      x(alpha - beta y) = 0  and  y(delta x - gamma) = 0
    giving (0, 0) (extinction) and (gamma/delta, alpha/beta) (coexistence).
    """
    return [(0.0, 0.0), (p["gamma"] / p["delta"], p["alpha"] / p["beta"])]


def plot_orbit(ax, x, y, t, cbar=True, lw=1.1):
    """Phase-space orbit coloured by time, with the start point marked."""
    # Draw at most ~3000 segments: visually identical, much faster to save
    step = max(1, len(t) // 3000)
    xs, ys, ts = x[::step], y[::step], t[::step]
    points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=PHASE_CMAP, linewidth=lw)
    lc.set_array(ts[:-1])
    ax.add_collection(lc)
    ax.autoscale()
    ax.plot(x[0], y[0], "o", color="black", ms=4, zorder=5, label="start")
    if cbar:
        cb = ax.figure.colorbar(lc, ax=ax, pad=0.01, aspect=25)
        cb.set_label("time (days)")
    return lc


def plot_lv_timeseries(ax, t, u):
    """Rabbit and fox populations against time on one axis."""
    ax.plot(t, u[:, 0], color=COLOURS["rabbits"], label="rabbits $x$")
    ax.plot(t, u[:, 1], color=COLOURS["foxes"], label="foxes $y$")
    ax.set_xlabel("time (days)")
    ax.set_ylabel("population")
    ax.set_xlim(t[0], t[-1])


def solve_lotka_volterra():
    """Solve Eq. (1) with both methods on the same grid (timed)."""
    t = make_time_grid(LV_T0, LV_TN, LV_H)
    start = time.perf_counter()
    u_e = euler(lotka_volterra, LV_U0, t, LV_PARAMS)
    t_euler = time.perf_counter() - start
    start = time.perf_counter()
    u_r = rk4(lotka_volterra, LV_U0, t, LV_PARAMS)
    t_rk4 = time.perf_counter() - start
    RESULTS["Compare: run time ratio RK4/Euler (100 d, h=0.01)"] = \
        f"{t_rk4 / t_euler:.1f} (RK4 uses 4 function evaluations per step, Euler 1)"
    return t, u_e, u_r


def question_1_1(t, u_e, u_r):
    """Q1.1: Euler solution of Lotka-Volterra, with the RK4 orbit for reference."""
    print("\nQuestion 1.1 - Euler method")
    x, y = u_e[:, 0], u_e[:, 1]
    pk_x, pk_y = local_maxima(x), local_maxima(y)

    fig, (ax1, ax2) = new_figure(2, width_ratios=[1.2, 1])
    plot_lv_timeseries(ax1, t, u_e)
    ax1.set_ylim(0, 1.25 * x.max())
    ax1.legend(loc="upper left", ncol=2)
    panel_title(ax1, "a", "Populations (Euler)")

    ax2.plot(u_r[:, 0], u_r[:, 1], color="grey", ls="--", lw=0.9,
             label="exact orbit (RK4)", zorder=4)
    plot_orbit(ax2, x, y, t)
    ax2.set_xlim(*LV_XLIM)
    ax2.set_ylim(*LV_YLIM)
    ax2.set_xlabel("rabbits $x$")
    ax2.set_ylabel("foxes $y$")
    ax2.legend(loc="upper right")
    panel_title(ax2, "b", "Phase space (Euler)")
    save_figure(fig, "q1_1_euler")

    RESULTS["Q1.1 Euler: rabbit peaks first -> last"] = \
        f"{x[pk_x][0]:.1f} -> {x[pk_x][-1]:.1f} ({len(pk_x)} peaks)"
    RESULTS["Q1.1 Euler: fox peaks first -> last"] = \
        f"{y[pk_y][0]:.1f} -> {y[pk_y][-1]:.1f}"
    RESULTS["Q1.1 Euler: max rabbits / max foxes"] = f"{x.max():.1f} / {y.max():.1f}"
    print(f"  rabbit peaks {x[pk_x][0]:.1f} -> {x[pk_x][-1]:.1f}")


def find_period(t, u):
    """
    Period from returns to the starting point: times where x(t) passes back
    through x0 in the same direction as at t = 0 (a Poincare section).
    """
    x0 = u[0, 0]
    direction = np.sign(u[1, 0] - u[0, 0])          # x is falling at t = 0
    returns = crossing_times(t[1:], u[1:, 0] - x0, direction=int(direction))
    periods = np.diff(np.concatenate([[t[0]], returns]))
    return periods, returns


def question_1_2(t, u):
    """Q1.2: RK4 solution, fixed points/nullclines, period."""
    print("\nQuestion 1.2 - RK4 method")
    p = LV_PARAMS

    # Validate our RK4 against SciPy's adaptive solver at tight tolerance
    ref = solve_ivp(lotka_volterra, (LV_T0, LV_TN), LV_U0, args=(p,),
                    method="DOP853", t_eval=t, rtol=1e-12, atol=1e-12)
    max_diff = np.max(np.abs(u - ref.y.T))
    print(f"  max |RK4 - solve_ivp| over 100 days = {max_diff:.2e}")
    RESULTS["Q1.2a max |own RK4 - scipy DOP853| (populations)"] = f"{max_diff:.2e}"

    # ---- (b) fixed points and nullclines ----
    x_star, y_star = lv_fixed_points(p)[1]
    RESULTS["Q1.2b fixed points"] = f"(0, 0) saddle and ({x_star:g}, {y_star:g}) centre"
    RESULTS["Q1.2b rabbit nullclines (dx/dt=0)"] = f"x = 0 and y = alpha/beta = {y_star:g}"
    RESULTS["Q1.2b fox nullclines (dy/dt=0)"] = f"y = 0 and x = gamma/delta = {x_star:g}"
    lin_period = 2 * np.pi / np.sqrt(p["alpha"] * p["gamma"])
    RESULTS["Q1.2b small-oscillation period 2pi/sqrt(alpha*gamma)"] = f"{lin_period:.3f} days"

    # ---- (c) period ----
    periods, _ = find_period(t, u)
    T = periods.mean()
    end_state = (np.interp(T, t, u[:, 0]), np.interp(T, t, u[:, 1]))
    print(f"  period T = {T:.4f} days; linearised period = {lin_period:.3f}")
    RESULTS["Q1.2c period T (RK4)"] = (f"{T:.4f} days (mean of {len(periods)} cycles, "
                                       f"spread {periods.std():.1e} days)")
    RESULTS["Q1.2c state at t=T (should equal start (100, 20))"] = \
        f"({end_state[0]:.3f}, {end_state[1]:.3f})"

    # Phase lag of fox peaks behind rabbit peaks (nonlinear orbit: << T/4)
    tx, ty = t[local_maxima(u[:, 0])], t[local_maxima(u[:, 1])]
    lags = np.array([ty[ty > s][0] - s for s in tx if np.any(ty > s)])
    RESULTS["Q1.2 rabbit peak times (first 3)"] = ", ".join(f"{v:.2f}" for v in tx[:3])
    RESULTS["Q1.2 fox peak times (first 3)"] = ", ".join(f"{v:.2f}" for v in ty[:3])
    RESULTS["Q1.2 fox lag behind rabbit peak"] = \
        f"{lags.mean():.2f} days = {lags.mean() / T:.3f} T (small-oscillation limit: T/4)"

    # ---- Figure ----
    fig, (ax1, ax2) = new_figure(2, width_ratios=[1.2, 1])
    plot_lv_timeseries(ax1, t, u)
    ax1.axvspan(0, T, color="#f2cc8f", alpha=0.4, lw=0, label=f"one period $T$")
    ax1.set_ylim(0, 1.3 * u[:, 0].max())
    ax1.legend(loc="upper left", ncol=3)
    panel_title(ax1, "a", "Populations (RK4)")

    ax2.plot(u[:, 0], u[:, 1], color=COLOURS["rk4"], lw=1.2)
    # Nullclines and fixed points labelled directly on the plot
    ax2.axhline(y_star, color=COLOURS["rabbits"], ls="--", lw=0.9)
    ax2.axvline(x_star, color=COLOURS["foxes"], ls="--", lw=0.9)
    ax2.text(40, y_star + 0.8, r"$\dot x = 0$ ($y=\alpha/\beta$)", ha="left",
             color=COLOURS["rabbits"], fontsize=7)
    ax2.text(x_star + 2, 43.5, r"$\dot y = 0$ ($x=\gamma/\delta$)", va="top",
             color=COLOURS["foxes"], fontsize=7)
    ax2.plot(x_star, y_star, "*", color="gold", mec="black", mew=0.6, ms=9, zorder=6)
    ax2.text(x_star + 3, y_star - 3, f"({x_star:g}, {y_star:g})", fontsize=7)
    ax2.plot(0, 0, "o", mfc="white", mec="black", ms=5, zorder=6, clip_on=False)
    ax2.annotate("(0, 0)", xy=(0.8, 0.5), xytext=(9, 5), fontsize=7,
                 arrowprops=dict(arrowstyle="->", lw=0.6))
    # Start (large ring) and end (small dot) of one period: they coincide
    ax2.plot(*u[0], "o", mfc="none", mec=COLOURS["humans"], mew=1.5, ms=10,
             zorder=7, label="$t = 0$")
    ax2.plot(*end_state, "o", color="black", ms=3.5, zorder=8, label=f"$t = T$")
    ax2.annotate("$t=0$ and $t=T$\ncoincide at (100, 20)", xy=(100, 20),
                 xytext=(118, 29), fontsize=7, ha="left",
                 arrowprops=dict(arrowstyle="->", lw=0.7))
    # Arrow on top of the orbit showing the (anticlockwise) direction
    first = t < T
    i_top = np.argmax(np.where(first, u[:, 1], -np.inf))
    ax2.annotate("", xy=u[i_top + 6], xytext=u[i_top - 6],
                 arrowprops=dict(arrowstyle="-|>", color=COLOURS["rk4"],
                                 lw=1.2, mutation_scale=14))
    ax2.set_xlim(*LV_XLIM)
    ax2.set_ylim(*LV_YLIM)
    ax2.set_xlabel("rabbits $x$")
    ax2.set_ylabel("foxes $y$")
    ax2.legend(loc="lower right", borderaxespad=0.1, labelspacing=0.3)
    panel_title(ax2, "b", "Phase space (RK4)")
    save_figure(fig, "q1_2_rk4")
    return T


def compare_methods(t, u_e, u_r):
    """
    Quantitative Euler vs RK4 comparison:
      (a) drift of the conserved quantity V(x, y) over 100 days;
      (b) global error at a fixed time against computational cost
          (number of right-hand-side evaluations).
    """
    print("\nEuler vs RK4 comparison")
    p = LV_PARAMS
    V0 = lv_invariant(*LV_U0, p)
    drift_e = np.abs(lv_invariant(u_e[:, 0], u_e[:, 1], p) - V0) / abs(V0)
    drift_r = np.abs(lv_invariant(u_r[:, 0], u_r[:, 1], p) - V0) / abs(V0)
    RESULTS["Compare: V0"] = f"{V0:.6f}"
    RESULTS["Compare: relative drift in V after 100 d (Euler, RK4), h=0.01"] = \
        f"{drift_e[-1]:.2e}, {drift_r[-1]:.2e}"

    # Convergence study: absolute error at t_end against a high-accuracy reference
    t_end = 20.0
    ref = solve_ivp(lotka_volterra, (0, t_end), LV_U0, args=(p,),
                    method="DOP853", rtol=1e-13, atol=1e-13).y[:, -1]
    hs = np.array([0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001])
    err = {"euler": [], "rk4": []}
    for h in hs:
        tg = make_time_grid(0, t_end, h)
        err["euler"].append(np.linalg.norm(euler(lotka_volterra, LV_U0, tg, p)[-1] - ref))
        err["rk4"].append(np.linalg.norm(rk4(lotka_volterra, LV_U0, tg, p)[-1] - ref))
    stages = {"euler": 1, "rk4": 4}      # function evaluations per step
    # Orders fitted only in the asymptotic (small-h) region, h <= 0.02 d
    fitted = hs <= 0.02

    fig, (ax1, ax2) = new_figure(2)
    ax1.semilogy(t[1:], drift_e[1:], color=COLOURS["euler"], label="Euler")
    ax1.semilogy(t[1:], drift_r[1:], color=COLOURS["rk4"], label="RK4")
    ax1.set_xlabel("time (days)")
    ax1.set_ylabel(r"$|V(t) - V_0| \,/\, |V_0|$")
    ax1.set_xlim(0, t[-1])
    ax1.set_ylim(1e-12, 5)
    ax1.legend(loc="center right")
    panel_title(ax1, "a", "Drift of conserved $V$ ($h$ = 0.01 d)")

    for method, marker in (("euler", "o"), ("rk4", "s")):
        e = np.array(err[method])
        n_eval = stages[method] * t_end / hs
        order = np.polyfit(np.log(hs[fitted]), np.log(e[fitted]), 1)[0]
        c = COLOURS[method]
        ax2.loglog(n_eval, e, "-", color=c, lw=0.9)
        ax2.loglog(n_eval[fitted], e[fitted], marker, color=c,
                   label=f"{'Euler' if method == 'euler' else 'RK4'} (order {order:.2f})")
        ax2.loglog(n_eval[~fitted], e[~fitted], marker, mfc="white", mec=c)
        RESULTS[f"Compare: fitted convergence order {method}"] = f"{order:.2f}"
        for h, ee in zip(hs, e):
            RESULTS[f"Compare: abs error at t=20 d, {method}, h={h:g}"] = f"{ee:.2e}"

    # Equal-cost comparison: Euler h=0.005 and RK4 h=0.02 both use 4000 evaluations
    e_eu = err["euler"][list(hs).index(0.005)]
    e_rk = err["rk4"][list(hs).index(0.02)]
    RESULTS["Compare: equal cost (4000 evaluations) error Euler / RK4"] = \
        f"{e_eu:.2e} / {e_rk:.2e} (ratio {e_eu / e_rk:.1e})"
    ax2.axvline(4000, color="grey", lw=0.7, ls=":")
    ax2.text(4600, 1e-2, "equal cost:\nRK4 error\n"
             rf"$\sim10^{{{np.log10(e_eu / e_rk):.0f}}}\times$ smaller",
             fontsize=6.5, color="dimgrey", va="top")
    ax2.plot([], [], "o", mfc="white", mec="grey", label="$h>0.02$ d (not fitted)")
    ax2.set_xlabel("function evaluations")
    ax2.set_ylabel(r"$|\mathbf{u}_\mathrm{num}-\mathbf{u}_\mathrm{ref}|$ at 20 d (individuals)")
    ax2.legend(loc="lower left", fontsize=6.5)
    panel_title(ax2, "b", "Absolute error against cost")
    save_figure(fig, "q1_2_comparison")


# ---------------------------------------------------------------------------
# PART 2: Zombie outbreak model
# ---------------------------------------------------------------------------
Z_PARAMS = {"alpha": 0.0, "beta": 1.0e-4, "gamma": 9.5e-3,
            "delta": 1.0e-4, "epsilon": 1.0e-4}
Z_U0 = np.array([1000.0, 5.0, 0.0])       # humans, zombies, dead
Z_T0, Z_TN, Z_H = 0.0, 5.0, 0.01          # days


def zombie(t, u, p):
    """Right-hand side of Eq. (2). u = [H, Z, D]. Parameters may vary in time."""
    H, Z, D = u
    a, b, g = value(p["alpha"], t), value(p["beta"], t), value(p["gamma"], t)
    d, e = value(p["delta"], t), value(p["epsilon"], t)
    dH = a * H - b * H - g * H * Z
    dZ = g * H * Z - d * H * Z + e * D
    dD = b * H + d * H * Z - e * D
    return np.array([dH, dZ, dD])


def outbreak_time(t, u, k=1.0):
    """
    Outbreak condition: first time zombies outnumber humans by factor k,
    i.e. Z >= k H. Returns NaN if it never happens in the time window.
    """
    times = crossing_times(t, u[:, 1] - k * u[:, 0], direction=+1)
    return times[0] if len(times) else np.nan


def logistic_zombies(t, gamma, N, Z0):
    """
    Logistic approximation: ignoring the small rates beta, delta, epsilon,
    H + Z = N and dZ/dt = gamma Z (N - Z), so
    Z(t) = N / (1 + (N/Z0 - 1) exp(-gamma N t)).
    """
    return N / (1 + (N / Z0 - 1) * np.exp(-gamma * N * t))


def logistic_outbreak_time(gamma, k, N, Z0):
    """
    From the logistic solution, Z/H = exp(gamma N t) / (N/Z0 - 1), so
    Z = k H at t = ln[k (N/Z0 - 1)] / (gamma N) = A / gamma.
    """
    return np.log(k * (N / Z0 - 1)) / (N * gamma)


def question_2_1():
    """Q2.1: solve the zombie system with RK4."""
    print("\nQuestion 2.1 - zombie outbreak")
    p = Z_PARAMS
    t = make_time_grid(Z_T0, Z_TN, Z_H)
    u = rk4(zombie, Z_U0, t, p)
    N = Z_U0.sum()

    t_eq = outbreak_time(t, u, k=1.0)
    t_logistic = logistic_outbreak_time(p["gamma"], 1.0, N, Z_U0[1])
    Z_log = logistic_zombies(t, p["gamma"], N, Z_U0[1])
    print(f"  H = Z at t = {t_eq:.4f} days (logistic estimate {t_logistic:.4f})")
    RESULTS["Q2.1 time when H = Z"] = f"{t_eq:.4f} days"
    RESULTS["Q2.1 logistic (analytic) estimate of H = Z time"] = f"{t_logistic:.4f} days"
    RESULTS["Q2.1 population at H = Z"] = f"{np.interp(t_eq, t, u[:, 0]):.1f} each"
    RESULTS["Q2.1 final (H, Z, D) at t = 5 d"] = \
        f"({u[-1, 0]:.3f}, {u[-1, 1]:.1f}, {u[-1, 2]:.2f})"
    RESULTS["Q2.1 final D vs analytic H0*delta/gamma"] = \
        f"{u[-1, 2]:.2f} vs {Z_U0[0] * p['delta'] / p['gamma']:.2f}"
    RESULTS["Q2.1 max deviation of H+Z+D from 1005 (alpha=0 => conserved)"] = \
        f"{np.max(np.abs(u.sum(axis=1) - N)):.1e}"

    fig, (ax1, ax2) = new_figure(2, width_ratios=[1.15, 1])
    for ax, log in ((ax1, False), (ax2, True)):
        plot = ax.semilogy if log else ax.plot
        plot(t, u[:, 0], color=COLOURS["humans"], label="humans $H$")
        plot(t, u[:, 1], color=COLOURS["zombies"], label="zombies $Z$")
        plot(t[1:], u[1:, 2], color=COLOURS["dead"], label="dead $D$")
        plot(t, N - Z_log, color="black", ls="--", lw=0.7)
        plot(t, Z_log, color="black", ls="--", lw=0.7, label="logistic approx.")
        ax.axvline(t_eq, color="black", lw=0.6, ls=":")
        ax.set_xlabel("time (days)")
        ax.set_xlim(t[0], t[-1])
    ax1.plot(t, u.sum(axis=1), color="black", ls=":", lw=0.8, label="total")
    ax1.plot(t_eq, np.interp(t_eq, t, u[:, 0]), "o", color="black", zorder=5)
    ax1.annotate(f"$H = Z$ at\n$t$ = {t_eq:.2f} d", xy=(t_eq, 500),
                 xytext=(t_eq + 0.5, 560), fontsize=7,
                 arrowprops=dict(arrowstyle="->", lw=0.7))
    ax1.set_ylim(0, 1100)
    ax1.set_ylabel("population")
    ax1.legend(loc="center right")
    panel_title(ax1, "a", "Zombie outbreak (linear axis)")
    ax2.set_ylim(1e-3, 3e3)
    ax2.set_ylabel("population")
    panel_title(ax2, "b", "Logarithmic axis")
    save_figure(fig, "q2_1_zombies")


def question_2_2a():
    """Q2.2a: zombie growth for a range of zombification rates."""
    print("\nQuestion 2.2a - varying gamma")
    gammas = np.array([1e-3, 2e-3, 5e-3, 9.5e-3, 2e-2, 5e-2])
    delta, N = Z_PARAMS["delta"], Z_U0.sum()
    t = make_time_grid(0.0, 12.0, 0.002)
    colours = sweep_colours(len(gammas))

    # Solve all gamma values in one batched RK4 run: u has shape (nt, 3, n_gamma)
    u0 = np.repeat(Z_U0[:, None], len(gammas), axis=1)
    u_all = rk4(zombie, u0, t, dict(Z_PARAMS, gamma=gammas))

    fig, (ax1, ax2) = new_figure(2)
    t_out = []
    for j, (g, c) in enumerate(zip(gammas, colours)):
        u = u_all[:, :, j]
        to = outbreak_time(t, u)
        t_out.append(to)
        lab = rf"$\gamma = {sci(g)}$"
        ax1.plot(t, u[:, 1], color=c, label=lab)
        ax1.plot(to, np.interp(to, t, u[:, 1]), "o", color=c, ms=3)
        # Rescaled time: the net conversion rate (gamma - delta) sets the clock
        ax2.plot((g - delta) * t, u[:, 1] / N, color=c, lw=1.6 - 0.15 * j, label=lab)
        RESULTS[f"Q2.2a gamma={g:g}: H=Z time"] = f"{to:.3f} d"
        RESULTS[f"Q2.2a gamma={g:g}: final D vs H0*delta/gamma"] = \
            f"{u[-1, 2]:.2f} vs {Z_U0[0] * delta / g:.2f}"
    t_out = np.array(t_out)
    for name, scale in (("gamma*t", gammas), ("(gamma-delta)*t", gammas - delta)):
        s = scale * t_out
        RESULTS[f"Q2.2a spread of outbreak point in {name}"] = f"{np.ptp(s) / s.mean():.2%}"

    ax1.set_xlim(0, 8)
    ax1.set_ylim(0, 1050)
    ax1.set_xlabel("time (days)")
    ax1.set_ylabel("zombies $Z$")
    panel_title(ax1, "a", r"Zombies for several $\gamma$ (dots: $Z=H$)")
    ax2.set_xlim(0, 0.012)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel(r"rescaled time $(\gamma-\delta)\,t$")
    ax2.set_ylabel("$Z / N$")
    ax2.legend(loc="lower right", fontsize=6.5)
    panel_title(ax2, "b", "Collapse onto one curve")
    save_figure(fig, "q2_2a_gamma_sweep")


def question_2_2bc():
    """
    Q2.2b/c: outbreak time against gamma and fit t = A / gamma.
    Main condition: Z >= H (zombies become the majority). This is the
    midpoint (inflection) of the near-logistic growth, where zombies are
    spreading fastest, so it is sharply defined and insensitive to small
    numerical errors. Z >= 10 H is shown for comparison (stricter takeover).
    """
    print("\nQuestion 2.2b/c - outbreak time vs gamma")
    N, Z0 = Z_U0.sum(), Z_U0[1]
    gammas = np.logspace(-3, -1, 25)
    conditions = {1.0: COLOURS["zombies"], 10.0: "#e76f51"}

    # One batched RK4 run for every gamma. The step h = 1e-3 d is small
    # compared with the fastest growth time 1/(gamma_max N) ~ 0.01 d, and the
    # end time covers the slowest outbreak (gamma_min).
    t = make_time_grid(0.0, 15.0 / (gammas.min() * N), 1e-3)
    u0 = np.repeat(Z_U0[:, None], len(gammas), axis=1)
    u_all = rk4(zombie, u0, t, dict(Z_PARAMS, gamma=gammas))

    def inverse(g, A):
        return A / g

    def shifted(g, A, c):
        return A / (g - c)

    def power(g, A, n):
        return A * g ** n

    fig, (ax1, ax2) = new_figure(2, width_ratios=[1.1, 1])
    residuals = {}
    for k, colour in conditions.items():
        t_out = np.array([outbreak_time(t, u_all[:, :, j], k)
                          for j in range(len(gammas))])

        # Data span two decades, so weight by relative (not absolute) error
        (A,), covA = curve_fit(inverse, gammas, t_out, p0=[5e-3], sigma=t_out)
        (As, c), covS = curve_fit(shifted, gammas, t_out, p0=[5e-3, 1e-4], sigma=t_out)
        (B, n), covB = curve_fit(power, gammas, t_out, p0=[5e-3, -1], sigma=t_out)
        dA = np.sqrt(covA[0, 0])
        dAs, dc = np.sqrt(np.diag(covS))
        dB, dn = np.sqrt(np.diag(covB))
        A_th = logistic_outbreak_time(1.0, k, N, Z0)
        res_inv = 100 * (t_out - inverse(gammas, A)) / t_out
        res_shift = 100 * (t_out - shifted(gammas, As, c)) / t_out
        residuals[k] = (res_inv, res_shift)
        print(f"  Z >= {k:g}H: A = {A:.4e} +/- {dA:.1e} (logistic theory "
              f"{A_th:.4e}); n = {n:.4f}; shifted fit c = {c:.3e}")

        lab = r"Z \geq H" if k == 1 else rf"Z \geq {k:g}H"
        key = "Z>=H" if k == 1 else f"Z>={k:g}H"
        RESULTS[f"Q2.2c [{key}] fit t = A/gamma: A"] = f"{A:.4e} +/- {dA:.1e}"
        RESULTS[f"Q2.2c [{key}] logistic theory A = ln(k(N/Z0-1))/N"] = f"{A_th:.4e}"
        RESULTS[f"Q2.2c [{key}] residuals of A/gamma fit"] = \
            f"{res_inv.min():.2f}% to {res_inv.max():.2f}% (positive at small gamma)"
        RESULTS[f"Q2.2c [{key}] fit t = B*gamma^n: B, n"] = \
            f"{B:.4e} +/- {dB:.1e}, {n:.4f} +/- {dn:.4f}"
        RESULTS[f"Q2.2c [{key}] fit t = A/(gamma - c): A, c"] = \
            f"{As:.4e} +/- {dAs:.1e}, {c:.3e} +/- {dc:.1e} (compare delta = 1e-4)"
        RESULTS[f"Q2.2c [{key}] max |residual| of A/(gamma-c) fit"] = \
            f"{np.abs(res_shift).max():.4f}%"
        for g, to in zip(gammas[::6], t_out[::6]):
            RESULTS[f"Q2.2c [{key}] outbreak time at gamma={g:.3e}"] = f"{to:.4f} d"

        ax1.loglog(gammas, t_out, "o", color=colour, ms=3, label=f"model, ${lab}$")
        ax1.loglog(gammas, inverse(gammas, A), "-", color=colour, lw=0.9,
                   label=rf"fit $A/\gamma$, $A = {sci(A, 2)}$")
        ax1.loglog(gammas, inverse(gammas, A_th), ":", color="black", lw=0.8)

    diff = np.max(np.abs(residuals[1.0][0] - residuals[10.0][0]))
    RESULTS["Q2.2c max difference between Z>=H and Z>=10H residuals (A/gamma)"] = \
        f"{diff:.4f} percentage points"

    ax1.plot([], [], ":", color="black", lw=0.8, label="logistic estimate")
    ax1.set_xlabel(r"zombification rate $\gamma$ (day$^{-1}$)")
    ax1.set_ylabel("outbreak time (days)")
    ax1.legend(loc="lower left", fontsize=6.5)
    panel_title(ax1, "a", r"Outbreak time against $\gamma$")

    res_inv, res_shift = residuals[1.0]
    c = conditions[1.0]
    ax2.semilogx(gammas, res_inv, "o-", color=c, ms=3, label=r"$A/\gamma$")
    ax2.semilogx(gammas, res_shift, "s--", color=c, ms=3, mfc="white",
                 label=r"$A/(\gamma - c)$")
    ax2.axhline(0, color="black", lw=0.6)
    ax2.text(2e-2, 0.6, rf"$|$residual$| < {max(0.01, np.abs(res_shift).max()):.2g}\%$"
             "\nfor $A/(\\gamma-c)$", fontsize=6.5, ha="center")
    ax2.set_xlabel(r"zombification rate $\gamma$ (day$^{-1}$)")
    ax2.set_ylabel("residual (%)")
    ax2.legend(loc="upper right")
    panel_title(ax2, "b", r"Fit residuals ($Z \geq H$)")
    save_figure(fig, "q2_2c_outbreak_fit")


# ---------------------------------------------------------------------------
# PART 3a: Vector fields in phase space
# ---------------------------------------------------------------------------
def quiver_field(ax, X, Y, U, V, cmap="magma_r"):
    """Unit-length arrows showing flow direction, coloured by flow speed."""
    speed = np.hypot(U, V)
    with np.errstate(invalid="ignore", divide="ignore"):
        Un, Vn = U / speed, V / speed
    return ax.quiver(X, Y, Un, Vn, speed, cmap=cmap, norm=LogNorm(),
                     angles="xy", pivot="mid", scale=30, width=0.005,
                     headwidth=3.5, alpha=0.85)


def question_3a():
    """Part 3a: vector fields for the rabbit-fox and zombie systems."""
    print("\nPart 3a - vector fields")
    fig, (ax1, ax2) = new_figure(2, height=2.7)
    speed_label = "flow speed (individuals day$^{-1}$)"

    # ---- Rabbit-fox ----
    p = LV_PARAMS
    X, Y = np.meshgrid(np.linspace(0.5, 120, 17), np.linspace(0.5, 36, 14))
    dX, dY = lotka_volterra(0, np.array([X, Y]), p)
    q = quiver_field(ax1, X, Y, dX, dY)
    fig.colorbar(q, ax=ax1, pad=0.01, aspect=25).set_label(speed_label)
    t = make_time_grid(0, 15, 0.01)
    starts = np.array([(6, 3), (9, 4), (12, 5), (15, 6), (18, 7)], float)
    u = rk4(lotka_volterra, starts.T, t, p)          # all orbits in one run
    for j, c in enumerate(plt.get_cmap("Greens")(np.linspace(0.4, 1, len(starts)))):
        x0, y0 = starts[j]
        ax1.plot(u[:, 0, j], u[:, 1, j], color=c, lw=1.3, label=f"({x0:g}, {y0:g})")
    x_s, y_s = lv_fixed_points(p)[1]
    ax1.plot(x_s, y_s, "*", color="gold", mec="black", mew=0.6, ms=9, zorder=6)
    ax1.set_xlim(-2, 122)
    ax1.set_ylim(-1, 37)
    ax1.set_xlabel("rabbits $x$")
    ax1.set_ylabel("foxes $y$")
    ax1.grid(False)
    ax1.legend(title="$(x_0, y_0)$", title_fontsize=6.5, fontsize=6.5,
               loc="upper right", frameon=True, framealpha=0.92, edgecolor="none")
    panel_title(ax1, "a", "Rabbit-fox flow field")

    # ---- Zombie system ----
    # With alpha = 0, H + Z + D = N is conserved, so the 3D flow lives on a
    # plane: D = N - H - Z. Plot the flow projected onto the (H, Z) plane.
    pz = Z_PARAMS
    N = Z_U0.sum()
    Hg, Zg = np.meshgrid(np.linspace(10, N, 15), np.linspace(10, N, 15))
    Dg = N - Hg - Zg
    dH, dZ, _ = zombie(0, np.array([Hg, Zg, Dg]), pz)
    outside = Dg < 0                                  # not physical (D < 0)
    dH[outside], dZ[outside] = np.nan, np.nan
    q2 = quiver_field(ax2, Hg, Zg, dH, dZ)
    fig.colorbar(q2, ax=ax2, pad=0.01, aspect=25).set_label(speed_label)
    ax2.plot([0, N], [N, 0], color="black", lw=0.7, ls="--")
    ax2.text(610, 470, "$D = 0$", rotation=-40, fontsize=7)
    t = make_time_grid(0, 10, 0.005)
    # Last start has no zombies at all: resurrection of the dead seeds the outbreak
    starts = np.array([(1000, 5, 0), (700, 5, 300), (400, 5, 600),
                       (150, 5, 850), (505, 0, 500)], float)
    u = rk4(zombie, starts.T, t, pz)
    for j, c in enumerate(sweep_colours(len(starts), "cool")):
        H0, Z0, D0 = starts[j]
        ax2.plot(u[:, 0, j], u[:, 1, j], color=c, lw=1.3,
                 label=f"({H0:g}, {Z0:g}, {D0:g})")
        ax2.plot(H0, Z0, "o", color=c, ms=3, clip_on=False, zorder=6)
    ax2.plot(0, N, "*", color="gold", mec="black", mew=0.6, ms=9, zorder=6, clip_on=False)
    ax2.annotate("attractor\n$(0, N, 0)$", xy=(10, N - 10), xytext=(150, 870),
                 fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.7))
    ax2.set_xlim(-10, N + 10)
    ax2.set_ylim(-10, N + 10)
    ax2.set_xlabel("humans $H$")
    ax2.set_ylabel("zombies $Z$")
    ax2.grid(False)
    ax2.legend(title="$(H_0, Z_0, D_0)$", title_fontsize=6.5, fontsize=6.5,
               loc="upper right", frameon=True, framealpha=0.92, edgecolor="none")
    panel_title(ax2, "b", "Zombie flow on plane $H+Z+D=N$")
    save_figure(fig, "q3a_vector_fields")

    t_seed = outbreak_time(t, u[:, :, -1])
    RESULTS["Q3a zombie equilibria"] = \
        ("every point with H = D = 0 is an equilibrium (a line); on the conserved "
         "plane H+Z+D=N the only one is (0, N, 0), which attracts all orbits")
    RESULTS["Q3a start (505, 0, 500): no zombies initially, Z = H at"] = \
        f"{t_seed:.2f} d (outbreak seeded by resurrection)"


# ---------------------------------------------------------------------------
# PART 3b: Time-varying parameters
# ---------------------------------------------------------------------------
def learning_delta(delta0, delta_max, tau):
    """
    Humans learn to fight back: the zombie destruction rate rises from
    delta0 towards delta_max on a learning timescale tau (days):
        delta(t) = delta_max - (delta_max - delta0) exp(-t / tau)
    tau may be an array (one value per batched run); tau = inf means no learning.
    """
    return lambda t: delta_max - (delta_max - delta0) * np.exp(-t / tau)


def question_3b_learning():
    """Zombie model with a time-dependent destruction rate delta(t)."""
    print("\nPart 3b - humans learn to fight (delta(t))")
    delta0, delta_max, gamma = Z_PARAMS["delta"], 3e-2, Z_PARAMS["gamma"]
    t = make_time_grid(0.0, 10.0, 0.005)
    taus = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
    tau_scan = np.logspace(-2, 1, 40)
    colours = sweep_colours(len(taus), "viridis")

    # One batched run: tau = inf (constant delta), the example taus for
    # panel (a), then the fine scan for panel (b)
    all_taus = np.concatenate([[np.inf], taus, tau_scan])
    u0 = np.repeat(Z_U0[:, None], len(all_taus), axis=1)
    p = dict(Z_PARAMS, delta=learning_delta(delta0, delta_max, all_taus))
    u_all = rk4(zombie, u0, t, p)

    fig, (ax1, ax2) = new_figure(2)
    ax1.plot(t, u_all[:, 0, 0], color="black", ls="--", lw=0.9, label=r"constant $\delta$")
    for j, (tau, c) in enumerate(zip(taus, colours), start=1):
        ax1.plot(t, u_all[:, 0, j], color=c, label=rf"$\tau$ = {tau:g} d")
    ax1.set_xlabel("time (days)")
    ax1.set_ylabel("humans $H$")
    ax1.set_xlim(0, 4)
    ax1.set_ylim(0, 1050)
    ax1.legend(loc="center right", ncol=2, fontsize=6.5, columnspacing=1.0)
    panel_title(ax1, "a", r"Humans with rising $\delta(t)$")

    H_end = u_all[-1, 0, 1 + len(taus):]
    tau_half = np.interp(-Z_U0[0] / 2, -H_end, tau_scan)    # H_end falls with tau
    ax2.semilogx(tau_scan, H_end, "o-", color=COLOURS["humans"], ms=2.5)
    ax2.axhline(Z_U0[0] / 2, color="grey", lw=0.6, ls="--")
    ax2.axvline(tau_half, color=COLOURS["zombies"], lw=0.7, ls="--")
    ax2.text(tau_half * 0.85, 300, f"half survive\nfor $\\tau <$ {tau_half:.2f} d",
             fontsize=6.5, color=COLOURS["zombies"], ha="right")
    ax2.set_xlabel(r"learning timescale $\tau$ (days)")
    ax2.set_ylabel("humans left at $t$ = 10 d")
    ax2.set_ylim(0, 1050)
    panel_title(ax2, "b", "Survival against learning speed")
    save_figure(fig, "q3b_learning")

    # Time at which delta(t) overtakes gamma (zombies then start to shrink)
    crossover = np.log((delta_max - delta0) / (delta_max - gamma))
    RESULTS["Q3b learning: delta_max"] = f"{delta_max:g} day^-1 (gamma = {gamma:g})"
    RESULTS["Q3b learning: tau at which half of humans survive to t=10 d"] = \
        f"{tau_half:.3f} days"
    RESULTS["Q3b learning: delta(t) passes gamma at t = c*tau, c"] = \
        f"{crossover:.3f}; for tau_half this is {crossover * tau_half:.2f} d"
    for j, tau in enumerate(taus, start=1):
        RESULTS[f"Q3b learning: humans left at 10 d, tau={tau:g}"] = \
            f"{u_all[-1, 0, j]:.0f}"


def seasonal_alpha(alpha0, amp, period):
    """Seasonal rabbit birth rate alpha(t) = alpha0 (1 + amp sin(2 pi t / P))."""
    return lambda t: alpha0 * (1 + amp * np.sin(2 * np.pi * t / period))


def question_3b_seasonal(T):
    """
    Rabbit-fox model with a seasonal rabbit birth rate. The strength of the
    response is measured by the lowest rabbit population reached in 200 days,
    swept over the forcing period P (in units of the natural period T).
    """
    print("\nPart 3b - seasonal rabbit birth rate (alpha(t))")
    amp, t_end = 0.3, 200.0
    ratios = np.linspace(0.3, 3.5, 321)
    examples = {"b": 2.5, "c": 1.0}                    # P/T for the orbit panels
    periods = np.concatenate([[np.inf], ratios, list(examples.values())]) * T

    # One batched run over every forcing period (P = inf: unforced)
    t = make_time_grid(0.0, t_end, 0.02)
    u0 = np.repeat(LV_U0[:, None], len(periods), axis=1)
    p = dict(LV_PARAMS, alpha=seasonal_alpha(LV_PARAMS["alpha"], amp, periods))
    u_all = rk4(lotka_volterra, u0, t, p)
    x_min = u_all[:, 0, :].min(axis=0)
    unforced_min = x_min[0]
    sweep_min = x_min[1:1 + len(ratios)]

    fig, axes = new_figure(3, height=2.4, width_ratios=[1.25, 1, 1.15])
    ax = axes[0]
    ax.semilogy(ratios, sweep_min, color=COLOURS["rabbits"], lw=0.9)
    # The unforced minimum (~1 rabbit) doubles as the "one individual" level
    ax.axhline(unforced_min, color="grey", lw=0.7, ls="--")
    ax.text(3.45, unforced_min * 2, "unforced", fontsize=6.5, color="grey", ha="right")
    ax.set_ylim(top=30)
    for m in (1, 2, 3):
        ax.axvline(m, color=COLOURS["foxes"], lw=0.6, ls=":")
    for letter, r in examples.items():
        ax.plot(r, x_min[-2 if letter == "b" else -1], "o", mfc="white",
                mec="black", ms=4, zorder=5)
        ax.annotate(f"({letter})", (r, x_min[-2 if letter == "b" else -1]),
                    xytext=(4, -9), textcoords="offset points", fontsize=6.5)
    ax.set_xlabel("forcing period $P / T$")
    ax.set_ylabel("lowest rabbit population")
    ax.set_xlim(ratios[0], ratios[-1])
    panel_title(ax, "a", "Response to forcing period")

    u_free = u_all[:, :, 0]
    for (letter, r), ax, idx in zip(examples.items(), axes[1:], (-2, -1)):
        u = u_all[:, :, idx]
        ax.plot(u_free[:, 0], u_free[:, 1], color="grey", lw=0.8, ls="--")
        plot_orbit(ax, u[:, 0], u[:, 1], t, cbar=(letter == "c"), lw=0.9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("rabbits $x$")
        ax.set_ylabel("foxes $y$")
        panel_title(ax, letter, "$P = T$" if r == 1 else f"$P = {r:g}T$")
        RESULTS[f"Q3b seasonal P={r:g}T: rabbit range over 200 d"] = \
            f"{u[:, 0].min():.2e} to {u[:, 0].max():.1f}"
    save_figure(fig, "q3b_seasonal")

    # Summarise the strongest responses near each multiple of T
    RESULTS["Q3b seasonal: alpha(t)"] = f"1 + {amp} sin(2 pi t / P), 200 days, h = 0.02"
    RESULTS["Q3b seasonal: unforced lowest rabbits"] = f"{unforced_min:.2f}"
    for lo, hi in ((0.3, 0.8), (0.8, 1.3), (1.3, 2.3), (2.3, 3.5)):
        sel = (ratios >= lo) & (ratios < hi)
        i = np.argmin(sweep_min[sel])
        RESULTS[f"Q3b seasonal: lowest rabbits for P/T in [{lo}, {hi})"] = \
            f"{sweep_min[sel][i]:.1e} at P/T = {ratios[sel][i]:.2f}"


# ---------------------------------------------------------------------------
# Main: run every question in order
# ---------------------------------------------------------------------------
def write_results():
    """Write the collected numerical results to a text file."""
    path = os.path.join(FIG_DIR, "results_summary.txt")
    with open(path, "w", encoding="utf-8") as fh:
        for key, val in RESULTS.items():
            fh.write(f"{key}: {val}\n")
    print("\nResults written to figures/results_summary.txt")


def main():
    t, u_euler, u_rk4 = solve_lotka_volterra()
    question_1_1(t, u_euler, u_rk4)
    T = question_1_2(t, u_rk4)
    compare_methods(t, u_euler, u_rk4)
    question_2_1()
    question_2_2a()
    question_2_2bc()
    question_3a()
    question_3b_learning()
    question_3b_seasonal(T)
    write_results()
    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
