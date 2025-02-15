import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.sin(x)


def solve_task(n1: float, n2: float, count: int = 10, axes=None):
    if count % 2 != 0:
        raise ValueError("The number of subintervals must be even for Simpson's rule.")

    h = (n2 - n1) / count

    x_vals = np.linspace(n1, n2, count + 1)
    f_vals = f(x_vals)

    approx = simpson_rule(f_vals, h)

    if axes is None:
        fig, axes = plt.subplots(figsize=(8, 6))
    else:
        axes.clear()
    axes.plot(x_vals, f_vals, label="f(x)", color="blue")
    axes.scatter(x_vals, f_vals, color="red", zorder=5)
    for i in range(0, count, 2):
        xi = x_vals[i : i + 3]
        yi = f_vals[i : i + 3]
        coeffs = np.polyfit(xi, yi, 2)
        poly = np.poly1d(coeffs)
        x_parab = np.linspace(xi[0], xi[-1], 100)
        axes.fill_between(x_parab, poly(x_parab), alpha=0.2, color="green")
        axes.plot(
            x_parab,
            poly(x_parab),
            linestyle="--",
            color="green",
            label="Approximation" if i == 0 else None,
        )
    axes.axhline(0, color="black", linewidth=0.5)
    axes.set_xlabel("x")
    axes.set_ylabel("f(x)")
    axes.set_title(f"Integration of f(x) from {n1} to {n2} using Simpson's 1/3 Rule")
    axes.legend()
    axes.grid(True)
    if axes.figure is not None:
        axes.figure.canvas.draw_idle()

    return {
        "approx": approx,
        "x_vals": x_vals,
        "f_vals": f_vals,
    }


def simpson_rule(f_vals: np.ndarray, h: float):
    """
    Computes the integral of the function f over [a, b] using Simpson's 1/3 rule.

    Parameters:
        f_vals (ndarray): Array of function values at node points.
        h (float): Step size.

    Returns:
        I (float): The approximated integral.
    """

    # Simpson's 1/3 rule formula
    I = f_vals[0] + f_vals[-1]
    I += 4 * np.sum(f_vals[1:-1:2])
    I += 2 * np.sum(f_vals[2:-1:2])
    I *= h / 3

    return I
