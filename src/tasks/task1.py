import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

def f(x, a, b, c):
    """
    Computes the value of the function f(x) = a*x^4 - b*x^2 + c.
    """
    return a * x**4 - b * x**2 + c

def plot_function(a, b, c, n):
    """
    Plots the function f(x) = a*x^4 - b*x^2 + c on the interval [-n, n].
    Returns the array of x nodes and the corresponding f(x) values.
    """
    x_vals = np.linspace(-n, n, 1000)
    f_vals = f(x_vals, a, b, c)

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, f_vals, label=f'f(x) = {a}x⁴ - {b}x² + {c}', color='blue')
    plt.axhline(0, color='black', linewidth=0.5)  # x-axis
    plt.axvline(0, color='black', linewidth=0.5)  # y-axis
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Function Plot on the Interval [-{n}, {n}]')
    plt.legend()
    plt.grid(True)
    plt.show()

    return x_vals, f_vals

def find_graphical_root(x_vals, f_vals):
    """
    Finds an approximate root using a graphical search method.
    It searches for the first sign change between neighboring points,
    and the root is approximated as the average of these x values.
    Returns a tuple (root, index of the start of the interval).
    """
    for i in range(len(x_vals) - 1):
        if f_vals[i] * f_vals[i+1] < 0:
            approx_root = (x_vals[i] + x_vals[i+1]) / 2.0
            return approx_root, i
    return None, None

def bisection_method(f, x_left, x_right, a, b, c, tol=1e-10, max_iter=1000):
    """
    Finds the root of the function f(x) = a*x^4 - b*x^2 + c on the interval [x_left, x_right]
    using the bisection method.

    Returns a tuple (approximate root, number of iterations).
    """
    iterations = 0
    left = x_left
    right = x_right

    while (right - left) > tol and iterations < max_iter:
        mid = (left + right) / 2.0
        f_mid = f(mid, a, b, c)

        if abs(f_mid) < tol:
            return mid, iterations

        if f(left, a, b, c) * f_mid < 0:
            right = mid
        else:
            left = mid

        iterations += 1

    return (left + right) / 2.0, iterations

def main():
    # Input coefficients and interval
    n_input = Decimal(input("Enter the value of n (the interval will be [-n, n]): "))
    a_input = Decimal(input("Enter coefficient a (for x^4): "))
    b_input = Decimal(input("Enter coefficient b (for x^2): "))
    c_input = Decimal(input("Enter coefficient c (constant term): "))

    # Convert to float
    n_val = float(n_input)
    a_val = float(a_input)
    b_val = float(b_input)
    c_val = float(c_input)

    # If the target function is x^4 - 10x^2 + 9, then enter a=1, b=10, c=9.

    # Plot the function
    x_vals, f_vals = plot_function(a_val, b_val, c_val, n_val)

    # Search for a root using the graphical method
    approx_root_graph, index = find_graphical_root(x_vals, f_vals)
    if approx_root_graph is None:
        print("No sign change detected on the interval. The graphical method could not find a root.")
        return
    else:
        print("Approximate root found by the graphical method:", approx_root_graph)

    # Refine the root using the bisection method
    x_left = x_vals[index]
    x_right = x_vals[index+1]
    approx_root_bis, iter_count = bisection_method(f, x_left, x_right, a_val, b_val, c_val)
    print("Root found by the bisection method:", approx_root_bis)
    print("Number of iterations in the bisection method:", iter_count)

    # Absolute error
    abs_error = abs(approx_root_graph - approx_root_bis)
    print("Absolute error between the graphical method and the bisection method:", abs_error)

if __name__ == "__main__":
    main()
