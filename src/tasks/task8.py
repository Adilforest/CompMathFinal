import numpy as np
import matplotlib.pyplot as plt

def read_input():
    """
    Prompts the user to enter the number of subintervals.
    The number must be even for Simpson's rule.

    Returns:
        n (int): Number of subintervals.
        a (float): Lower bound of integration.
        b (float): Upper bound of integration.
    """
    n = int(input("Enter the number of subintervals (even number, e.g., 10): "))
    if n % 2 != 0:
        raise ValueError("The number of subintervals must be even for Simpson's rule.")
    a = 0.0
    b = np.pi
    return n, a, b

def simpson_rule(n, a, b, f):
    """
    Computes the integral of the function f over [a, b] using Simpson's 1/3 rule.

    Parameters:
        n (int): Number of subintervals (must be even).
        a (float): Lower limit of integration.
        b (float): Upper limit of integration.
        f (function): The function to integrate.

    Returns:
        I (float): The approximated integral.
        x_vals (ndarray): Array of node points.
        f_vals (ndarray): Array of function values at the node points.
    """
    h = (b - a) / n  # Step size
    x_vals = np.linspace(a, b, n+1)
    f_vals = f(x_vals)

    # Simpson's 1/3 rule formula
    I = f_vals[0] + f_vals[-1]
    I += 4 * np.sum(f_vals[1:-1:2])
    I += 2 * np.sum(f_vals[2:-1:2])
    I *= h / 3

    return I, x_vals, f_vals

def print_table(x_vals, f_vals):
    """
    Prints a table of the node points and corresponding f(x) values.

    Parameters:
        x_vals (ndarray): Array of x values.
        f_vals (ndarray): Array of f(x) values.
    """
    print("\nTable of f(x)=sin(x) values:")
    print("{:<10} {:<15}".format("x", "sin(x)"))
    for x, val in zip(x_vals, f_vals):
        print("{:<10.4f} {:<15.8f}".format(x, val))

def print_results(I_simpson, I_exact, abs_error):
    """
    Prints the computed integral, the exact integral, and the absolute error.

    Parameters:
        I_simpson (float): Approximated integral.
        I_exact (float): Exact value of the integral.
        abs_error (float): Absolute error.
    """
    print("\nComputed integral using Simpson's 1/3 rule:", I_simpson)
    print("Exact value of the integral:", I_exact)
    print("Absolute error:", abs_error)

def plot_integration(a, b, x_vals, f_vals, f):
    """
    Plots the function f(x) on the interval [a, b] and marks the node points used in Simpson's rule.

    Parameters:
        a (float): Lower bound.
        b (float): Upper bound.
        x_vals (ndarray): Node points.
        f_vals (ndarray): Function values at node points.
        f (function): The function to plot.
    """
    x_plot = np.linspace(a, b, 400)
    y_plot = f(x_plot)

    plt.figure(figsize=(8, 6))
    plt.plot(x_plot, y_plot, label='f(x)=sin(x)', color='blue')
    plt.scatter(x_vals, f_vals, color='red', zorder=5, label='Node Points')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.title("Integration of sin(x) from 0 to π using Simpson's 1/3 Rule")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Step 1: Read input data from the user
    n, a, b = read_input()

    # Step 2: Compute the integral using Simpson's 1/3 rule
    I_simpson, x_vals, f_vals = simpson_rule(n, a, b, np.sin)

    # Print the table of node points and f(x) values
    print_table(x_vals, f_vals)

    # Step 3: Print the computed integral, exact value, and absolute error
    I_exact = 2.0  # The exact value of the integral ∫ sin(x) dx from 0 to π is 2
    abs_error = abs(I_exact - I_simpson)
    print_results(I_simpson, I_exact, abs_error)

    # Step 4: Plot the function and the node points
    plot_integration(a, b, x_vals, f_vals, np.sin)

if __name__ == "__main__":
    main()
