from decimal import Decimal
import numpy as np
import matplotlib.pyplot as plt

def get_x_value():
    """
    Step 1: Get the value of x from the user.

    Prompts the user to enter a value for x (e.g., 0.2) and returns it as a float.
    """
    x_input = Decimal(input("Enter the value of x to compute y (e.g., 0.2): "))
    return float(x_input)

def y0(x):
    """
    Initial approximation: y0(x) = 1.
    """
    return 1.0

def y1(x):
    """
    First approximation: y1(x) = 1 + x + x²/2.
    """
    return 1.0 + x + x**2 / 2

def y2(x):
    """
    Second approximation: y2(x) = 1 + x + x² + x³/6.
    """
    return 1.0 + x + x**2 + x**3 / 6

def y3(x):
    """
    Third approximation: y3(x) = 1 + x + x² + x³/3 + x⁴/24.
    """
    return 1.0 + x + x**2 + x**3 / 3 + x**4 / 24

def y4(x):
    """
    Fourth approximation: y4(x) = 1 + x + x² + x³/3 + x⁴/12 + x⁵/120.
    """
    return 1.0 + x + x**2 + x**3 / 3 + x**4 / 12 + x**5 / 120

def compute_approximations(x_val):
    """
    Computes the approximated y values at x_val using the Picard method.

    Returns:
        A tuple containing (y0, y1, y2, y3, y4) evaluated at x_val.
    """
    y0_val = y0(x_val)
    y1_val = y1(x_val)
    y2_val = y2(x_val)
    y3_val = y3(x_val)
    y4_val = y4(x_val)
    return y0_val, y1_val, y2_val, y3_val, y4_val

def print_approximation_table(x_val, approximations):
    """
    Prints a table of successive Picard approximations.

    Parameters:
        x_val (float): The x value at which the approximations are computed.
        approximations (tuple): The tuple (y0, y1, y2, y3, y4).
    """
    y0_val, y1_val, y2_val, y3_val, y4_val = approximations
    print("\nTable of approximate values using the Picard method:")
    print("{:<10} {:<20}".format("Iteration", "y(x)"))
    print("{:<10} {:<20.12f}".format("y0", y0_val))
    print("{:<10} {:<20.12f}".format("y1", y1_val))
    print("{:<10} {:<20.12f}".format("y2", y2_val))
    print("{:<10} {:<20.12f}".format("y3", y3_val))
    print("{:<10} {:<20.12f}".format("y4", y4_val))
    print("\nValue of y at x = {:.4f} by the 4th approximation: {:.12f}".format(x_val, y4_val))

def plot_approximations(x_val):
    """
    Plots the successive Picard approximations on the interval [0, 0.5].

    Also marks the computed values at x = x_val.
    """
    # Generate points for plotting
    x_plot = np.linspace(0, 0.5, 200)
    y0_plot = np.array([y0(x) for x in x_plot])
    y1_plot = np.array([y1(x) for x in x_plot])
    y2_plot = np.array([y2(x) for x in x_plot])
    y3_plot = np.array([y3(x) for x in x_plot])
    y4_plot = np.array([y4(x) for x in x_plot])

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y0_plot, label="y0(x) = 1", linestyle="--")
    plt.plot(x_plot, y1_plot, label="y1(x) = 1 + x + x²/2", linestyle="-")
    plt.plot(x_plot, y2_plot, label="y2(x) = 1 + x + x² + x³/6", linestyle="-.")
    plt.plot(x_plot, y3_plot, label="y3(x) = 1 + x + x² + x³/3 + x⁴/24", linestyle=":")
    plt.plot(x_plot, y4_plot, label="y4(x) = 1 + x + x² + x³/3 + x⁴/12 + x⁵/120", linewidth=2)

    # Compute approximated values at x_val and mark them on the plot
    approximations = compute_approximations(x_val)
    plt.scatter([x_val]*5, list(approximations), color="red", zorder=5)

    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title("Picard Method: Successive Approximations for the Differential Equation Solution")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Step 1: Get x value from the user
    x_val = get_x_value()

    # Step 2: Compute the approximations at x_val
    approximations = compute_approximations(x_val)

    # Step 3: Print the table of approximations
    print_approximation_table(x_val, approximations)

    # Step 4: Plot the approximations on [0, 0.5]
    plot_approximations(x_val)

if __name__ == "__main__":
    main()
