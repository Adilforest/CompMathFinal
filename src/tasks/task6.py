import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def get_user_data():
    """
    Prompts the user to input x and y values separated by spaces.

    Returns:
        x_data (numpy.ndarray): Array of x values.
        y_data (numpy.ndarray): Array of y values.
    """
    x_input = input("Enter x values separated by space (e.g., 0 0.5 1.0 1.5): ")
    x_data = np.array([float(val) for val in x_input.split()])

    y_input = input("Enter y values separated by space (e.g., 0 0.25 0.75 2.25): ")
    y_data = np.array([float(val) for val in y_input.split()])

    if len(x_data) != len(y_data):
        print("Error: The number of x and y values must be equal!")
        exit()

    return x_data, y_data

def create_spline_interpolator(x_data, y_data):
    """
    Creates a cubic spline interpolator with natural boundary conditions.

    Parameters:
        x_data (numpy.ndarray): Array of x values.
        y_data (numpy.ndarray): Array of y values.

    Returns:
        cs (CubicSpline): Cubic spline interpolator.
    """
    cs = CubicSpline(x_data, y_data, bc_type='natural')
    return cs

def generate_interpolation_data(x_data, cs):
    """
    Generates data for plotting the spline curve and for a comparison table.

    Parameters:
        x_data (numpy.ndarray): Array of original x values.
        cs (CubicSpline): Cubic spline interpolator.

    Returns:
        x_interp (numpy.ndarray): Dense x values for plotting the curve.
        y_interp (numpy.ndarray): Interpolated y values on the dense grid.
        x_table (numpy.ndarray): 10 evenly spaced x values for the table.
        y_table (numpy.ndarray): Interpolated y values for the table.
    """
    x_min, x_max = np.min(x_data), np.max(x_data)
    x_interp = np.linspace(x_min, x_max, 200)
    y_interp = cs(x_interp)

    x_table = np.linspace(x_min, x_max, 10)
    y_table = cs(x_table)

    return x_interp, y_interp, x_table, y_table

def print_interpolation_table(x_table, y_table):
    """
    Prints a table of the interpolated values.

    Parameters:
        x_table (numpy.ndarray): Array of x values for the table.
        y_table (numpy.ndarray): Array of corresponding y values.
    """
    print("\nInterpolation Table:")
    print("{:<10} {:<15}".format("x", "Spline(y)"))
    for xi, yi in zip(x_table, y_table):
        print("{:<10.4f} {:<15.4f}".format(xi, yi))

def plot_spline(x_data, y_data, x_interp, y_interp):
    """
    Plots the cubic spline interpolation curve along with the original data.

    Parameters:
        x_data (numpy.ndarray): Original x values.
        y_data (numpy.ndarray): Original y values.
        x_interp (numpy.ndarray): Dense x values for the interpolation curve.
        y_interp (numpy.ndarray): Interpolated y values on the dense grid.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(x_interp, y_interp, label='Cubic Spline', color='blue', linewidth=2)
    plt.scatter(x_data, y_data, color='red', label='Original Data', zorder=5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cubic Spline Interpolation')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Step 1: Get input data from the user
    x_data, y_data = get_user_data()

    # Step 2: Create the cubic spline interpolator with natural boundary conditions
    cs = create_spline_interpolator(x_data, y_data)

    # Step 3: Generate data for the curve and table
    x_interp, y_interp, x_table, y_table = generate_interpolation_data(x_data, cs)

    # Print the interpolation table
    print_interpolation_table(x_table, y_table)

    # Step 4: Plot the interpolation curve along with the original data points
    plot_spline(x_data, y_data, x_interp, y_interp)

if __name__ == "__main__":
    main()
