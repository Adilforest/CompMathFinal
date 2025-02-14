import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


def solve_task(x_input, y_input, axes=None):
    """
    Solves the task:
      Performs cubic spline interpolation on given data points.
      Generates interpolated values and plots the results.
    Parameters:
      x_input: String of space-separated x values (e.g., "0 0.5 1.0 1.5").
      y_input: String of space-separated y values (e.g., "0 0.25 0.75 2.25").
      axes: A matplotlib.axes object to draw the plot on. If None, a new figure is created.
    Returns:
      dict with keys:
         - "x_data": Array of input x values,
         - "y_data": Array of input y values,
         - "x_interp": Array of x values for plotting the spline,
         - "y_interp": Array of interpolated y values,
         - "x_table": Array of x values for the table,
         - "y_table": Array of interpolated y values for the table.
    """
    # ===============================
    # Step 1. Parse input data
    # ===============================
    x_data = np.array([float(val) for val in x_input.split()])
    y_data = np.array([float(val) for val in y_input.split()])

    # Check if the number of x and y values match
    if len(x_data) != len(y_data):
        raise ValueError("Error: The number of x and y values must match!")

    # ===============================
    # Step 2. Create cubic spline interpolator
    # ===============================
    # Use natural boundary conditions (second derivative equals 0 at endpoints)
    cs = CubicSpline(x_data, y_data, bc_type="natural")

    # ===============================
    # Step 3. Generate data for plotting and table
    # ===============================
    # Generate a dense grid for plotting the spline curve
    x_interp = np.linspace(np.min(x_data), np.max(x_data), 200)
    y_interp = cs(x_interp)

    # Select 10 evenly spaced points for the table
    x_table = np.linspace(np.min(x_data), np.max(x_data), 10)
    y_table = cs(x_table)

    # Print the table of interpolated values
    print("\nTable of interpolated values:")
    print("{:<10} {:<15}".format("x", "Spline(y)"))
    for xi, yi in zip(x_table, y_table):
        print("{:<10.4f} {:<15.4f}".format(xi, yi))

    # ===============================
    # Step 4. Plot the results
    # ===============================
    if axes is None:
        fig, axes = plt.subplots(figsize=(8, 6))
    else:
        axes.clear()

    # Plot the cubic spline curve
    axes.plot(x_interp, y_interp, label="Cubic Spline", color="blue", linewidth=2)
    # Plot the original data points
    axes.scatter(x_data, y_data, color="red", label="Input Data", zorder=5)
    # Add labels and title
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_title("Cubic Spline Interpolation")
    axes.legend()
    axes.grid(True)
    if axes.figure is not None:
        axes.figure.canvas.draw_idle()

    # ===============================
    # Step 5. Return results as a dictionary
    # ===============================
    return {
        "x_data": x_data,
        "y_data": y_data,
        "x_interp": x_interp,
        "y_interp": y_interp,
        "x_table": x_table,
        "y_table": y_table,
    }


# Example usage
if __name__ == "__main__":
    # Input from the user
    x_input = input("Enter x values separated by spaces (e.g., 0 0.5 1.0 1.5): ")
    y_input = input("Enter y values separated by spaces (e.g., 0 0.25 0.75 2.25): ")

    # Solve the task
    result = solve_task(x_input, y_input)

    # Results are printed within the function
    plt.show()
