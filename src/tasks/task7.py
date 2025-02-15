from decimal import Decimal
import numpy as np
import matplotlib.pyplot as plt


def solve_task(x_input: float, y0_value: float = 1.0, axes=None):
    """
    Solves the task:
      Computes approximations of y(x) using Picard's method at a given point x.
      Plots successive approximations and returns results in a dictionary.
    Parameters:
      x_input: The value of x (can be passed as Decimal, str, or float).
      y0: The initial value y0(0)=1.0 by default.
      axes: A matplotlib.axes object to draw the plot on. If None, a new figure is created.
    Returns:
      dict with keys:
         - "x_plot": Array of x values for plotting,
         - "y_plots": Dictionary of y values for each approximation,
         - "approx_values": Dictionary of y(x) values at the given x,
         - "final_approx": Value of y(x) from the 4th approximation.
    """
    # Convert input to float
    x_val = float(x_input)

    # ===============================
    # Step 2. Define Picard's approximations
    # ===============================
    def y0(x):
        # Initial approximation: constant function y0(x)=y0
        return y0_value

    def y1(x):
        # First approximation: y1(x)=1 + x + x^2/2
        return y0_value + x + x**2 / 2

    def y2(x):
        # Second approximation: y2(x)=1 + x + x^2 + x^3/6
        return y0_value + x + x**2 + x**3 / 6

    def y3(x):
        # Third approximation: y3(x)=1 + x + x^2 + x^3/3 + x^4/24
        return y0_value + x + x**2 + x**3 / 3 + x**4 / 24

    def y4(x):
        # Fourth approximation: y4(x)=1 + x + x^2 + x^3/3 + x^4/12 + x^5/120
        return y0_value + x + x**2 + x**3 / 3 + x**4 / 12 + x**5 / 120

    # ===============================
    # Step 3. Compute approximations at x_val
    # ===============================
    approx_values = {
        "y0": y0(x_val),
        "y1": y1(x_val),
        "y2": y2(x_val),
        "y3": y3(x_val),
        "y4": y4(x_val),
    }

    # ===============================
    # Step 4. Plot approximations
    # ===============================
    x_plot = np.linspace(0, x_input + 1, 200)
    y_plots = {
        "y0": np.array([y0(x) for x in x_plot]),
        "y1": np.array([y1(x) for x in x_plot]),
        "y2": np.array([y2(x) for x in x_plot]),
        "y3": np.array([y3(x) for x in x_plot]),
        "y4": np.array([y4(x) for x in x_plot]),
    }

    if axes is None:
        fig, axes = plt.subplots(figsize=(10, 6))
    else:
        axes.clear()

    # Plot each approximation
    axes.plot(x_plot, y_plots["y0"], label="y0(x)=1", linestyle="--")
    axes.plot(x_plot, y_plots["y1"], label="y1(x)=1+x+x²/2", linestyle="-")
    axes.plot(x_plot, y_plots["y2"], label="y2(x)=1+x+x²+x³/6", linestyle="-.")
    axes.plot(x_plot, y_plots["y3"], label="y3(x)=1+x+x²+x³/3+x⁴/24", linestyle=":")
    axes.plot(
        x_plot, y_plots["y4"], label="y4(x)=1+x+x²+x³/3+x⁴/12+x⁵/120", linewidth=2
    )

    # Mark the value at x_val for each approximation
    axes.scatter([x_val] * 5, list(approx_values.values()), color="red", zorder=5)

    # Add labels and title
    axes.set_xlabel("x")
    axes.set_ylabel("y(x)")
    axes.set_title(
        "Picard's Method: Successive Approximations of the Differential Equation Solution"
    )
    axes.legend()
    axes.grid(True)
    if axes.figure is not None:
        axes.figure.canvas.draw_idle()

    # ===============================
    # Step 5. Return results as a dictionary
    # ===============================
    return {
        "x_plot": x_plot,
        "y_plots": y_plots,
        "approx_values": approx_values,
        "final_approx": approx_values["y4"],
    }


# Example usage
if __name__ == "__main__":
    # Input from the user
    x_input = Decimal(input("Enter the value of x to compute y (e.g., 0.2): "))

    # Solve the task
    result = solve_task(x_input)

    # Print the table of approximations
    print("\nTable of approximate values using Picard's method:")
    print("{:<10} {:<20}".format("Iteration", "y(x)"))
    for key, value in result["approx_values"].items():
        print("{:<10} {:<20.12f}".format(key, value))

    print(
        "\nValue of y at x = {:.4f} using the 4th approximation: {:.12f}".format(
            float(x_input), result["final_approx"]
        )
    )

    plot = plt.show()
