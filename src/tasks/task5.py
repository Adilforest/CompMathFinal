import numpy as np
import matplotlib.pyplot as plt


def solve_task(x_input: np.ndarray, y_input: np.ndarray, axes=None):
    """
    Solves the task by performing the following steps:
    1. Get the initial value n from the user.
    2. Generate the data points based on n.
    3. Perform an exponential fit of the model: y = A * exp(B * x).
    4. Print a comparison table of the original data and the model approximations.
    5. Plot the original data and the fitted exponential model.
    """
    # Step 1: Perform the exponential fit
    A_fit, B_fit = exponential_fit(x_input, y_input)

    errors = []
    for xi, yi in zip(x_input, y_input):
        yi_fit = A_fit * np.exp(B_fit * xi)
        error = yi - yi_fit
        errors.append((xi, yi, yi_fit, error))

    if axes is None:
        plt.figure(figsize=(8, 6))
    else:
        axes.clear()
    axes.scatter(x_input, y_input, color="red", label="Original Data", zorder=5)
    x_fit = np.linspace(x_input[0], x_input[-1], 100)
    y_fit = A_fit * np.exp(B_fit * x_fit)
    axes.plot(x_fit, y_fit, label="Fitted Model", color="blue", linewidth=2)
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_title("Exponential Fit: y = A * exp(B * x)")
    axes.legend()
    axes.grid(True)
    if axes.figure is not None:
        axes.figure.canvas.draw_idle()

    return {
        "A_fit": A_fit,
        "B_fit": B_fit,
        "x_fit": x_fit,
        "y_fit": y_fit,
        "errors": errors,
    }


def exponential_fit(x_data, y_data):
    """
    Performs an exponential fit of the model: y = A * exp(B * x)

    By taking the natural logarithm, the model becomes:
        ln(y) = ln(A) + B * x
    which is linear and can be fitted using linear regression.

    Parameters:
        x_data (numpy.ndarray): Array of x values.
        y_data (numpy.ndarray): Array of y values.

    Returns:
        A_fit (float): Fitted coefficient A.
        B_fit (float): Fitted coefficient B.
    """
    # Transform the data: ln(y) = ln(A) + B * x
    Y = np.log(y_data)
    coeff = np.polyfit(x_data, Y, 1)  # coeff[0] = B, coeff[1] = ln(A)
    B_fit = coeff[0]
    lnA_fit = coeff[1]
    A_fit = np.exp(lnA_fit)
    return A_fit, B_fit
