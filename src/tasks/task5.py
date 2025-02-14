import numpy as np
import matplotlib.pyplot as plt

def get_initial_value():
    """
    Prompts the user to enter the initial value n for the data.

    Returns:
        n (float): The initial value.
    """
    n = float(input("Enter the initial value n for the data (e.g., 0): "))
    return n

def generate_data(n):
    """
    Generates data points based on the initial value n.

    The x-data consists of four points: n, n+1, n+2, n+3.
    The y-data is computed as y = exp(x) for each x.

    Parameters:
        n (float): The starting value.

    Returns:
        x_data (numpy.ndarray): Array of x values.
        y_data (numpy.ndarray): Array of y values computed as exp(x).
    """
    x_data = np.array([n, n+1, n+2, n+3])
    y_data = np.exp(x_data)
    return x_data, y_data

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

def print_comparison_table(x_data, y_data, A_fit, B_fit):
    """
    Prints a comparison table of the original data and the model approximations.

    The table includes:
      - x: The x value.
      - y (data): The original y value.
      - y (model): The y value computed from the fitted model.
      - Error: The difference between the data and model values.

    Parameters:
        x_data (numpy.ndarray): Array of x values.
        y_data (numpy.ndarray): Array of y values.
        A_fit (float): Fitted coefficient A.
        B_fit (float): Fitted coefficient B.
    """
    print("\nModel coefficients:")
    print(f"A = {A_fit:.6f}")
    print(f"B = {B_fit:.6f}")

    print("\nComparison Table:")
    print("{:<10} {:<15} {:<15} {:<15}".format("x", "y (data)", "y (model)", "Error"))
    for xi, yi in zip(x_data, y_data):
        yi_fit = A_fit * np.exp(B_fit * xi)
        error = yi - yi_fit
        print(f"{xi:<10.4f} {yi:<15.4f} {yi_fit:<15.4f} {error:<15.4e}")

def plot_approximation(x_data, y_data, A_fit, B_fit):
    """
    Plots the original data and the fitted exponential model.

    A smooth curve is generated for the model fit.

    Parameters:
        x_data (numpy.ndarray): Array of x values.
        y_data (numpy.ndarray): Array of y values.
        A_fit (float): Fitted coefficient A.
        B_fit (float): Fitted coefficient B.
    """
    # Create a smooth curve for the fitted model
    x_fit = np.linspace(x_data[0], x_data[-1], 100)
    y_fit = A_fit * np.exp(B_fit * x_fit)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_data, color='red', label='Original Data', zorder=5)
    plt.plot(x_fit, y_fit, label='Fitted Model', color='blue', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Exponential Fit: y = A * exp(B * x)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Step 1: Get the initial value from the user
    n = get_initial_value()

    # Step 2: Generate the data points
    x_data, y_data = generate_data(n)

    # Step 3: Perform the exponential fit
    A_fit, B_fit = exponential_fit(x_data, y_data)

    # Step 4: Print the comparison table
    print_comparison_table(x_data, y_data, A_fit, B_fit)

    # Step 5: Plot the data and the fitted model
    plot_approximation(x_data, y_data, A_fit, B_fit)

if __name__ == "__main__":
    main()
