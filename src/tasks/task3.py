from decimal import Decimal
import matplotlib.pyplot as plt

def get_parameters():
    """
    Step 1: Input parameters.

    Prompts the user to enter the relaxation parameter ω and the coefficients a, b, c.
    Returns the corresponding float values.
    """
    omega = Decimal(input("Enter the relaxation parameter ω: "))
    a = Decimal(input("Enter the value of a: "))
    b = Decimal(input("Enter the value of b: "))
    c = Decimal(input("Enter the value of c: "))

    # Convert Decimal to float for computation
    return float(omega), float(a), float(b), float(c)

def relaxation_method(omega, a, b, c, tol=1e-10, max_iter=1000):
    """
    Step 2 & 3: Performs the relaxation method for solving the system.

    The "exact" relations are:
        z = b + c - a
        x = b - z
        y = c - z

    Parameters:
        omega   : relaxation parameter
        a, b, c : coefficients
        tol     : tolerance for convergence (max change)
        max_iter: maximum number of iterations

    Returns:
        iterations      : number of iterations performed
        x, y, z         : approximate solution
        iterations_list : list of iteration numbers (for plotting)
        x_history       : history of x values
        y_history       : history of y values
        z_history       : history of z values
    """
    # Initial approximations (can be arbitrary)
    x = 0.0
    y = 0.0
    z = 0.0

    # Lists to store iteration history for plotting
    iterations_list = []
    x_history = []
    y_history = []
    z_history = []

    for iter in range(max_iter):
        # Save previous values to check for convergence
        x_old, y_old, z_old = x, y, z

        # Update z using the relaxation rule:
        # "Exact" relation: z = b + c - a
        z = (1 - omega) * z_old + omega * (b + c - a)

        # Update x: x = b - z (using the new z)
        x = (1 - omega) * x_old + omega * (b - z)

        # Update y: y = c - z (using the new z)
        y = (1 - omega) * y_old + omega * (c - z)

        # Store values for plotting
        iterations_list.append(iter)
        x_history.append(x)
        y_history.append(y)
        z_history.append(z)

        # Check for convergence: stop if the maximum change is below tol
        if max(abs(x - x_old), abs(y - y_old), abs(z - z_old)) < tol:
            break

    return iter + 1, x, y, z, iterations_list, x_history, y_history, z_history

def print_results(iterations, x, y, z, a, b, c):
    """
    Step 4: Outputs the results.

    Prints the number of iterations, the approximate solution from the relaxation method,
    and the analytical solution.
    """
    print("\nNumber of iterations:", iterations)
    print("Approximate solution using the relaxation method:")
    print(f"x = {x:.12f}")
    print(f"y = {y:.12f}")
    print(f"z = {z:.12f}")

    # Analytical solution
    exact_x = a - c
    exact_y = a - b
    exact_z = b + c - a

    print("\nAnalytical solution:")
    print(f"x = {exact_x:.12f}")
    print(f"y = {exact_y:.12f}")
    print(f"z = {exact_z:.12f}")

def plot_convergence(iterations_list, x_history, y_history, z_history, a, b, c):
    """
    Step 5: Plots the convergence of the variables.

    Plots the iteration history of x, y, and z values, and adds horizontal lines for the
    analytical solutions for comparison.
    """
    # Analytical solution
    exact_x = a - c
    exact_y = a - b
    exact_z = b + c - a

    plt.figure(figsize=(10, 6))
    plt.plot(iterations_list, x_history, label='x', marker='o', markersize=4)
    plt.plot(iterations_list, y_history, label='y', marker='s', markersize=4)
    plt.plot(iterations_list, z_history, label='z', marker='^', markersize=4)

    # Add horizontal lines for the analytical solutions
    plt.axhline(exact_x, color='blue', linestyle='--', linewidth=1)
    plt.axhline(exact_y, color='orange', linestyle='--', linewidth=1)
    plt.axhline(exact_z, color='green', linestyle='--', linewidth=1)

    plt.xlabel('Iteration')
    plt.ylabel('Variable Value')
    plt.title('Convergence of Variables Using the Relaxation Method')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Step 1: Get parameters from the user
    omega, a, b, c = get_parameters()

    # Step 2 & 3: Execute the relaxation method
    tol = 1e-10         # Tolerance for convergence
    max_iter = 1000     # Maximum number of iterations
    iterations, x, y, z, iterations_list, x_history, y_history, z_history = relaxation_method(omega, a, b, c, tol, max_iter)

    # Step 4: Print the results
    print_results(iterations, x, y, z, a, b, c)

    # Step 5: Plot the convergence graph
    plot_convergence(iterations_list, x_history, y_history, z_history, a, b, c)

if __name__ == "__main__":
    main()
