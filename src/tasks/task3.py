from decimal import Decimal
import matplotlib.pyplot as plt


def solve_task(omega, a, b, c, axes=None, tol=1e-10, max_iter=1000):
    """
    Solves the task by performing the following steps:
    1. Perform the relaxation method for solving the system.
    2. Print the iteration table.
    3. Print the results.
    4. Plot the convergence of the variables.

    Parameters:
        omega   : relaxation parameter
        a, b, c : coefficients
        axes    : matplotlib axes for plotting
        tol     : tolerance for convergence (max change)
        max_iter: maximum number of iterations
    """
    # Step 2 & 3: Execute the relaxation method
    iterations, x, y, z, iterations_list, x_history, y_history, z_history = (
        relaxation_method(omega, a, b, c, tol, max_iter)
    )

    # Step 3.1: Print the iteration table
    print_iteration_table(iterations_list, x_history, y_history, z_history)

    # Step 4: Print the results
    print_results(iterations, x, y, z, a, b, c)

    if axes is None:
        plt.figure(figsize=(8, 6))
    else:
        axes.clear()

    # Plot the convergence graph
    axes.plot(iterations_list, x_history, label="x", marker="o", markersize=4)
    axes.plot(iterations_list, y_history, label="y", marker="s", markersize=4)
    axes.plot(iterations_list, z_history, label="z", marker="^", markersize=4)

    # Add horizontal lines for the analytical solutions
    exact_x = a - c
    exact_y = a - b
    exact_z = b + c - a
    axes.axhline(exact_x, color="blue", linestyle="--", linewidth=1)
    axes.axhline(exact_y, color="orange", linestyle="--", linewidth=1)
    axes.axhline(exact_z, color="green", linestyle="--", linewidth=1)

    axes.set_xlabel("Iteration")
    axes.set_ylabel("Variable Value")
    axes.set_title("Convergence of Variables Using the Relaxation Method")
    axes.legend()
    axes.grid(True)
    if axes.figure is not None:
        axes.figure.canvas.draw_idle()

    return {
        "iterations": iterations,
        "x": x,
        "y": y,
        "z": z,
        "iterations_list": iterations_list,
        "x_history": x_history,
        "y_history": y_history,
        "z_history": z_history,
    }


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

    # Lists to store iteration history for plotting and table
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

        # Store values for plotting and table
        iterations_list.append(iter)
        x_history.append(x)
        y_history.append(y)
        z_history.append(z)

        # Check for convergence: stop if the maximum change is below tol
        if max(abs(x - x_old), abs(y - y_old), abs(z - z_old)) < tol:
            break

    return iter + 1, x, y, z, iterations_list, x_history, y_history, z_history


def print_iteration_table(iterations_list, x_history, y_history, z_history):
    """
    Step 3.1: Prints the iteration table.

    Prints a formatted table showing the values of x, y, and z at each iteration.
    """
    print("\nIteration Table:")
    print("-" * 40)
    print("{:<10} | {:<10} | {:<10} | {:<10}".format("Iteration", "x", "y", "z"))
    print("-" * 40)
    for i in range(len(iterations_list)):
        print(
            "{:<10} | {:<10.6f} | {:<10.6f} | {:<10.6f}".format(
                iterations_list[i], x_history[i], y_history[i], z_history[i]
            )
        )
    print("-" * 40)


def print_results(iterations, x, y, z, a, b, c):
    """
    Step 4: Outputs the results.

    Prints the approximate solution from the relaxation method,
    and the analytical solution.
    """
    print("\nNumber of iterations:", iterations)  # Keep iteration count in results
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
