import numpy as np
import matplotlib.pyplot as plt

def f(x, a, b, c):
    """Returns f(x) = x^3 - a*x^2 + b*x - c."""
    return x**3 - a*x**2 + b*x - c

def df(x, a, b):
    """Returns the derivative f'(x) = 3*x^2 - 2*a*x + b."""
    return 3*x**2 - 2*a*x + b

def bisection_method(func, a_int, b_int, tol=1e-10, max_iter=1000):
    """
    Finds a root of the function 'func' in the interval [a_int, b_int] using the bisection method.

    Returns:
      - approximate root
      - number of iterations
    """
    iterations = 0
    # Check if there is a sign change at the endpoints
    if func(a_int) * func(b_int) > 0:
        raise ValueError("No sign change at the endpoints of the interval. Bisection method is not applicable.")

    while (b_int - a_int) > tol and iterations < max_iter:
        mid = (a_int + b_int) / 2.0
        f_mid = func(mid)
        if abs(f_mid) < tol:  # if the function value is very close to 0
            return mid, iterations
        # Determine in which subinterval the sign change occurs
        if func(a_int) * f_mid < 0:
            b_int = mid
        else:
            a_int = mid
        iterations += 1
    return (a_int + b_int) / 2.0, iterations

def newton_raphson_method(func, dfunc, x0, tol=1e-10, max_iter=1000):
    """
    Finds a root of the function 'func' using the Newton-Raphson method starting from initial guess x0.

    Returns:
      - approximate root
      - number of iterations
    """
    x = x0
    iterations = 0
    while iterations < max_iter:
        f_val = func(x)
        d_val = dfunc(x)
        if d_val == 0:
            raise ValueError("Division by zero encountered (derivative is 0).")
        x_new = x - f_val / d_val
        if abs(x_new - x) < tol:
            return x_new, iterations + 1
        x = x_new
        iterations += 1
    return x, iterations

def get_exact_root(a, b, c, lower_bound, upper_bound):
    """
    Attempts to find a real root of the cubic equation f(x) = x^3 - a*x^2 + b*x - c
    that lies within the interval [lower_bound, upper_bound] using np.roots.
    """
    roots = np.roots([1, -a, b, -c])
    exact_root = None
    for r in roots:
        if np.isreal(r):
            r_real = np.real(r)
            if lower_bound <= r_real <= upper_bound:
                exact_root = r_real
                break
    return exact_root

def plot_function_and_roots(f, lower_bound, upper_bound, root_bis, root_nr, exact_root, a, b, c):
    """
    Plots the function f(x)= x^3 - a*x^2 + b*x - c over an extended interval and marks the roots.
    """
    # Extended interval for clarity
    x_plot = np.linspace(lower_bound - 0.5, upper_bound + 0.5, 400)
    y_plot = np.array([f(x, a, b, c) for x in x_plot])

    plt.figure(figsize=(8, 6))
    plt.plot(x_plot, y_plot, label=r'$f(x)=x^3 - {}x^2 + {}x - {}$'.format(a, b, c), color='blue')
    plt.axhline(0, color='black', linewidth=0.5)

    # Mark the roots found by the methods
    plt.plot(root_bis, f(root_bis, a, b, c), 'ro', label='Bisection root')
    plt.plot(root_nr, f(root_nr, a, b, c), 'go', label='Newton-Raphson root')

    # If an exact root was found, mark it as well
    if exact_root is not None:
        plt.plot(exact_root, f(exact_root, a, b, c), 'ks', label='Exact root')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Finding the root in the interval [{:.2f}, {:.2f}]'.format(lower_bound, upper_bound))
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # ===============================
    # Reading user input
    # ===============================
    lower_bound = float(input("Enter the lower bound of the interval x (upper bound will be x+3): "))
    upper_bound = lower_bound + 3  # interval [x, x+3]

    a = float(input("Enter coefficient a: "))
    b = float(input("Enter coefficient b: "))
    c = float(input("Enter coefficient c: "))

    # Tolerance and maximum iterations
    tol = 1e-10
    max_iter = 1000

    # ===============================
    # Calculate the exact solution (if possible)
    # ===============================
    exact_root = get_exact_root(a, b, c, lower_bound, upper_bound)
    if exact_root is None:
        print("\nCould not find a real root in the interval [{}, {}].".format(lower_bound, upper_bound))
        print("It is possible that there are no roots in this interval or they are complex.")
    else:
        print("\nExact root in the interval [{}, {}]: {:.12f}".format(lower_bound, upper_bound, exact_root))

    # ===============================
    # Compute roots using methods
    # ===============================
    try:
        # Bisection method
        root_bis, iter_bis = bisection_method(lambda x: f(x, a, b, c), lower_bound, upper_bound, tol, max_iter)
        # For Newton-Raphson method, choose the midpoint of the interval as the initial guess
        initial_guess = (lower_bound + upper_bound) / 2.0
        root_nr, iter_nr = newton_raphson_method(lambda x: f(x, a, b, c), lambda x: df(x, a, b), initial_guess, tol, max_iter)
    except ValueError as ve:
        print("\nError:", ve)
        return

    # Compute relative errors if the exact root is known
    if exact_root is not None and exact_root != 0:
        rel_error_bis = abs(root_bis - exact_root) / abs(exact_root)
        rel_error_nr = abs(root_nr - exact_root) / abs(exact_root)
    else:
        rel_error_bis = rel_error_nr = None

    # ===============================
    # Output the results
    # ===============================
    print("\nComparison of root-finding methods:")
    print("-------------------------------------------")
    print("{:<25} {:<20} {:<20}".format("Method", "Iterations", "Relative Error"))
    print("{:<25} {:<20} {:<20}".format("Bisection method", iter_bis,
                                        f"{rel_error_bis:.2e}" if rel_error_bis is not None else "N/A"))
    print("{:<25} {:<20} {:<20}".format("Newton-Raphson method", iter_nr,
                                        f"{rel_error_nr:.2e}" if rel_error_nr is not None else "N/A"))

    print("\nFound roots:")
    print("Root (Bisection): {:.12f}".format(root_bis))
    print("Root (Newton-Raphson): {:.12f}".format(root_nr))
    if exact_root is not None:
        print("Exact root: {:.12f}".format(exact_root))

    # ===============================
    # Plot the function and mark the roots
    # ===============================
    plot_function_and_roots(f, lower_bound, upper_bound, root_bis, root_nr, exact_root, a, b, c)

if __name__ == "__main__":
    main()
