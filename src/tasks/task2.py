import numpy as np
import matplotlib.pyplot as plt


def f(x: float, a: float, b: float, c: float, d: float) -> float:
    """Returns f(x) = a*x^3 + b*x^2 + c*x + d."""
    return a * x**3 + b * x**2 + c * x + d


def df(x: float, a: float, b: float, c: float):
    """Returns the derivative f'(x) = 3*x^2 - 2*a*x + b."""
    return 3 * a * x**2 + 2 * b * x + c


def bisection_method(
    func: callable, n1: float, n2: float, tol: float = 1e-10, max_iter: int = 1000
) -> tuple:
    """
    Finds a root of the function 'func' in the interval [n1, n2] using the bisection method.

    Returns: list of iterations and approximate root
    """
    iterations = []
    # Check if there is a sign change at the endpoints
    if func(n1) * func(n2) > 0:
        raise ValueError(
            "No sign change at the endpoints of the interval. Bisection method is not applicable."
        )

    while (n2 - n1) > tol and len(iterations) < max_iter:
        mid = (n1 + n2) / 2.0
        f_mid = func(mid)
        iterations += [mid]
        if abs(f_mid) < tol:  # if the function value is very close to 0
            return iterations
        # Determine in which subinterval the sign change occurs
        if func(n1) * f_mid < 0:
            n2 = mid
        else:
            n1 = mid
    return iterations


def newton_raphson_method(
    func: callable, dfunc: callable, x0: float, tol=1e-10, max_iter=1000
) -> tuple:
    """
    Finds a root of the function 'func' using the Newton-Raphson method starting from initial guess x0.

    Returns:
      - approximate root
      - number of iterations
    """
    x = x0
    iterations = []
    while len(iterations) < max_iter:
        f_val = func(x)
        d_val = dfunc(x)
        if d_val == 0:
            raise ValueError("Division by zero encountered (derivative is 0).")
        x_new = x - f_val / d_val
        iterations += [x_new]
        if abs(x_new - x) < tol:
            return iterations
        x = x_new
    return iterations


def get_exact_root(
    a: float, b: float, c: float, d: float, n1: float, n2: float
) -> float:
    """
    Attempts to find a real root of the cubic equation f(x) = ax^3 + b*x^2 + c*x - d
    that lies within the interval [n1, n2] using np.roots.
    """
    roots = np.roots([a, b, c, d])
    exact_root = None
    for r in roots:
        if np.isreal(r):
            r_real = np.real(r)
            if n1 <= r_real <= n2:
                exact_root = r_real
                break
    return exact_root


def plot_function_and_roots(
    f, lower_bound, upper_bound, root_bis, root_nr, exact_root, a, b, c
):
    """
    Plots the function f(x)= x^3 - a*x^2 + b*x - c over an extended interval and marks the roots.
    """
    # Extended interval for clarity
    x_plot = np.linspace(lower_bound - 0.5, upper_bound + 0.5, 400)
    y_plot = np.array([f(x, a, b, c) for x in x_plot])

    plt.figure(figsize=(8, 6))
    plt.plot(
        x_plot,
        y_plot,
        label=r"$f(x)=x^3 - {}x^2 + {}x - {}$".format(a, b, c),
        color="blue",
    )
    plt.axhline(0, color="black", linewidth=0.5)

    # Mark the roots found by the methods
    plt.plot(root_bis, f(root_bis, a, b, c), "ro", label="Bisection root")
    plt.plot(root_nr, f(root_nr, a, b, c), "go", label="Newton-Raphson root")

    # If an exact root was found, mark it as well
    if exact_root is not None:
        plt.plot(exact_root, f(exact_root, a, b, c), "ks", label="Exact root")

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(
        "Finding the root in the interval [{:.2f}, {:.2f}]".format(
            lower_bound, upper_bound
        )
    )
    plt.legend()
    plt.grid(True)
    plt.show()


def solve_task(
    n1: float, n2: float, a: float, b: float, c: float, d: float, tol: float, axes=None
) -> dict:
    """
    Анализирует функцию f(x)= a*x^3 + b*x^2 + c*x + d на интервале [n1, n2]:
      - Находит корень методом бисекции и методом Ньютона–Рафсона (с сохранением таблиц итераций).
      - Измеряет число итераций для каждого метода.
      - Вычисляет относительные погрешности по сравнению с точным корнем (через np.roots).
      - Строит график зависимости абсолютной ошибки от номера итерации для обоих методов.

    Параметры:
      n1, n2: Концы интервала.
      a, b, c, d: Коэффициенты кубической функции.
      tol: Заданная точность.
      axes: Объект matplotlib.axes для построения графика. Если None, создаётся новый.

    Возвращает:
      dict с результатами, включая найденные корни, число итераций, ошибки и построенный график.
    """
    # Приводим входные данные к float
    n1_val = float(n1)
    n2_val = float(n2)
    a_val = float(a)
    b_val = float(b)
    c_val = float(c)
    d_val = float(d)
    tol_val = float(tol)

    # Определяем функцию и её производную с фиксированными коэффициентами
    func = lambda x: f(x, a_val, b_val, c_val, d_val)
    dfunc = lambda x: df(x, a_val, b_val, c_val)

    # Находим точный корень с помощью np.roots
    exact_root = get_exact_root(a_val, b_val, c_val, d_val, n1_val, n2_val)
    if exact_root is None:
        raise ValueError("No real roots found in the interval.")

    # Метод бисекции
    bisect_iterations = bisection_method(func, n1_val, n2_val, tol_val)
    bisect_root = bisect_iterations[-1]
    bisect_iter_count = len(bisect_iterations)

    # Метод Ньютона–Рафсона (начальное приближение – середина интервала)
    initial_guess = (n1_val + n2_val) / 2.0
    newton_iterations = newton_raphson_method(func, dfunc, initial_guess, tol_val)
    newton_root = newton_iterations[-1]
    newton_iter_count = len(newton_iterations)

    # Вычисляем относительные и абсолютные погрешности для каждого приближения
    bisect_relative_errors = [
        abs(x - exact_root) / abs(exact_root) for x in bisect_iterations
    ]
    newton_relative_errors = [
        abs(x - exact_root) / abs(exact_root) for x in newton_iterations
    ]

    bisect_absolute_errors = [abs(x - exact_root) for x in bisect_iterations]
    newton_absolute_errors = [abs(x - exact_root) for x in newton_iterations]

    # Построение графика зависимости абсолютной ошибки от номера итерации
    if axes is None:
        fig, axes = plt.subplots(figsize=(10, 6))
    else:
        axes.clear()

    iterations_bis = range(1, bisect_iter_count + 1)
    iterations_newton = range(1, newton_iter_count + 1)

    axes.plot(iterations_bis, bisect_absolute_errors, marker="o", label="Bisection")
    axes.plot(
        iterations_newton, newton_absolute_errors, marker="s", label="Newton-Raphson"
    )
    axes.set_xlabel("Iteration number")
    axes.set_ylabel("Absolute error")
    axes.set_title("Absolute error vs Iteration number")
    axes.legend()
    axes.grid(True)

    # Формируем и возвращаем словарь с результатами
    return {
        "exact_root": exact_root,
        "bisection_root": bisect_root,
        "bisection_iterations": bisect_iter_count,
        "newton_root": newton_root,
        "newton_iterations": newton_iter_count,
        "bisection_relative_errors": bisect_relative_errors,
        "newton_relative_errors": newton_relative_errors,
        "bisection_absolute_errors": bisect_absolute_errors,
        "newton_absolute_errors": newton_absolute_errors,
        "axes": axes,  # можно вернуть объект axes для дальнейшего использования
    }


def main():
    lower_bound = float(
        input("Enter the lower bound of the interval x (upper bound will be x+3): ")
    )
    upper_bound = lower_bound + 3  # interval [x, x+3]

    a = float(input("Enter coefficient a: "))
    b = float(input("Enter coefficient b: "))
    c = float(input("Enter coefficient c: "))

    # Tolerance and maximum iterations
    tol = 1e-10
    max_iter = 1000

    exact_root = get_exact_root(a, b, c, lower_bound, upper_bound)
    if exact_root is None:
        print(
            "\nCould not find a real root in the interval [{}, {}].".format(
                lower_bound, upper_bound
            )
        )
        print(
            "It is possible that there are no roots in this interval or they are complex."
        )
    else:
        print(
            "\nExact root in the interval [{}, {}]: {:.12f}".format(
                lower_bound, upper_bound, exact_root
            )
        )

    try:
        # Bisection method
        root_bis, iter_bis = bisection_method(
            lambda x: f(x, a, b, c), lower_bound, upper_bound, tol, max_iter
        )
        # For Newton-Raphson method, choose the midpoint of the interval as the initial guess
        initial_guess = (lower_bound + upper_bound) / 2.0
        root_nr, iter_nr = newton_raphson_method(
            lambda x: f(x, a, b, c), lambda x: df(x, a, b), initial_guess, tol, max_iter
        )
    except ValueError as ve:
        print("\nError:", ve)
        return

    # Compute relative errors if the exact root is known
    if exact_root is not None and exact_root != 0:
        rel_error_bis = abs(root_bis - exact_root) / abs(exact_root)
        rel_error_nr = abs(root_nr - exact_root) / abs(exact_root)
    else:
        rel_error_bis = rel_error_nr = None

    print("\nComparison of root-finding methods:")
    print("-------------------------------------------")
    print("{:<25} {:<20} {:<20}".format("Method", "Iterations", "Relative Error"))
    print(
        "{:<25} {:<20} {:<20}".format(
            "Bisection method",
            iter_bis,
            f"{rel_error_bis:.2e}" if rel_error_bis is not None else "N/A",
        )
    )
    print(
        "{:<25} {:<20} {:<20}".format(
            "Newton-Raphson method",
            iter_nr,
            f"{rel_error_nr:.2e}" if rel_error_nr is not None else "N/A",
        )
    )

    print("\nFound roots:")
    print("Root (Bisection): {:.12f}".format(root_bis))
    print("Root (Newton-Raphson): {:.12f}".format(root_nr))
    if exact_root is not None:
        print("Exact root: {:.12f}".format(exact_root))

    plot_function_and_roots(
        f, lower_bound, upper_bound, root_bis, root_nr, exact_root, a, b, c
    )


if __name__ == "__main__":
    main()
