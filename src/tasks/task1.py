import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal


def solve_task(n: float, a: float, b: float, c: float, axes=None):
    """
    Решает задачу:
      Вычисляет функцию f(x) = a*x^4 - b*x^2 + c на интервале [-n, n],
      ищет приближённый корень (графическим методом и методом бисекции),
      строит график функции и возвращает результаты в виде словаря.
    Параметры:
      n, a, b, c: значения (можно передавать как Decimal, str или float).
      axes: объект matplotlib.axes, на котором нужно нарисовать график.
             Если не передан, создаётся новое окно.
    Возвращает:
      dict с ключами:
         - "x_vals": массив значений x,
         - "f_vals": массив значений f(x),
         - "approx_root_graph": приближённый корень, найденный графическим методом (или None, если не найден),
         - "approx_root_bis": уточнённый корень методом бисекции (или None),
         - "abs_error": абсолютная ошибка между методами (или None),
         - "iterations": число итераций метода бисекции (или None).
    """
    # Transform input to float
    n_val = float(n)
    a_val = float(a)
    b_val = float(b)
    c_val = float(c)

    # Defining the function f(x) = a*x^4 - b*x^2 + c
    def f(x: float) -> float:
        return a_val * x**4 - b_val * x**2 + c_val

    # Creating x and f(x) values for corresponding plot
    x_vals = np.linspace(-n_val, n_val, 1000)
    f_vals = np.array([f(x) for x in x_vals])

    # Finding the approximate root using graphical method (if sign change exists)
    approx_root_graph = None
    sign_change_index = None
    for i in range(len(x_vals) - 1):
        if f_vals[i] * f_vals[i + 1] < 0:
            approx_root_graph = (x_vals[i] + x_vals[i + 1]) / 2.0
            sign_change_index = i
            break

    # If no sign change found, set other values to None
    if approx_root_graph is None:
        approx_root_bis = None
        abs_error = None
        iterations = None
    else:
        # Bisecting the interval to find the root more accurately
        x_left = x_vals[sign_change_index]
        x_right = x_vals[sign_change_index + 1]
        tol = 1e-10
        max_iter = 1000
        iterations = 0
        while (x_right - x_left) > tol and iterations < max_iter:
            x_mid = (x_left + x_right) / 2.0
            f_mid = f(x_mid)
            if abs(f_mid) < tol:
                break
            # Determine the new interval
            if f(x_left) * f_mid < 0:
                x_right = x_mid
            else:
                x_left = x_mid
            iterations += 1
        approx_root_bis = (x_left + x_right) / 2.0
        abs_error = abs(approx_root_graph - approx_root_bis)

    # Plotting the function f(x) = a*x^4 + b*x^2 + c
    if axes is None:
        fig, axes = plt.subplots(figsize=(8, 6))
    else:
        axes.clear()
    axes.plot(x_vals, f_vals, label=f"f(x) = {a_val}x⁴ + {b_val}x² + {c_val}")
    axes.axhline(0, color="black", linewidth=0.5)  # axis x
    axes.axvline(0, color="black", linewidth=0.5)  # axis y
    axes.set_xlabel("x")
    axes.set_ylabel("f(x)")
    axes.set_title(f"Plot of the graph on interval [-{n_val}, {n_val}]")
    axes.legend()
    axes.grid(True)
    if axes.figure is not None:
        axes.figure.canvas.draw_idle()

    # Returning the results
    return {
        "x_vals": x_vals,
        "f_vals": f_vals,
        "approx_root_graph": approx_root_graph,
        "approx_root_bis": approx_root_bis,
        "abs_error": abs_error,
        "iterations": iterations,
    }


def main():
    # Ввод коэффициентов и интервала
    n_input = Decimal(input("Введите значение n (интервал будет [-n, n]): "))
    a_input = Decimal(input("Введите коэффициент a (для x^4): "))
    b_input = Decimal(input("Введите коэффициент b (для x^2): "))
    c_input = Decimal(input("Введите коэффициент c (свободный член): "))

    # Преобразование в float
    n_val = float(n_input)
    a_val = float(a_input)
    b_val = float(b_input)
    c_val = float(c_input)

    # Решение задачи
    result = solve_task(n_val, a_val, b_val, c_val)

    # Вывод результатов
    if result["approx_root_graph"] is None:
        print(
            "Смена знака не обнаружена на интервале. Графический метод не смог найти корень."
        )
    else:
        print(
            "Приблизительный корень, найденный графическим методом:",
            result["approx_root_graph"],
        )
        print("Корень, найденный методом бисекции:", result["approx_root_bis"])
        print("Количество итераций в методе бисекции:", result["iterations"])
        print(
            "Абсолютная ошибка между графическим методом и методом бисекции:",
            result["abs_error"],
        )


if __name__ == "__main__":
    main()
